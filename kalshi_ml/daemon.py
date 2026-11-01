"""Four-second, restartable paper/live supervisor for KXBTC15M.

Risk exits run before feature refresh or inference. Paper fills use production
book depth; live fills use the exchange fill ledger. SQLite commits the intent
BEFORE a mutation and commits accounting plus intent removal atomically.
No unresolved order is resubmitted. Settlement is official, never spot-inferred.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, TextIO
import uuid

import numpy as np
import pandas as pd

from kalshi_ml.config import FEATURE_NAMES, KalshiEnvironment, StrategyConfig, TradingMode
from kalshi_ml.data.collector import MarketDataCollector, MarketDataError, SpotTicker
from kalshi_ml.data.features import CyclePrices, FeatureDataError, bind_cycle_prices, compute_features, intracycle_volatility
from kalshi_ml.model.safety import VolatilityReference, net_edge_cents
from kalshi_ml.data.orderbook import MAX_AGE_SEC
from kalshi_ml.model.inference import RealtimePredictor
from kalshi_ml.monitoring.notifier import Notifier
from kalshi_ml.execution.kalshi_client import (
    APIError, ActiveMarket, AuthenticationError, FillTotals, InvalidResponseError,
    KalshiClient, KalshiError, OrderBook, OrderSubmissionUnknown, TargetSide,
    safe_to_cents,
)

from kalshi_ml.storage.journal import (
    Phase, CycleLock, Position, PendingOrder, RuntimeState, StateStore, parse_datetime,
)

LOGGER = logging.getLogger(__name__)




def utc_now() -> datetime:
    """UTC wall clock used for quote freshness and cycle boundaries."""
    return datetime.now(timezone.utc)




@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """One bounded position; paper fee is an explicit simulation assumption."""

    mode: TradingMode = TradingMode.PAPER
    poll_sec: float = 4.0
    initial_paper_cash: Decimal = Decimal("1000.00")
    paper_fee_per_contract: Decimal = Decimal("0.02")
    contracts: int = 1
    max_data_age_sec: float = 8.0
    settlement_retry_sec: float = 30.0
    series: str = "KXBTC15M"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TradingMode) or self.mode not in {TradingMode.PAPER, TradingMode.LIVE}:
            raise ValueError("daemon mode must be paper or live")
        for value in (self.poll_sec, self.max_data_age_sec, self.settlement_retry_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timing parameters must be finite and positive")
        if type(self.contracts) is not int or not 1 <= self.contracts <= 2_147_483_647:
            raise ValueError("contracts must be a positive integer")
        for value in (self.initial_paper_cash, self.paper_fee_per_contract):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError("cash and paper fee must be nonnegative Decimals")
        if not self.initial_paper_cash:
            raise ValueError("initial paper cash must be positive")












class StatusConsole:
    """One carriage-return line, cleared before complete event/log records."""

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream, self._width = stream, 0

    def status(self, message: str) -> None:
        self.stream.write("\r" + message.ljust(self._width))
        self._width = len(message)
        self.stream.flush()

    def clear(self) -> None:
        if self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0


class TradingDaemon:
    """Inject clients/model/clock for deterministic offline integration testing."""

    def __init__(
        self, client: KalshiClient, collector: MarketDataCollector, artifact: dict[str, Any],
        store: StateStore, *, config: DaemonConfig | None = None,
        strategy: StrategyConfig | None = None, console: StatusConsole | None = None,
        clock: Callable[[], datetime] = utc_now, monotonic: Callable[[], float] = time.monotonic,
        notifier: Notifier | None = None,
    ) -> None:
        self.config = config if config is not None else DaemonConfig()
        self.strategy = strategy if strategy is not None else StrategyConfig()
        if client.environment is not KalshiEnvironment.PROD:
            raise ValueError("paper and live daemon modes both require the production book")
        if client.runtime.mode is not self.config.mode:
            raise ValueError("client/daemon execution modes differ")
        self.client, self.collector, self.artifact, self.store = client, collector, artifact, store
        self._predictor = RealtimePredictor(artifact["model"], artifact["metadata"], calibrator=artifact.get("calibrator"))
        try:
            self._volatility_reference = VolatilityReference.from_metadata(artifact["metadata"])
        except (ValueError, KeyError, TypeError):
            self._volatility_reference = None
        self._entry_volatility: float | None = None
        self._last_obi_timestamp: datetime | None = None
        self.notifier = notifier
        self.console = console if console is not None else StatusConsole()
        self.clock, self.monotonic = clock, monotonic
        self.stop_event = threading.Event()
        try:
            loaded = store.load()
        except (ValueError, KeyError, TypeError, ArithmeticError):
            raise ValueError("invalid journal checkpoint; inspect it before restarting") from None
        self.state = loaded or RuntimeState(Phase.WARMING, self.config.initial_paper_cash)
        self._loaded = loaded is not None
        self._initialized = self._loaded or self.config.mode is TradingMode.PAPER
        self._candles: pd.DataFrame | None = None
        self._next_history_attempt = 0.0
        self._last_account_check = -math.inf
        self._last_problem: str | None = None
        self._last_probability: float | None = None
        self._last_spot: Decimal | None = None
        self._last_spot_timestamp: datetime | None = None
        self._inference_cutoff = parse_datetime(artifact["metadata"]["inference_not_before"])
        self._started = False
        self._journal_failed = False

    def _event(self, event: str, **details: Any) -> None:
        try:
            self.store.save(self.state, event, details, self.clock())
        except (sqlite3.Error, OSError):
            self._journal_failed = True
            raise
        self.console.clear()
        LOGGER.info("%s %s", event, json.dumps(details, default=str, sort_keys=True))
        if self.notifier is not None:
            self.notifier.submit(event, details)

    def _problem(self, code: str) -> None:
        if code != self._last_problem:
            self._event("WAIT", reason=code)
            self._last_problem = code

    def _halt(self, reason: str) -> None:
        self.state.phase, self.state.halt_reason = Phase.HALTED, reason
        self._event("HALTED", reason=reason, pending_client_id=self.state.pending.client_id if self.state.pending else None)

    def _assert_account(self) -> None:
        """Do not adopt manual/external BTC15M positions or resting orders."""
        if self.config.mode is TradingMode.PAPER:
            return
        actual = self.client.get_series_positions(self.config.series)
        expected: dict[str, Decimal] = {}
        if self.state.position is not None:
            position = self.state.position
            expected[position.ticker] = position.quantity * (1 if position.side == "YES" else -1)
        if actual != expected or self.client.get_series_orders(self.config.series, resting_only=True):
            self._halt("exchange exposure/resting orders differ from the managed journal")
            raise RuntimeError("account reconciliation failed")
        self._last_account_check = self.monotonic()

    def startup(self) -> None:
        """Recover pending work before allowing any new risk."""
        if self.state.halt_reason:
            raise RuntimeError("journal is HALTED; inspect account and journal before manual recovery")
        if self._started:
            return
        if self.config.mode is TradingMode.LIVE and not self._loaded:
            self.state.cash = self.client.get_balance().balance_dollars
            self._initialized = True
        self._event("START", mode=self.config.mode, model_price_mode=self.artifact["metadata"].get("price_mode"),
                    exchange_index=self.client.runtime.exchange_index,
                    poll_sec=self.config.poll_sec, restored=self._loaded, cash=self.state.cash,
                    strategy=asdict(self.strategy), contracts=self.config.contracts,
                    model_dataset_sha256=self.artifact["metadata"].get("dataset_sha256"))
        if self.state.pending:
            self._resolve_pending()
        if not self.state.pending:
            # A matured position may already be absent from unsettled positions.
            self._settle_due()
            if self.state.position is None or self.clock() < self.state.position.end:
                self._assert_account()
        self._started = True

    def _apply_fill(self, side: TargetSide, closing: bool, fills: FillTotals, reason: str, ticker: str) -> None:
        quantity = fills.quantity
        intent = self.state.pending
        if quantity < 0 or fills.notional < 0 or fills.fees < 0 or fills.notional > quantity:
            raise RuntimeError("invalid fill accounting")
        pnl = Decimal(0)
        if quantity:
            if closing:
                position = self.state.position
                if position is None or position.ticker != ticker or position.side != side or quantity > position.quantity:
                    raise RuntimeError("exit fills exceed the managed position")
                allocated_basis = position.basis * quantity / position.quantity
                proceeds = fills.notional - fills.fees
                pnl = proceeds - allocated_basis
                self.state.cash += proceeds
                self.state.realized_pnl += pnl
                position.quantity -= quantity
                position.basis -= allocated_basis
                if not position.quantity:
                    self.state.position = None
            else:
                cycle = self.state.cycle
                if cycle is None or cycle.ticker != ticker or cycle.strike is None or self.state.position:
                    raise RuntimeError("entry cannot be attached to its locked cycle")
                cost = fills.notional + fills.fees
                self.state.cash -= cost
                self.state.position = Position(ticker, side, quantity, fills.notional / quantity, cost, cycle.strike, cycle.end)
        self.state.pending = None
        self.state.phase = Phase.OPEN if self.state.position else Phase.FLAT
        self._event("EXIT_FILL" if closing else "ENTRY_FILL", ticker=ticker, side=side, quantity=quantity,
                    notional=fills.notional, fees=fills.fees, reason=reason, pnl_delta=pnl,
                    realized_pnl=self.state.realized_pnl, cash=self.state.cash,
                    remaining=self.state.position.quantity if self.state.position else 0,
                    average_fill_price=fills.notional / quantity if quantity else None,
                    client_order_id=intent.client_id if intent else None,
                    order_id=intent.exchange_id if intent else None)

    def _resolve_pending(self) -> None:
        """Read/cancel/read a known order, then wait for its complete fill ledger."""
        pending = self.state.pending
        if pending is None:
            return
        if pending.exchange_id is None:
            matches = [row for row in self.client.get_series_orders(self.config.series)
                       if row.get("client_order_id") == pending.client_id and row.get("ticker") == pending.ticker]
            if len(matches) != 1:
                self._problem("unresolved client order ID; no new orders until exchange history identifies it")
                return
            order_id = matches[0].get("order_id")
            if not isinstance(order_id, str) or not order_id:
                raise InvalidResponseError("pending order has no exchange ID")
            pending.exchange_id = order_id
            self._event("ORDER_IDENTIFIED", client_order_id=pending.client_id, order_id=order_id)
        row = self.client.get_order(pending.exchange_id)
        if row.get("ticker") != pending.ticker or row.get("client_order_id") != pending.client_id:
            raise InvalidResponseError("pending order identity mismatch")
        if row.get("status") == "resting":
            try:
                self.client.cancel_order(pending.exchange_id, client_order_id=pending.client_id)
            except APIError as exc:
                if exc.status_code != 404:
                    raise
            row = self.client.get_order(pending.exchange_id)
        if row.get("ticker") != pending.ticker or row.get("client_order_id") != pending.client_id:
            raise InvalidResponseError("terminal order identity mismatch")
        if row.get("status") not in {"canceled", "executed"}:
            self._problem("order not terminal; waiting for matching-engine state")
            return
        try:
            reported = Decimal(str(row["fill_count_fp"]))
        except (KeyError, ArithmeticError):
            raise InvalidResponseError("missing terminal filled quantity") from None
        if not reported.is_finite() or not 0 <= reported <= pending.count:
            raise InvalidResponseError("invalid terminal filled quantity")
        fills = self.client.get_fill_totals(pending.exchange_id, pending.ticker, pending.side)
        if fills.quantity != reported:
            self._problem("fill ledger has not caught up with terminal order")
            return
        self._apply_fill(pending.side, pending.closing, fills, pending.reason, pending.ticker)

    def _execute(self, side: TargetSide, closing: bool, book: OrderBook, reason: str) -> None:
        if self.stop_event.is_set() and not closing:
            return
        if not -2 <= (self.clock() - book.observed_at).total_seconds() <= self.config.max_data_age_sec:
            self._problem("stale orderbook")
            return
        price, size = book.quote(side, closing=closing)
        if price is None or size <= 0:
            self._problem("no executable depth for " + reason)
            return
        limit = safe_to_cents(price, rounding="floor" if closing else "ceil")
        if not 1 <= limit <= 99:
            self._problem("quote has no valid whole-cent order limit")
            return
        if not closing and not self._entry_safe(side, limit):
            return
        position = self.state.position
        count = int(position.quantity.to_integral_value(rounding=ROUND_CEILING)) if closing and position else self.config.contracts
        if self.config.mode is TradingMode.PAPER:
            quantity = min(position.quantity if closing and position else Decimal(count), size)
            fee = quantity * self.config.paper_fee_per_contract
            if not closing and self.state.cash < quantity * price + 2 * fee:
                self._problem("insufficient paper cash including both fees")
                return
            if self.state.cycle and not closing:
                self.state.cycle.traded = True
            self._apply_fill(side, closing, FillTotals(quantity, quantity * price, fee), reason, book.ticker)
            return
        if not closing:
            self._assert_account()
            available = self.client.get_balance().balance_dollars
            if available < Decimal(count) * (Decimal(limit) / 100 + Decimal("0.04")):
                self._problem("insufficient live available balance")
                return
        else:
            self._assert_account()
        if (self.clock() - book.observed_at).total_seconds() > self.config.max_data_age_sec:
            self._problem("orderbook expired before submission")
            return
        if not closing:
            cycle = self.state.cycle
            if self.stop_event.is_set():
                return
            if cycle is None or not self.strategy.entry_min_remaining_sec <= (cycle.end - self.clock()).total_seconds() <= self.strategy.entry_max_remaining_sec:
                return
            if self._last_spot_timestamp is None or (self.clock() - self._last_spot_timestamp).total_seconds() > self.config.max_data_age_sec:
                self._problem("spot expired before submission")
                return
            if not self._entry_safe(side, limit):
                return
        pending = PendingOrder(str(uuid.uuid4()), book.ticker, side, closing, count, limit, reason)
        self.state.pending = pending
        self.state.phase = Phase.CLOSING if closing else Phase.OPENING
        if not closing and self.state.cycle:
            self.state.cycle.traded = True
        self._event("ORDER_INTENT", **asdict(pending), p_yes=self._last_probability,
                    spot=self._last_spot, strike=self.state.cycle.strike if self.state.cycle else None)
        try:
            result = self.client.send_order(book.ticker, side, closing, count, limit, client_order_id=pending.client_id)
        except OrderSubmissionUnknown:
            self._event("ORDER_UNKNOWN", client_order_id=pending.client_id)
            return
        except AuthenticationError:
            self._halt("authentication failed during order submission; intent retained")
            return
        except APIError as exc:
            self.state.pending = None
            self.state.phase = Phase.OPEN if self.state.position else Phase.FLAT
            self._event("ORDER_REJECTED", status=exc.status_code, client_order_id=pending.client_id)
            return
        pending.exchange_id = result.order_id
        self._event("ORDER_ACK", order_id=result.order_id, client_order_id=pending.client_id,
                    immediate_fill=result.fill_count, remaining=result.remaining_count)
        self._resolve_pending()

    def _settle_due(self) -> None:
        now = self.clock()
        if self.state.cycle and now >= self.state.cycle.end:
            expired = self.state.cycle
            self.state.cycle = None
            if not any(cycle.ticker == expired.ticker for cycle in self.state.unsettled):
                self.state.unsettled.append(expired)
            self._event("CYCLE_ENDED", ticker=expired.ticker, settlement="pending")
        # One historical request per poll, prioritizing capital in an old position.
        pending = sorted(self.state.unsettled, key=lambda cycle: (self.state.position is None or cycle.ticker != self.state.position.ticker, cycle.end))
        for cycle in pending:
            if cycle.last_settlement_check and (now - cycle.last_settlement_check).total_seconds() < self.config.settlement_retry_sec:
                continue
            cycle.last_settlement_check = now
            result = self.client.get_market_lifecycle(cycle.ticker)
            if result.yes_payout is not None:
                position = self.state.position
                pnl = Decimal(0)
                if position and position.ticker == cycle.ticker:
                    payout = position.quantity * (result.yes_payout if position.side == "YES" else 1 - result.yes_payout)
                    pnl = payout - position.basis
                    self.state.cash += payout
                    self.state.realized_pnl += pnl
                    self.state.position = None
                self.state.unsettled.remove(cycle)
                self._event("SETTLEMENT", ticker=cycle.ticker, yes_payout=result.yes_payout,
                            pnl_delta=pnl, realized_pnl=self.state.realized_pnl, cash=self.state.cash)
            break

    def _lock_cycle(self, market: ActiveMarket) -> None:
        if market.exchange_index != self.client.runtime.exchange_index:
            self._halt("market shard differs from the account/order route")
            return
        cycle = self.state.cycle
        if cycle and cycle.ticker != market.ticker:
            self._halt("market identity changed before the locked cycle ended")
            return
        if cycle is None:
            self._last_probability = None
            self.state.cycle = CycleLock(market.ticker, market.cycle_start_ts, market.close_time,
                                        market.strike_price, market.observed_at if market.strike_price is not None else None)
            self._event("CYCLE_LOCKED", ticker=market.ticker, strike=market.strike_price,
                        start=market.cycle_start_ts, end=market.close_time)
        elif cycle.strike is None and market.strike_price is not None:
            cycle.strike, cycle.strike_available_at = market.strike_price, market.observed_at
            self._event("STRIKE_LOCKED", ticker=cycle.ticker, strike=cycle.strike)
        elif market.strike_price is not None and market.strike_price != cycle.strike:
            self._halt("official strike changed after lock")

    def _history(self) -> pd.DataFrame | None:
        end = pd.Timestamp(self.clock()).floor("min")
        if self._candles is not None and self._candles.index[-1] + pd.Timedelta(minutes=1) == end:
            return self._candles
        if self.monotonic() < self._next_history_attempt:
            return None
        self._next_history_attempt = self.monotonic() + self.config.poll_sec
        try:
            frame = self.collector.fetch_candles(end - pd.Timedelta(minutes=75), end)
        except MarketDataError:
            self._problem("completed candle window unavailable")
            return None
        frame["available_at"] = pd.Timestamp(self.clock())
        self._candles = frame
        return frame

    def _risk_exit(self, book: OrderBook, spot: SpotTicker | None) -> None:
        position = self.state.position
        if position is None:
            return
        bid, _ = book.quote(position.side, closing=True)
        reason = None
        if spot and -2 <= (self.clock() - spot.timestamp.to_pydatetime(warn=False)).total_seconds() <= self.config.max_data_age_sec:
            delta = float(spot.price) - position.strike
            if position.side == "YES" and delta <= 0 or position.side == "NO" and delta >= 0:
                reason = "Delta-SL"
        if reason is None and bid is not None:
            if position.average_entry - bid >= Decimal(self.strategy.hard_stop_cents) / 100:
                reason = "Hard-SL"
            elif bid >= Decimal(self.strategy.take_profit_cents) / 100:
                reason = "TP"
        if reason:
            self._execute(position.side, True, book, reason)

    def _entry_safe(self, side: TargetSide, limit_cents: int) -> bool:
        """Recheck OBI/spot freshness, two-sigma band and rounded-price EV.

        This gate applies only to new risk; TP/SL/reduce-only exits bypass it.
        Recheck after live balance/account GETs, immediately before intent.
        """
        now = self.clock()
        ages = ((self._last_obi_timestamp, min(MAX_AGE_SEC, self.config.max_data_age_sec)),
                (self._last_spot_timestamp, self.config.max_data_age_sec))
        if any(timestamp is None or not 0 <= (now - timestamp).total_seconds() <= limit
               for timestamp, limit in ages):
            self._problem("entry blocked: stale/missing OBI or spot")
            return False
        if (self._volatility_reference is None or self._entry_volatility is None
                or not self._volatility_reference.allows(self._entry_volatility, float(self.strategy.volatility_sigma_limit))):
            self._problem("entry blocked: intracycle volatility outside training two-sigma band")
            return False
        if self._last_probability is None:
            self._problem("entry blocked: calibrated probability unavailable")
            return False
        edge = net_edge_cents(self._last_probability, side, Decimal(limit_cents) / 100, self.strategy)
        paper_budget = 200 * self.config.paper_fee_per_contract
        if self.config.mode is TradingMode.PAPER and paper_budget > self.strategy.entry_fee_buffer_cents + self.strategy.exit_fee_buffer_cents:
            edge -= paper_budget - self.strategy.entry_fee_buffer_cents - self.strategy.exit_fee_buffer_cents
        if edge < self.strategy.minimum_edge_cents:
            self._problem("entry blocked: insufficient calibrated edge after price and fee budgets")
            return False
        return True

    def poll_once(self) -> None:
        """One supervised poll; a pending mutation blocks all further submissions."""
        if not self._started:
            self.startup()
        if self.state.halt_reason:
            return
        self._last_probability = None
        self._entry_volatility = None
        self._last_obi_timestamp = None
        self._last_spot = None
        if self.state.pending:
            self._resolve_pending()
            if self.state.pending:
                return
        self._settle_due()
        spot = None
        try:
            spot = self.collector.fetch_ticker(max_age_sec=self.config.max_data_age_sec)
            self._last_spot = spot.price
            self._last_spot_timestamp = spot.timestamp.to_pydatetime(warn=False)
        except MarketDataError:
            self._problem("spot unavailable; bid-based exits still enabled")
        book = None
        position = self.state.position
        if position:
            if self.clock() >= position.end:
                self.state.phase = Phase.SETTLING
            else:
                try:
                    if self.monotonic() - self._last_account_check >= 30:
                        self._assert_account()
                    book = self.client.get_orderbook(position.ticker)
                    self._risk_exit(book, spot)
                except KalshiError:
                    self._problem("position book/order unavailable; position retained")
                    return
            if self.state.pending or self.state.halt_reason:
                return
        market = self.client.get_active_15m_market(self.config.series)
        if market is None:
            self.state.phase = (Phase.SETTLING if self.clock() >= self.state.position.end else Phase.OPEN) if self.state.position else Phase.WAITING_MARKET
            self._problem(f"no active market on configured shard {self.client.runtime.exchange_index}")
            return
        self._lock_cycle(market)
        cycle = self.state.cycle
        if self.state.halt_reason or cycle is None:
            return
        if book is None or book.ticker != market.ticker:
            book = self.client.get_orderbook(market.ticker)
        self.state.phase = (Phase.SETTLING if self.clock() >= self.state.position.end else Phase.OPEN) if self.state.position else Phase.FLAT
        if spot is None or cycle.strike is None or cycle.strike_available_at is None:
            self._problem("waiting for observable official strike and spot")
            return
        history = self._history()
        if history is None:
            if not self.state.position:
                self.state.phase = Phase.WARMING
            return
        now = self.clock()
        if now < self._inference_cutoff or not cycle.start <= now < cycle.end:
            self._problem("model/cycle time boundary prevents inference")
            return
        if not -2 <= (now - spot.timestamp.to_pydatetime(warn=False)).total_seconds() <= self.config.max_data_age_sec:
            self._problem("spot expired before inference")
            return
        record = CyclePrices(pd.Timestamp(cycle.start), cycle.strike, pd.Timestamp(cycle.strike_available_at), f"kalshi:{cycle.ticker}")
        try:
            obi = self.collector.fetch_order_book(max_age_sec=min(MAX_AGE_SEC, self.config.max_data_age_sec))
            now = self.clock()
            self._last_obi_timestamp = min(obi.timestamp, obi.received_at).to_pydatetime(warn=False)
            if not 0 <= (now - spot.timestamp.to_pydatetime(warn=False)).total_seconds() <= self.config.max_data_age_sec:
                raise FeatureDataError("spot expired while fetching Coinbase OBI")
            features = compute_features(bind_cycle_prices(history, [record]), float(spot.price), pd.Timestamp(now), pd.Timestamp(cycle.start), order_book_imbalance=obi.imbalance)
            self._entry_volatility = intracycle_volatility(history, float(spot.price), pd.Timestamp(now), pd.Timestamp(cycle.start))
            probability = self._predictor.predict_yes(features, as_of=now)
        except (ValueError, RuntimeError, IndexError, TypeError, AttributeError, MarketDataError):
            self._last_probability = None
            self._problem("OBI/features/calibrated inference unavailable; risk exits remain enabled")
            return
        self._last_probability = probability
        if self.state.position or cycle.traded:
            return
        side: TargetSide | None = None
        self._last_problem = None
        delta = features["delta_usd"]
        if probability >= float(self.strategy.yes_probability_min) and delta >= float(self.strategy.minimum_delta_usd):
            side = "YES"
        elif probability <= float(self.strategy.no_probability_max) and delta <= -float(self.strategy.minimum_delta_usd):
            side = "NO"
        if side:
            ask, _ = book.quote(side, closing=False)
            if ask is not None and self.strategy.entry_min_cents <= ask * 100 <= self.strategy.entry_max_cents:
                now = self.clock()
                remaining = (cycle.end - now).total_seconds()
                if (now - book.observed_at).total_seconds() <= self.config.max_data_age_sec and self.strategy.entry_min_remaining_sec <= remaining <= self.strategy.entry_max_remaining_sec:
                    self._execute(side, False, book, "SIGNAL")
    def _render(self) -> None:
        cycle = self.state.cycle
        remaining = max(0, int((cycle.end - self.clock()).total_seconds())) if cycle else 0
        position = self.state.position
        probability = f"{self._last_probability:.3f}" if self._last_probability is not None else "---"
        self.console.status(
            f"{self.clock():%H:%M:%S}Z {self.config.mode.value.upper()} {self.state.phase.value} "
            f"{cycle.ticker if cycle else '-'} left={remaining:3d}s spot={self._last_spot or '-'} "
            f"P(YES)={probability} pos={position.side + ':' + str(position.quantity) if position else '-'} "
            f"cash=${self.state.cash:.2f} realized=${self.state.realized_pnl:+.2f}"
        )

    def run(self, *, iterations: int | None = None) -> None:
        """Poll on monotonic four-second deadlines; skip missed deadlines, no bursts.

        Network timeouts/retries can exceed four seconds. This is a REST polling
        supervisor, not a guarantee of sub-second stop execution.
        """
        if iterations is not None and iterations < 1:
            raise ValueError("iterations must be positive")
        deadline, count = self.monotonic(), 0
        try:
            self.startup()
            while not self.stop_event.is_set() and not self.state.halt_reason:
                try:
                    self.poll_once()
                except AuthenticationError:
                    self._halt("authentication/permissions unavailable")
                except (KalshiError, MarketDataError, FeatureDataError) as exc:
                    self._problem(type(exc).__name__)
                except (ValueError, RuntimeError) as exc:
                    if not self.state.halt_reason:
                        self._halt(type(exc).__name__)
                self._render()
                count += 1
                if iterations is not None and count >= iterations:
                    break
                deadline += self.config.poll_sec
                if deadline <= self.monotonic():
                    deadline = self.monotonic() + self.config.poll_sec
                self.stop_event.wait(max(0.0, deadline - self.monotonic()))
        finally:
            self.console.clear()
            if not self._journal_failed and self.config.mode is TradingMode.LIVE and self.state.pending:
                try:
                    # Resolve/cancel existing intents only; never submit a new
                    # entry or flatten an unrelated position during shutdown.
                    self._resolve_pending()
                except (KalshiError, ValueError, RuntimeError) as exc:
                    self._event("SHUTDOWN_PENDING", error=type(exc).__name__,
                                client_order_id=self.state.pending.client_id if self.state.pending else None)
            if not self.state.halt_reason:
                self.state.phase = Phase.STOPPED
            if not self._journal_failed and self._initialized:
                self._event("STOP", open_position=asdict(self.state.position) if self.state.position else None,
                            pending=asdict(self.state.pending) if self.state.pending else None,
                            cash=self.state.cash, realized_pnl=self.state.realized_pnl)
            elif self._journal_failed:
                LOGGER.critical("Journal write failed; last durable intent retained for restart reconciliation")


def state_identity(mode: TradingMode, key_id: str, contracts: int, paper_cash: Decimal, paper_fee: Decimal,
                   *, exchange_index: int = 2) -> str:
    """Bind a journal to mode/account/sizing; no API key identifier is stored raw."""
    encoded = json.dumps([mode.value, "PROD", 0, exchange_index, key_id if mode is TradingMode.LIVE else "public",
                          contracts, str(paper_cash.normalize()), str(paper_fee.normalize())])
    return hashlib.sha256(encoded.encode()).hexdigest()
