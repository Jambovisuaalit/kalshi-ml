"""Event-driven BTC15M backtest with integer-cent accounting.

Actual quote snapshots can be replayed via BacktestEngine.run. The optional
OHLC CLI replay is a SPOT/QUOTE PROXY scenario, not observed Kalshi execution:
it synthesizes independent normal-CDF midpoint prices, a two-cent spread and
two assumed intraminute paths. Queue priority, latency and depth are not
recoverable from spot OHLCV. A displayed marketable entry is assumed filled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalshi_ml.config import FEATURE_NAMES, StrategyConfig, load_settings
from kalshi_ml.data.collector import utc_timestamp
from kalshi_ml.data.features import CYCLE, MINUTE, FeatureDataError, bind_cycle_prices, compute_features, cycle_start
from kalshi_ml.model.dataset import TrainingDataset


class ExitReason(StrEnum):
    """Mutually exclusive reason recorded for each completed position."""

    TAKE_PROFIT = "TP"
    HARD_STOP = "Hard-SL"
    DELTA_STOP = "Delta-SL"
    SETTLEMENT = "Settlement"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """User-specified fee scenario; this is not a claim about live Kalshi fees.

    Two cents per executed contract per side: round-trip fee is four cents.
    Settlement pays 0/100 cents without a second order or fee. Extra slippage
    is adverse to every order; executable bid/ask already includes spread.
    One position and at most one entry per cycle, with no leverage.
    """

    fee_cents: int = 2
    slippage_cents: int = 0
    contracts: int = 1
    initial_cash_cents: int = 100_000

    def __post_init__(self) -> None:
        for name in ("fee_cents", "slippage_cents", "contracts", "initial_cash_cents"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.contracts < 1 or self.initial_cash_cents < 1:
            raise ValueError("contracts and initial cash must be positive")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Quotes/spot/probability all observable at timestamp, for one active cycle.

    available_contracts is a conservative displayed size shared by both sides.
    The caller must align feeds without forward filling future observations.
    p_yes must be from an OOS model; run() validates time against model cutoff.
    """

    timestamp: pd.Timestamp
    cycle_start_ts: pd.Timestamp
    spot: float
    strike: float
    p_yes: float
    yes_bid_cents: int
    yes_ask_cents: int
    no_bid_cents: int
    no_ask_cents: int
    available_contracts: int = 1

    def __post_init__(self) -> None:
        timestamp, start = utc_timestamp(self.timestamp), utc_timestamp(self.cycle_start_ts)
        if cycle_start(timestamp) != start:
            raise ValueError("snapshot cycle does not match timestamp")
        if not all(math.isfinite(value) for value in (self.spot, self.strike, self.p_yes)):
            raise ValueError("snapshot values must be finite")
        if self.spot <= 0 or self.strike <= 0 or not 0 <= self.p_yes <= 1:
            raise ValueError("invalid spot, strike or probability")
        for bid, ask in ((self.yes_bid_cents, self.yes_ask_cents), (self.no_bid_cents, self.no_ask_cents)):
            if type(bid) is not int or type(ask) is not int or not 0 <= bid <= ask <= 100:
                raise ValueError("quotes must be integer cents in [0,100] with bid <= ask")
        if self.yes_bid_cents != 100 - self.no_ask_cents or self.no_bid_cents != 100 - self.yes_ask_cents:
            raise ValueError("single-book YES/NO quotes must be complementary")
        if type(self.available_contracts) is not int or self.available_contracts < 0:
            raise ValueError("available_contracts must be a nonnegative integer")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "cycle_start_ts", start)


@dataclass(frozen=True, slots=True)
class Settlement:
    """Actual binary payout and its publication time, supplied outside signals."""

    cycle_start_ts: pd.Timestamp
    available_at: pd.Timestamp
    yes_won: bool

    def __post_init__(self) -> None:
        start, available = utc_timestamp(self.cycle_start_ts), utc_timestamp(self.available_at)
        if cycle_start(start) != start or available < start + CYCLE:
            raise ValueError("invalid settlement time")
        if type(self.yes_won) is not bool:
            raise TypeError("yes_won must be a boolean")
        object.__setattr__(self, "cycle_start_ts", start)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, slots=True)
class Trade:
    """Completed fill pair or entry plus settlement; prices/fees are per contract."""

    cycle_start_ts: pd.Timestamp
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_cents: int
    exit_cents: int
    contracts: int
    total_fee_cents: int
    net_pnl_cents: int
    reason: ExitReason


@dataclass(frozen=True)
class BacktestReport:
    """Dollar report including marked-to-bid liquidation equity drawdown."""

    trades: tuple[Trade, ...]
    equity_cents: tuple[int, ...]
    scenario: str
    rejected_entries: int = 0
    unfilled_exits: int = 0

    @property
    def net_pnl_cents(self) -> int:
        return sum(trade.net_pnl_cents for trade in self.trades)

    @property
    def win_percent(self) -> float:
        return 100 * sum(trade.net_pnl_cents > 0 for trade in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float | None:
        wins = sum(max(0, trade.net_pnl_cents) for trade in self.trades)
        losses = -sum(min(0, trade.net_pnl_cents) for trade in self.trades)
        return wins / losses if losses else (math.inf if wins else None)

    @property
    def max_drawdown_cents(self) -> int:
        peak, maximum = self.equity_cents[0], 0
        for value in self.equity_cents:
            peak = max(peak, value)
            maximum = max(maximum, peak - value)
        return maximum

    def format_report(self) -> str:
        """Render requested performance statistics and net results by exit reason."""
        factor = self.profit_factor
        text = "N/A" if factor is None else ("inf" if math.isinf(factor) else f"{factor:.3f}")
        counts = Counter(trade.reason for trade in self.trades)
        lines = [
            f"=== BTC15M BACKTEST | {self.scenario} ===",
            f"Trades: {len(self.trades)} | Win: {self.win_percent:.2f}%",
            f"Net P&L: ${self.net_pnl_cents / 100:.2f} | Profit Factor: {text}",
            f"Max Drawdown (mark-to-bid, exit fee included): ${self.max_drawdown_cents / 100:.2f}",
            "Exit reason       Trades       Net P&L",
        ]
        for reason in ExitReason:
            pnl = sum(trade.net_pnl_cents for trade in self.trades if trade.reason is reason) / 100
            lines.append(f"{reason.value:<17} {counts[reason]:>6}      ${pnl:>9.2f}")
        lines.append(f"Rejected entries: {self.rejected_entries} | Unfilled IOC exits: {self.unfilled_exits}")
        return "\n".join(lines)


@dataclass(slots=True)
class _Position:
    cycle: pd.Timestamp
    side: str
    entered_at: pd.Timestamp
    entry_cents: int


class BacktestEngine:
    """Sequential fills at observed quotes; no minute-high limit-fill shortcut.

    Exit priority is Delta-SL, Hard-SL, TP when triggers coincide. A price gap
    executes at the actual bid, not the stop threshold. An IOC with too little
    displayed size is unfilled and may be retried at the next snapshot. Entry
    GTC orders are simulated only when immediately marketable; no passive
    queue fill is invented. End-of-data open positions are an explicit error.
    """

    def __init__(self, strategy: StrategyConfig | None = None, execution: ExecutionConfig | None = None) -> None:
        self.strategy = strategy if strategy is not None else StrategyConfig()
        self.execution = execution if execution is not None else ExecutionConfig()

    def run(
        self,
        snapshots: Sequence[MarketSnapshot],
        settlements: Mapping[pd.Timestamp, Settlement],
        *,
        inference_not_before: pd.Timestamp,
        end_ts: pd.Timestamp | None = None,
        scenario: str = "historical quotes",
    ) -> BacktestReport:
        """Replay strictly increasing snapshots, settling only published outcomes.

        Publication cash is processed before the snapshot at that timestamp.
        A matured, unpublished position is conservatively marked at zero until
        the payout becomes available. Cash limits reserve both entry and exit
        fees so even a zero-price exit cannot create a negative cash balance.
        """
        cutoff = utc_timestamp(inference_not_before)
        if not snapshots:
            return BacktestReport((), (self.execution.initial_cash_cents,), scenario)
        if any(b.timestamp <= a.timestamp for a, b in zip(snapshots, snapshots[1:])):
            raise ValueError("snapshots must have unique, strictly increasing timestamps")
        if snapshots[0].timestamp < cutoff:
            raise ValueError("in-sample replay is forbidden: snapshot precedes model fit cutoff")
        horizon = utc_timestamp(end_ts) if end_ts is not None else snapshots[-1].timestamp
        if horizon < snapshots[-1].timestamp:
            raise ValueError("end_ts precedes the last snapshot")
        for key, settlement in settlements.items():
            if key != settlement.cycle_start_ts:
                raise ValueError("settlement key/cycle mismatch")
        cash = self.execution.initial_cash_cents
        quantity, fee = self.execution.contracts, self.execution.fee_cents
        slippage = self.execution.slippage_cents
        position: _Position | None = None
        trades: list[Trade] = []
        equity = [cash]
        entered_cycles: set[pd.Timestamp] = set()
        strikes: dict[pd.Timestamp, float] = {}
        rejected, unfilled = 0, 0

        def finish(timestamp: pd.Timestamp, exit_price: int, reason: ExitReason) -> None:
            nonlocal cash, position
            assert position is not None
            exit_fee = 0 if reason is ExitReason.SETTLEMENT else fee
            cash += quantity * (exit_price - exit_fee)
            fees = quantity * (fee + exit_fee)
            trades.append(Trade(
                position.cycle, position.side, position.entered_at, timestamp,
                position.entry_cents, exit_price, quantity, fees,
                quantity * (exit_price - position.entry_cents) - fees, reason,
            ))
            position = None
            equity.append(cash)

        def settle_due(timestamp: pd.Timestamp) -> None:
            if position is None:
                return
            settlement = settlements.get(position.cycle)
            if settlement is not None and settlement.available_at <= timestamp:
                won = settlement.yes_won == (position.side == "YES")
                finish(settlement.available_at, 100 if won else 0, ExitReason.SETTLEMENT)

        for snapshot in snapshots:
            known_strike = strikes.setdefault(snapshot.cycle_start_ts, snapshot.strike)
            if known_strike != snapshot.strike:
                raise ValueError("strike changed within a contract cycle")
            settle_due(snapshot.timestamp)
            if position is not None:
                if position.cycle == snapshot.cycle_start_ts:
                    bid = snapshot.yes_bid_cents if position.side == "YES" else snapshot.no_bid_cents
                    delta = snapshot.spot - snapshot.strike
                    reason: ExitReason | None = None
                    if (position.side == "YES" and delta <= 0) or (position.side == "NO" and delta >= 0):
                        reason = ExitReason.DELTA_STOP
                    elif position.entry_cents - bid >= self.strategy.hard_stop_cents:
                        reason = ExitReason.HARD_STOP
                    elif bid >= self.strategy.take_profit_cents:
                        reason = ExitReason.TAKE_PROFIT
                    if reason is not None:
                        if snapshot.available_contracts >= quantity:
                            finish(snapshot.timestamp, max(0, bid - slippage), reason)
                        else:
                            unfilled += 1
                    if position is not None:
                        equity.append(cash + quantity * (max(0, bid - slippage) - fee))
                else:
                    # Position is matured but settlement has not been published.
                    equity.append(cash)
            if position is None and snapshot.cycle_start_ts not in entered_cycles:
                remaining = (snapshot.cycle_start_ts + CYCLE - snapshot.timestamp).total_seconds()
                delta = snapshot.spot - snapshot.strike
                side: str | None = None
                ask = 0
                if self.strategy.entry_min_remaining_sec <= remaining <= self.strategy.entry_max_remaining_sec:
                    if snapshot.p_yes >= float(self.strategy.yes_probability_min) and delta >= float(self.strategy.minimum_delta_usd):
                        side, ask = "YES", snapshot.yes_ask_cents
                    elif snapshot.p_yes <= float(self.strategy.no_probability_max) and delta <= -float(self.strategy.minimum_delta_usd):
                        side, ask = "NO", snapshot.no_ask_cents
                if side is not None and self.strategy.entry_min_cents <= ask <= self.strategy.entry_max_cents:
                    fill = min(100, ask + slippage)
                    if fill > self.strategy.entry_max_cents or snapshot.available_contracts < quantity or cash < quantity * (fill + 2 * fee):
                        rejected += 1
                    else:
                        cash -= quantity * (fill + fee)
                        position = _Position(snapshot.cycle_start_ts, side, snapshot.timestamp, fill)
                        entered_cycles.add(snapshot.cycle_start_ts)
                        bid = snapshot.yes_bid_cents if side == "YES" else snapshot.no_bid_cents
                        equity.append(cash + quantity * (max(0, bid - slippage) - fee))
        settle_due(horizon)
        if position is not None:
            raise ValueError("open position at end of replay: provide remaining quotes or published settlement")
        if cash - self.execution.initial_cash_cents != sum(trade.net_pnl_cents for trade in trades):
            raise AssertionError("cash ledger and trade P&L do not reconcile")
        equity.append(cash)
        return BacktestReport(tuple(trades), tuple(equity), scenario, rejected, unfilled)


def synthetic_snapshots(
    dataset: TrainingDataset, artifact: dict[str, Any], *, path: str,
) -> tuple[list[MarketSnapshot], dict[pd.Timestamp, Settlement]]:
    """Construct a documented OHLC sensitivity scenario, not factual tick data.

    Path is open-low-high-close or open-high-low-close at offsets
    0/20/40/60 seconds. At shared minute boundaries the previous close takes
    priority over the next open. A completed minute's close is used only at its
    actual close. Midpoint = NormalCDF(z_score), independent of the trained
    predictor. Clip midpoint to 1..99 and quote midpoint +/-1 cent, yielding
    exactly a 2-cent YES and NO spread. Assumed size is one contract.
    """
    if path not in {"OLHC", "OHLC"}:
        raise ValueError("path must be OLHC or OHLC")
    metadata = artifact["metadata"]
    if metadata["price_mode"] != dataset.price_mode:
        raise ValueError("model and replay strike/outcome price modes must match")
    cutoff = utc_timestamp(pd.Timestamp(metadata["inference_not_before"]))
    horizon = dataset.candles.index[-1] + MINUTE
    settlements: dict[pd.Timestamp, Settlement] = {}
    for start, record in dataset.cycles.items():
        if record.cycle_settle is not None and record.settle_available_at is not None and record.settle_available_at <= horizon:
            if record.cycle_settle != record.strike_open:
                settlements[start] = Settlement(start, record.settle_available_at, bool(record.cycle_settle > record.strike_open))
    feature_rows: list[dict[str, float]] = []
    event_rows: list[tuple[pd.Timestamp, pd.Timestamp, float, float]] = []
    for timestamp, candle in dataset.candles.iterrows():
        if timestamp + MINUTE <= cutoff:
            continue
        high_low = (float(candle.low), float(candle.high)) if path == "OLHC" else (float(candle.high), float(candle.low))
        for offset, spot in zip((0, 20, 40, 60), (float(candle.open), *high_low, float(candle.close))):
            now = timestamp + pd.Timedelta(seconds=offset)
            start = cycle_start(now)
            record = dataset.cycles.get(start)
            if now < cutoff or start not in settlements or record is None:
                continue
            if event_rows and now == event_rows[-1][0]:
                continue
            history = dataset.candles.loc[(dataset.candles.index >= now.floor("min") - 61 * MINUTE) & (dataset.candles.index < now.floor("min"))]
            try:
                features = compute_features(bind_cycle_prices(history, [record]), spot, now, start)
            except FeatureDataError:
                continue
            feature_rows.append(features)
            event_rows.append((now, start, spot, record.strike_open))
    if not feature_rows:
        return [], settlements
    matrix = pd.DataFrame(feature_rows, columns=list(FEATURE_NAMES))
    model = artifact["model"]
    probabilities = model.predict_proba(matrix)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    snapshots: list[MarketSnapshot] = []
    for row, features, probability in zip(event_rows, feature_rows, probabilities):
        midpoint = max(1, min(99, int(math.floor(100 * 0.5 * (1 + math.erf(features["z_score"] / math.sqrt(2))) + 0.5))))
        bid, ask = midpoint - 1, midpoint + 1
        snapshots.append(MarketSnapshot(*row, float(probability), bid, ask, 100 - ask, 100 - bid))
    return snapshots, settlements


def replay_synthetic(
    dataset: TrainingDataset, artifact: dict[str, Any], *, strategy: StrategyConfig | None = None,
) -> list[BacktestReport]:
    """Replay BOTH assumed OHLC paths; do not select the more profitable one."""
    cutoff = utc_timestamp(pd.Timestamp(artifact["metadata"]["inference_not_before"]))
    reports: list[BacktestReport] = []
    for path in ("OLHC", "OHLC"):
        snapshots, settlements = synthetic_snapshots(dataset, artifact, path=path)
        if not snapshots:
            raise ValueError("no usable OOS snapshots: check data horizon, model cutoff and cycle records")
        reports.append(BacktestEngine(strategy).run(
            snapshots, settlements, inference_not_before=cutoff,
            end_ts=dataset.candles.index[-1] + MINUTE,
            scenario=f"SYNTHETIC QUOTES / {dataset.price_mode} / path={path}; fee=2c/fill, spread=2c",
        ))
    return reports


def main() -> int:
    """CLI replay using a trusted local model and newly fetched or local candles."""
    from kalshi_ml.model.trainer import DEFAULT_MODEL_PATH, add_data_arguments, dataset_from_arguments, load_artifact

    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    try:
        artifact = load_artifact(args.model)
        dataset = dataset_from_arguments(args)
        strategy = load_settings(args.env_file).strategy
        for report in replay_synthetic(dataset, artifact, strategy=strategy):
            print(report.format_report())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Backtest failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
