"""Offline state-machine, restart, exact accounting and stop-priority tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from kalshi_ml.config import KalshiEnvironment, RuntimeConfig, TradingMode
from kalshi_ml.daemon import DaemonConfig, Phase, StateStore, StatusConsole, TradingDaemon, state_identity
from kalshi_ml.data.collector import MarketDataCollector, MarketDataError, SpotTicker
from kalshi_ml.execution.kalshi_client import (
    ActiveMarket, Balance, FillTotals, KalshiClient, MarketLifecycle, OrderBook,
    OrderResult, OrderSubmissionUnknown,
)


class DaemonTests(unittest.TestCase):
    """All clients and prices are synthetic fixtures; no real orders are sent."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = datetime(2026, 11, 1, 12, 5, 5, tzinfo=timezone.utc)
        self.spot = Decimal("100010")
        self.yes_bid, self.no_bid = Decimal("0.54"), Decimal("0.44")
        self.yes_size = self.no_size = Decimal(1)
        self.client = Mock(spec=KalshiClient)
        self.client.environment = KalshiEnvironment.PROD
        self.client.runtime = RuntimeConfig()
        self.client.get_active_15m_market.side_effect = lambda *args: self.market()
        self.client.get_orderbook.side_effect = lambda ticker: self.book(ticker)
        self.client.get_market_lifecycle.side_effect = lambda ticker: MarketLifecycle(
            ticker, "finalized", datetime(2026, 11, 1, 12, 15, tzinfo=timezone.utc), Decimal(1), self.now)
        self.collector = Mock(spec=MarketDataCollector)
        self.collector.fetch_ticker.side_effect = lambda **kwargs: SpotTicker(self.spot, pd.Timestamp(self.now), pd.Timestamp(self.now))
        self.collector.fetch_candles.side_effect = self.candles
        self.model = Mock()
        self.model.classes_ = np.array([0, 1])
        self.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.artifact = {"model": self.model, "metadata": {"inference_not_before": "2026-11-01T00:00:00+00:00", "price_mode": "OFFICIAL_PRICES"}}
        self.store = StateStore(Path(self.directory.name) / "state.sqlite3", "test")
        self.addCleanup(self.store.close)
        self.daemon = self.make_daemon()

    def make_daemon(self, *, mode: TradingMode = TradingMode.PAPER) -> TradingDaemon:
        return TradingDaemon(self.client, self.collector, self.artifact, self.store,
                             config=DaemonConfig(mode=mode), console=StatusConsole(io.StringIO()),
                             clock=lambda: self.now, monotonic=lambda: self.now.timestamp())

    def market(self) -> ActiveMarket:
        start = self.now.replace(minute=self.now.minute // 15 * 15, second=0, microsecond=0)
        return ActiveMarket("KXBTC15M-" + start.strftime("%H%M"), "event", "active", start, start + timedelta(minutes=15),
                            start, self.now, 54, 56, 44, 46, 100000.0, "greater", Decimal(10))

    def book(self, ticker: str) -> OrderBook:
        return OrderBook(ticker, self.now, self.yes_bid, self.yes_size, self.no_bid, self.no_size)

    def candles(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({"open": 100000.0, "high": 100002.0, "low": 99998.0,
                             "close": 100000.0, "volume": 1.0},
                            index=pd.date_range(start, end, freq="min", inclusive="left"))

    def live(self) -> None:
        self.client.runtime = RuntimeConfig(mode=TradingMode.LIVE, live_trading_enabled=True)
        self.client.get_balance.return_value = Balance(Decimal(1000), Decimal(0), self.now)
        self.client.get_series_positions.side_effect = lambda *args: (
            {self.daemon.state.position.ticker: self.daemon.state.position.quantity * (1 if self.daemon.state.position.side == "YES" else -1)}
            if self.daemon.state.position else {})
        self.client.get_series_orders.return_value = []
        self.daemon = self.make_daemon(mode=TradingMode.LIVE)

    def live_ack(self, *args: object, **kwargs: object) -> OrderResult:
        # The durable intent must precede even the first network call.
        pending = self.store.load().pending
        self.assertEqual(pending.client_id, kwargs["client_order_id"])
        return OrderResult("order-id", pending.client_id, Decimal(1), Decimal(0), self.now, Decimal("0.56"), Decimal("0.02"))

    def terminal_order(self, order_id: str) -> dict[str, object]:
        pending = self.daemon.state.pending
        return {"order_id": order_id, "client_order_id": pending.client_id, "ticker": pending.ticker,
                "status": "executed", "fill_count_fp": "1.00"}

    def test_paper_entry_tp_and_no_duplicate_same_cycle(self) -> None:
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.position.quantity, 1)
        self.assertEqual(self.daemon.state.cash, Decimal("999.42"))
        self.yes_bid, self.no_bid = Decimal("0.88"), Decimal("0.10")
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("0.28"))
        self.assertEqual(self.daemon.state.cash, Decimal("1000.28"))
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.client.send_order.assert_not_called()
        self.client.get_balance.assert_not_called()
        self.assertGreaterEqual(self.model.predict_proba.call_count, 3)

    def test_delta_stop_precedes_history_and_model_failure(self) -> None:
        self.daemon.poll_once()
        self.spot = Decimal("100000")
        self.yes_bid = Decimal("0.30")
        self.collector.fetch_candles.side_effect = MarketDataError("fixture outage")
        self.daemon._candles = None
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("-0.30"))

    def test_hard_stop_and_tp_work_even_without_spot(self) -> None:
        self.daemon.poll_once()
        self.collector.fetch_ticker.side_effect = MarketDataError("fixture outage")
        self.yes_bid = Decimal("0.42")
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("-0.18"))

    def test_no_side_delta_stop(self) -> None:
        self.spot = Decimal("99990")
        self.yes_bid, self.no_bid = Decimal("0.44"), Decimal("0.54")
        self.model.predict_proba.return_value = np.array([[0.7, 0.3]])
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.position.side, "NO")
        self.spot = Decimal("100000")
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("-0.06"))

    def test_partial_paper_fills_allocate_entry_basis_once(self) -> None:
        self.no_size = Decimal("0.50")
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.position.quantity, Decimal("0.50"))
        self.yes_bid, self.no_bid, self.yes_size = Decimal("0.88"), Decimal("0.10"), Decimal("0.25")
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.position.quantity, Decimal("0.25"))
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("0.14"))

    def test_cycle_rollover_uses_official_settlement_once(self) -> None:
        self.daemon.poll_once()
        self.now += timedelta(minutes=10)
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("0.42"))
        self.assertEqual(self.daemon.state.cycle.start.minute, 15)
        self.assertFalse(self.daemon.state.unsettled)
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.realized_pnl, Decimal("0.42"))
        self.assertFalse(self.store.load().unsettled)

    def test_pending_settlement_does_not_get_inferred_from_spot(self) -> None:
        self.daemon.poll_once()
        self.now += timedelta(minutes=10)
        self.client.get_market_lifecycle.side_effect = lambda ticker: MarketLifecycle(
            ticker, "closed", self.now - timedelta(seconds=5), None, self.now)
        self.daemon.poll_once()
        self.assertIsNotNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.phase, Phase.SETTLING)
        self.assertEqual(self.daemon.state.cycle.start.minute, 15)
        self.assertEqual(self.daemon.state.realized_pnl, 0)

    def test_restart_restores_paper_position_and_prevents_duplicate_entry(self) -> None:
        self.daemon.poll_once()
        restored = self.make_daemon()
        restored.poll_once()
        self.assertEqual(restored.state.cash, Decimal("999.42"))
        self.assertEqual(restored.state.position.quantity, 1)
        self.assertTrue(restored.state.cycle.traded)

    def test_strike_cannot_change_after_lock(self) -> None:
        self.daemon.poll_once()
        self.client.get_active_15m_market.side_effect = lambda *args: replace(self.market(), strike_price=100001)
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.phase, Phase.HALTED)
        self.assertEqual(self.daemon.state.position.strike, 100000)

    def test_waiting_market_and_missing_strike_do_not_trade(self) -> None:
        self.client.get_active_15m_market.side_effect = None
        self.client.get_active_15m_market.return_value = None
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.phase, Phase.WAITING_MARKET)
        self.client.get_active_15m_market.side_effect = lambda *args: replace(self.market(), strike_price=None)
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.model.predict_proba.assert_not_called()

    def test_stale_inputs_and_entry_windows(self) -> None:
        self.now = self.now.replace(minute=14)
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.client.get_orderbook.side_effect = lambda ticker: replace(self.book(ticker), observed_at=self.now - timedelta(seconds=9))
        self.now = self.now.replace(minute=5)
        self.daemon.state.cycle = None
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)

    def test_live_fill_uses_exchange_ledger_and_durable_intent(self) -> None:
        self.live()
        self.client.send_order.side_effect = self.live_ack
        self.client.get_order.side_effect = self.terminal_order
        self.client.get_fill_totals.return_value = FillTotals(Decimal(1), Decimal("0.55"), Decimal("0.03"))
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.pending)
        self.assertEqual(self.daemon.state.position.average_entry, Decimal("0.55"))
        self.assertEqual(self.daemon.state.position.basis, Decimal("0.58"))
        self.assertEqual(self.daemon.state.cash, Decimal("999.42"))
        self.assertEqual(self.client.send_order.call_count, 1)

    def test_ambiguous_live_order_never_resubmitted(self) -> None:
        self.live()
        self.client.send_order.side_effect = lambda *a, **kw: (_ for _ in ()).throw(OrderSubmissionUnknown(kw["client_order_id"]))
        self.daemon.poll_once()
        self.assertIsNotNone(self.daemon.state.pending)
        self.daemon.poll_once()
        self.assertEqual(self.client.send_order.call_count, 1)
        self.assertIsNotNone(self.store.load().pending)
        self.assertIsNone(self.daemon.state.position)

    def test_late_fill_ledger_does_not_duplicate_accounting(self) -> None:
        self.live()
        self.client.send_order.side_effect = self.live_ack
        self.client.get_order.side_effect = self.terminal_order
        self.client.get_fill_totals.side_effect = [FillTotals(Decimal(0), Decimal(0), Decimal(0)),
                                                 FillTotals(Decimal(1), Decimal("0.56"), Decimal("0.02"))]
        self.daemon.poll_once()
        self.assertIsNotNone(self.daemon.state.pending)
        self.assertEqual(self.daemon.state.cash, 1000)
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.pending)
        self.assertEqual(self.daemon.state.cash, Decimal("999.42"))
        self.assertEqual(self.client.send_order.call_count, 1)

    def test_live_gtc_remainder_is_cancelled_before_final_accounting(self) -> None:
        self.live()
        self.client.send_order.side_effect = self.live_ack
        states = iter(("resting", "canceled"))

        def order(order_id: str) -> dict[str, object]:
            result = self.terminal_order(order_id)
            result.update(status=next(states), fill_count_fp="0.50")
            return result

        self.client.get_order.side_effect = order
        self.client.get_fill_totals.return_value = FillTotals(Decimal("0.5"), Decimal("0.28"), Decimal("0.01"))
        self.daemon.poll_once()
        self.client.cancel_order.assert_called_once()
        self.assertEqual(self.daemon.state.position.quantity, Decimal("0.5"))
        self.assertEqual(self.daemon.state.cash, Decimal("999.71"))

    def test_external_live_exposure_halts_startup(self) -> None:
        self.live()
        self.client.get_series_positions.side_effect = None
        self.client.get_series_positions.return_value = {"KXBTC15M-EXTERNAL": Decimal(1)}
        with self.assertRaises(RuntimeError):
            self.daemon.startup()
        self.assertEqual(self.daemon.state.phase, Phase.HALTED)
        self.client.send_order.assert_not_called()

    def test_state_identity_and_exclusive_process_lock(self) -> None:
        path = Path(self.directory.name) / "locked.sqlite3"
        store = StateStore(path, "first")
        try:
            with self.assertRaises(RuntimeError):
                StateStore(path, "first")
            store.save(self.daemon.state, "TEST", {}, self.now)
        finally:
            store.close()
        with self.assertRaises(ValueError):
            StateStore(path, "other-account")
        self.assertNotEqual(state_identity(TradingMode.PAPER, "key", 1, Decimal(1000), Decimal(".02")),
                            state_identity(TradingMode.LIVE, "key", 1, Decimal(1000), Decimal(".02")))

    def test_carriage_return_status_and_stop_checkpoint(self) -> None:
        output = self.daemon.console.stream
        self.daemon.run(iterations=1)
        self.assertIn("\r", output.getvalue())
        self.assertIn("P(YES)=0.700", output.getvalue())
        self.assertEqual(self.store.load().phase, Phase.STOPPED)
        self.assertIsNotNone(self.store.load().position)

    def test_model_outage_keeps_risk_management_running(self) -> None:
        self.daemon.poll_once()
        self.model.predict_proba.side_effect = RuntimeError("fixture model unavailable")
        self.daemon.poll_once()
        self.assertIsNotNone(self.daemon.state.position)
        self.assertIsNone(self.daemon.state.halt_reason)
        self.yes_bid, self.no_bid = Decimal(".88"), Decimal(".10")
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.assertEqual(self.daemon.state.realized_pnl, Decimal(".28"))

    def test_restart_identifies_ambiguous_order_without_new_submission(self) -> None:
        self.live()
        self.client.send_order.side_effect = lambda *a, **kw: (_ for _ in ()).throw(OrderSubmissionUnknown(kw["client_order_id"]))
        self.daemon.poll_once()
        pending = self.store.load().pending
        self.client.get_series_orders.side_effect = lambda *a, **kw: [] if kw.get("resting_only") else [
            {"order_id": "found-order", "client_order_id": pending.client_id, "ticker": pending.ticker},
        ]
        self.client.get_order.side_effect = self.terminal_order
        self.client.get_fill_totals.return_value = FillTotals(Decimal(1), Decimal(".56"), Decimal(".02"))
        self.daemon = self.make_daemon(mode=TradingMode.LIVE)
        self.daemon.poll_once()
        self.assertEqual(self.client.send_order.call_count, 1)
        self.assertIsNone(self.daemon.state.pending)
        self.assertEqual(self.daemon.state.cash, Decimal("999.42"))
        self.assertEqual(self.daemon.state.position.quantity, 1)

    def test_partial_live_ioc_reduces_only_confirmed_quantity(self) -> None:
        self.live()
        self.client.send_order.side_effect = self.live_ack
        self.client.get_order.side_effect = self.terminal_order
        self.client.get_fill_totals.return_value = FillTotals(Decimal(1), Decimal(".56"), Decimal(".02"))
        self.daemon.poll_once()
        self.yes_bid, self.no_bid = Decimal(".88"), Decimal(".10")

        def partial(order_id: str) -> dict[str, object]:
            row = self.terminal_order(order_id)
            row.update(status="canceled", fill_count_fp="0.25")
            return row

        self.client.get_order.side_effect = partial
        self.client.get_fill_totals.return_value = FillTotals(Decimal(".25"), Decimal(".22"), Decimal(".005"))
        self.daemon.poll_once()
        self.assertEqual(self.daemon.state.position.quantity, Decimal(".75"))
        self.assertEqual(self.daemon.state.realized_pnl, Decimal(".07"))
        self.assertTrue(self.client.send_order.call_args.args[2])

    def test_failed_initial_balance_never_persists_fictional_live_cash(self) -> None:
        self.live()
        self.client.get_balance.side_effect = RuntimeError("fixture failure")
        with self.assertRaises(RuntimeError):
            self.daemon.run(iterations=1)
        self.assertIsNone(self.store.load())

    def test_signal_prevents_new_entry_and_journal_failure_prevents_send(self) -> None:
        self.daemon.stop_event.set()
        self.daemon.poll_once()
        self.assertIsNone(self.daemon.state.position)
        self.daemon.stop_event.clear()
        self.live()
        original_save = self.store.save

        def failing_save(state: object, event: str, details: object, now: object) -> None:
            if event == "ORDER_INTENT":
                raise OSError("fixture disk full")
            original_save(state, event, details, now)

        self.store.save = failing_save
        with self.assertRaises(OSError):
            self.daemon.run(iterations=1)
        self.client.send_order.assert_not_called()
        self.assertIsNone(self.store.load().pending)

    def test_scheduler_uses_four_second_monotonic_deadlines(self) -> None:
        now = [0.0]
        delays: list[float] = []
        self.daemon.monotonic = lambda: now[0]
        self.daemon.poll_once = Mock()

        def wait(delay: float) -> None:
            delays.append(delay)
            now[0] += delay

        self.daemon.stop_event.wait = wait
        self.daemon.run(iterations=3)
        self.assertEqual(delays, [4.0, 4.0])


if __name__ == "__main__":
    unittest.main()
