"""Adversarial exit, execution, fee and cash-ledger tests."""

from dataclasses import replace
import unittest

import pandas as pd
from kalshi_ml.model.safety import VolatilityReference

from kalshi_ml.backtest.engine import (
    BacktestEngine, ExecutionConfig, ExitReason, MarketSnapshot, Settlement,
)


class BacktestTests(unittest.TestCase):
    """Replay known quotes independently of the synthetic pricing model."""

    def setUp(self) -> None:
        self.start = pd.Timestamp("2026-11-01T12:00:00Z")
        self.entered = self.start + pd.Timedelta(seconds=120)
        self.entry = MarketSnapshot(self.entered, self.start, 100_010, 100_000, 0.7, 54, 56, 44, 46, cycle_volatility=0.0001)
        self.engine = BacktestEngine(volatility_reference=VolatilityReference(0.0001, 0.00001, 100))

    def quote(self, bid: int, *, spot: float = 100_010, after: int = 20) -> MarketSnapshot:
        """Create a later single-book quote with exactly two cents of spread."""
        return replace(self.entry, timestamp=self.entered + pd.Timedelta(seconds=after), spot=spot,
                       yes_bid_cents=bid, yes_ask_cents=bid + 2,
                       no_bid_cents=98 - bid, no_ask_cents=100 - bid)

    def replay(self, *events: MarketSnapshot):
        return self.engine.run(events, {}, inference_not_before=self.entered)

    def test_take_profit_and_two_fees(self) -> None:
        report = self.replay(self.entry, self.quote(88))
        self.assertEqual(report.trades[0].reason, ExitReason.TAKE_PROFIT)
        self.assertEqual(report.net_pnl_cents, 88 - 56 - 4)
        self.assertEqual(report.trades[0].total_fee_cents, 4)
        self.assertEqual(report.win_percent, 100)
        self.assertEqual(report.max_drawdown_cents, 6)  # 2-cent spread + both fees
        self.assertEqual(report.equity_cents[-1] - report.equity_cents[0], report.net_pnl_cents)

    def test_hard_stop_fills_at_gapped_bid(self) -> None:
        report = self.replay(self.entry, self.quote(30))
        self.assertEqual(report.trades[0].reason, ExitReason.HARD_STOP)
        self.assertEqual(report.trades[0].exit_cents, 30)
        self.assertEqual(report.net_pnl_cents, -30)
        self.assertEqual(report.max_drawdown_cents, 30)

    def test_delta_stop_has_priority_over_hard_stop(self) -> None:
        report = self.replay(self.entry, self.quote(20, spot=100_000))
        self.assertEqual(report.trades[0].reason, ExitReason.DELTA_STOP)

    def test_no_side_delta_stop_is_symmetric(self) -> None:
        entry = replace(self.entry, spot=99_990, p_yes=0.3,
                        yes_bid_cents=44, yes_ask_cents=46, no_bid_cents=54, no_ask_cents=56)
        exit_event = replace(entry, timestamp=self.entered + pd.Timedelta(seconds=20), spot=100_000)
        report = self.replay(entry, exit_event)
        self.assertEqual(report.trades[0].side, "NO")
        self.assertEqual(report.trades[0].reason, ExitReason.DELTA_STOP)
        self.assertEqual(report.net_pnl_cents, -6)

    def test_settlement_has_entry_fee_only(self) -> None:
        settlement = Settlement(self.start, self.start + pd.Timedelta(minutes=15), True)
        report = self.engine.run([self.entry], {self.start: settlement}, inference_not_before=self.entered,
                                 end_ts=settlement.available_at)
        self.assertEqual(report.trades[0].reason, ExitReason.SETTLEMENT)
        self.assertEqual(report.net_pnl_cents, 100 - 56 - 2)

    def test_losing_no_settlement(self) -> None:
        entry = replace(self.entry, spot=99_990, p_yes=0.3,
                        yes_bid_cents=44, yes_ask_cents=46, no_bid_cents=54, no_ask_cents=56)
        settlement = Settlement(self.start, self.start + pd.Timedelta(minutes=15), True)
        report = self.engine.run([entry], {self.start: settlement}, inference_not_before=self.entered,
                                 end_ts=settlement.available_at)
        self.assertEqual(report.net_pnl_cents, -58)

    def test_unpublished_settlement_and_open_ends_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.replay(self.entry)
        settlement = Settlement(self.start, self.start + pd.Timedelta(minutes=16), True)
        with self.assertRaises(ValueError):
            self.engine.run([self.entry], {self.start: settlement}, inference_not_before=self.entered,
                            end_ts=self.start + pd.Timedelta(minutes=15))

    def test_ioc_requires_size_and_can_retry_later(self) -> None:
        no_size = replace(self.quote(30), available_contracts=0)
        report = self.replay(self.entry, no_size, self.quote(29, after=40))
        self.assertEqual(report.unfilled_exits, 1)
        self.assertEqual(report.trades[0].exit_cents, 29)

    def test_cash_and_slippage_caps_prevent_entry(self) -> None:
        for execution in (
            ExecutionConfig(initial_cash_cents=1), ExecutionConfig(initial_cash_cents=58),
            ExecutionConfig(slippage_cents=20),
        ):
            report = BacktestEngine(execution=execution, volatility_reference=self.engine.volatility_reference).run([self.entry], {}, inference_not_before=self.entered)
            self.assertEqual(len(report.trades), 0)
            self.assertEqual(report.rejected_entries, 1)

    def test_volatility_and_fee_adjusted_edge_gate_entries_but_not_exits(self) -> None:
        for event in (replace(self.entry, cycle_volatility=None), replace(self.entry, cycle_volatility=0.1),
                      replace(self.entry, p_yes=0.55)):
            report = self.replay(event)
            self.assertEqual(len(report.trades), 0)
            self.assertEqual(report.rejected_entries, 1)
        report = self.replay(self.entry, replace(self.quote(30), cycle_volatility=0.1))
        self.assertEqual(report.trades[0].reason, ExitReason.HARD_STOP)

    def test_only_one_trade_per_cycle(self) -> None:
        report = self.replay(self.entry, self.quote(88), replace(self.entry, timestamp=self.entered + pd.Timedelta(seconds=40)))
        self.assertEqual(len(report.trades), 1)

    def test_insample_unsorted_and_inconsistent_quotes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.run([self.entry], {}, inference_not_before=self.entered + pd.Timedelta(seconds=1))
        with self.assertRaises(ValueError):
            self.replay(self.quote(88), self.entry)
        with self.assertRaises(ValueError):
            replace(self.entry, no_ask_cents=50)
        with self.assertRaises(ValueError):
            self.replay(self.entry, replace(self.quote(88), strike=100_001))

    def test_report_and_no_trades(self) -> None:
        report = self.engine.run([], {}, inference_not_before=self.entered)
        self.assertEqual(report.net_pnl_cents, 0)
        self.assertEqual(report.max_drawdown_cents, 0)
        self.assertIsNone(report.profit_factor)
        for text in ("Trades:", "Win:", "Net P&L", "Profit Factor", "Max Drawdown", "Delta-SL", "Settlement"):
            self.assertIn(text, report.format_report())


if __name__ == "__main__":
    unittest.main()
