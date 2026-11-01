"""Causal feature formulas, cycle boundaries and outcome separation."""

from dataclasses import replace
import math
import unittest

import numpy as np
import pandas as pd

from kalshi_ml.config import FEATURE_NAMES
from kalshi_ml.data.orderbook import snapshot_records
from kalshi_ml.data.features import (
    CyclePrices, FeatureDataError, aggregate_15m, bind_cycle_prices,
    compute_features, cycle_start,
)


class FeatureTests(unittest.TestCase):
    """Use deterministic candles and alter future information adversarially."""

    def setUp(self) -> None:
        self.start = pd.Timestamp("2026-11-01T12:00:00Z")
        self.now = self.start + pd.Timedelta(minutes=5, seconds=30)
        index = pd.date_range(self.start - pd.Timedelta(minutes=90), periods=120, freq="min")
        closes = 100_000 * np.exp(np.cumsum(np.sin(np.arange(120)) * 0.0001))
        self.frame = pd.DataFrame({
            "open": closes, "high": closes + 2, "low": closes - 2,
            "close": closes, "volume": 1.0,
        }, index=index)
        self.record = CyclePrices(self.start, 100_000.0, self.start, "kalshi:fixture")
        self.frame = bind_cycle_prices(self.frame, [self.record])
        self.frame.attrs["order_books"] = snapshot_records(pd.DataFrame([{
            "timestamp": self.now, "received_at": self.now,
            "bid_volume": 3.0, "ask_volume": 1.0, "sequence": 1, "depth": 10,
        }]))

    def test_exact_formulas_and_schema(self) -> None:
        """Match independently computed sample volatilities and momentum."""
        result = compute_features(self.frame, 100_010.0, self.now, self.start)
        self.assertEqual(tuple(result), FEATURE_NAMES)
        self.assertEqual(result["obi"], 0.5)
        self.assertEqual(result["delta_usd"], 10)
        self.assertEqual(result["remaining_sec"], 570)
        closed = self.frame.loc[self.frame.index < self.now.floor("min"), "close"].iloc[-61:]
        returns = np.diff(np.log(closed))
        expected15 = np.std(returns[-15:], ddof=1) * math.sqrt(15)
        expected60 = np.std(returns, ddof=1) * math.sqrt(60)
        self.assertAlmostEqual(result["vol_15m"], expected15)
        self.assertAlmostEqual(result["vol_60m"], expected60)
        self.assertAlmostEqual(result["vol_ratio"], expected15 / expected60)
        sigma = 100_010 * np.std(returns[-15:], ddof=1) * math.sqrt(570 / 60)
        self.assertAlmostEqual(result["z_score"], 10 / sigma)
        self.assertAlmostEqual(result["mom_1m"], 100_010 / self.frame.loc[self.start + pd.Timedelta(minutes=3), "close"] - 1)
        self.assertAlmostEqual(result["mom_3m"], 100_010 / self.frame.loc[self.start + pd.Timedelta(minutes=1), "close"] - 1)

    def test_future_and_forming_bars_cannot_change_features(self) -> None:
        """Full history and a truncated replay yield identical feature vectors."""
        expected = compute_features(self.frame, 100_010, self.now, self.start)
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.index + pd.Timedelta(minutes=1) > self.now, :] = np.nan
        self.assertEqual(compute_features(poisoned, 100_010, self.now, self.start), expected)
        truncated = self.frame.loc[self.frame.index + pd.Timedelta(minutes=1) <= self.now].copy()
        self.assertEqual(compute_features(truncated, 100_010, self.now, self.start), expected)

    def test_future_settlement_does_not_change_features(self) -> None:
        """Official results remain inaccessible before publication and unused in X."""
        settled = replace(self.record, cycle_settle=110_000, settle_available_at=self.start + pd.Timedelta(minutes=16))
        with self.assertRaises(FeatureDataError):
            settled.settlement_delta(self.now)
        self.assertEqual(settled.settlement_delta(self.start + pd.Timedelta(minutes=16)), 10_000)
        frame = bind_cycle_prices(self.frame, [settled])
        self.assertEqual(compute_features(frame, 100_010, self.now, self.start), compute_features(self.frame, 100_010, self.now, self.start))

    def test_missing_or_unobservable_strike_is_rejected(self) -> None:
        """The first Coinbase candle's open must never substitute for Kalshi strike."""
        frame = self.frame.copy()
        frame.attrs.clear()
        with self.assertRaises(FeatureDataError):
            compute_features(frame, 100_010, self.now, self.start)
        late = replace(self.record, strike_available_at=self.now + pd.Timedelta(seconds=1))
        with self.assertRaises(FeatureDataError):
            compute_features(bind_cycle_prices(frame, [late]), 100_010, self.now, self.start)

    def test_gaps_duplicates_and_delayed_receipts_rejected(self) -> None:
        """A stale window is not forward-filled, including delayed received bars."""
        missing = self.frame.drop(self.start)
        duplicated = pd.concat([self.frame, self.frame.loc[[self.start]]])
        duplicated.attrs = self.frame.attrs.copy()
        delayed = self.frame.copy()
        delayed["available_at"] = delayed.index + pd.Timedelta(minutes=1)
        delayed.loc[self.start, "available_at"] = self.now + pd.Timedelta(seconds=1)
        for frame in (missing, duplicated, delayed):
            with self.subTest(kind=len(frame)), self.assertRaises(FeatureDataError):
                compute_features(frame, 100_010, self.now, self.start)

    def test_flat_history_is_finite(self) -> None:
        """A documented one-cent floor avoids NaN and infinity in flat markets."""
        frame = self.frame.copy()
        frame.loc[:, ["open", "high", "low", "close"]] = 100_000.0
        result = compute_features(frame, 100_010, self.now, self.start)
        self.assertEqual(result["vol_15m"], 0)
        self.assertEqual(result["vol_ratio"], 1)
        self.assertEqual(result["z_score"], 1000)

    def test_cycles_are_complete_and_aligned(self) -> None:
        """Only complete quarter-hours survive aggregation, including day rollover."""
        result = aggregate_15m(self.frame, self.now)
        self.assertTrue(all(timestamp.minute in (0, 15, 30, 45) for timestamp in result.index))
        self.assertNotIn(self.start, result.index)
        self.assertTrue((result.volume == 15).all())
        self.assertEqual(cycle_start(pd.Timestamp("2026-11-02T00:02:00Z")), pd.Timestamp("2026-11-02T00:00:00Z"))
        gapped = self.frame.drop(self.start - pd.Timedelta(minutes=1))
        self.assertNotIn(self.start - pd.Timedelta(minutes=15), aggregate_15m(gapped, self.now).index)

    def test_naive_times_and_wrong_cycle_rejected(self) -> None:
        """Prevent timezone guessing and accidental cross-cycle feature rows."""
        with self.assertRaises(ValueError):
            compute_features(self.frame, 100_010, self.now.tz_localize(None), self.start)
        with self.assertRaises(FeatureDataError):
            compute_features(self.frame, 100_010, self.start + pd.Timedelta(minutes=15), self.start)


if __name__ == "__main__":
    unittest.main()
