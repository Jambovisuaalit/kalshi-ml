"""Causal entry pricing, four outcomes, exact clocks and proxy provenance."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kalshi_ml.data.features import CyclePrices
from kalshi_ml.model.dataset import TrainingDataset
from kalshi_ml.model.labeler import (
    LABEL_EXP_0, LABEL_EXP_100, LABEL_HIT_SL, LABEL_HIT_TP, OUTPUT_DTYPES,
    compute_binary_option_proxy, generate_multiclass_labels, main,
)
from kalshi_ml.model.trainer import train_model


class LabelerTests(unittest.TestCase):
    """All candle paths are hand-built synthetic research fixtures."""

    def setUp(self) -> None:
        self.start = pd.Timestamp("2026-11-01T12:00:00Z")
        self.index = pd.date_range(self.start, periods=15, freq="min")
        self.frame = pd.DataFrame({
            "timestamp": self.index.as_unit("s").asi8,
            "open": 100010., "high": 100010., "low": 100010., "close": 100010., "vol_15m": 0.001,
        }, index=pd.Index([f"row-{i}" for i in range(15)], name="original_row"))
        self.frame.loc["row-0", "open"] = 100000.
        self.frame.loc["row-0", "low"] = 100000.

    def first(self, frame: pd.DataFrame | None = None, **kwargs: object) -> pd.Series:
        return generate_multiclass_labels(self.frame if frame is None else frame, **kwargs).loc["row-1"]

    def test_proxy_units_and_no_premature_expiry(self) -> None:
        fractional = compute_binary_option_proxy(100010, 100000, 0.001, 0.5)
        dollars = compute_binary_option_proxy(100010, 100000, 100, 0.5, volatility_unit="usd")
        self.assertEqual(fractional, dollars)
        self.assertAlmostEqual(compute_binary_option_proxy(100000, 100000, 0.001, 0.5), 50)
        self.assertLess(compute_binary_option_proxy(100000.01, 100000, 0.001, 0.0001), 100)
        self.assertEqual(compute_binary_option_proxy(100001, 100000, 0, 0), 100)
        self.assertEqual(compute_binary_option_proxy(99999, 100000, 0, 0), 0)
        with self.assertRaises(ValueError):
            compute_binary_option_proxy(100000, 100000, 0.001, 0)
        self.assertEqual(compute_binary_option_proxy(100000, 100000, 0.001, 0, tie_result=False), 0)

    def test_proxy_rejects_invalid_units_or_values(self) -> None:
        for values in ((-1, 100, .01, .5), (100, 0, .01, .5), (100, 100, -.01, .5),
                       (100, 100, .01, 1.1), (100, 100, float("nan"), .5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_binary_option_proxy(*values)
        with self.assertRaises(ValueError):
            compute_binary_option_proxy(100, 100, .01, .5, volatility_unit="unknown")

    def test_close_clock_entry_window_and_original_index_preserved(self) -> None:
        result = generate_multiclass_labels(self.frame, use_delta_sl=False, hard_sl_distance_cents=99)
        self.assertEqual(result.index.tolist(), [f"row-{i}" for i in range(1, 12)])
        self.assertEqual(result.loc["row-1", "decision_ts"], self.start + pd.Timedelta(minutes=2))
        self.assertEqual(result.loc["row-1", "remaining_sec"], 780)
        self.assertEqual(result.loc["row-11", "remaining_sec"], 180)
        self.assertTrue(result.cycle_strike.eq(100000).all())
        self.assertTrue(result.label_mode.eq("SPOT_AND_QUOTE_PROXY").all())
        self.assertEqual(str(result.target.dtype), "int8")

    def test_take_profit_on_future_high(self) -> None:
        self.frame.loc["row-3", "high"] = 100400
        row = self.first()
        self.assertEqual(row.target, LABEL_HIT_TP)
        self.assertEqual(row.exit_price_proxy, 88)
        self.assertEqual(row.label_available_at, self.start + pd.Timedelta(minutes=4))

    def test_hard_stop_and_delta_stop_are_distinct_reasons(self) -> None:
        self.frame.loc["row-3", "low"] = 99900
        hard = self.first(use_delta_sl=False)
        delta = self.first()
        self.assertEqual(hard.target, LABEL_HIT_SL)
        self.assertEqual(hard.exit_reason, "Hard-SL")
        self.assertEqual(delta.exit_reason, "Delta-SL")

    def test_same_bar_collision_is_sl_first_and_flagged(self) -> None:
        self.frame.loc["row-3", ["high", "low"]] = [100400, 99900]
        row = self.first()
        self.assertEqual(row.target, LABEL_HIT_SL)
        self.assertTrue(row.ambiguous_barrier)
        self.assertEqual(row.exit_time_lower, self.start + pd.Timedelta(minutes=3))
        self.assertEqual(row.exit_time_upper, self.start + pd.Timedelta(minutes=4))

    def test_known_opening_tp_gap_precedes_later_intrabar_stop(self) -> None:
        self.frame.loc["row-3", ["open", "high", "low"]] = [100400, 100400, 99900]
        row = self.first()
        self.assertEqual(row.target, LABEL_HIT_TP)
        self.assertFalse(row.ambiguous_barrier)
        self.assertEqual(row.exit_time_lower, row.exit_time_upper)
        self.assertGreater(row.exit_price_proxy, 88)

    def test_held_settlement_win_and_loss_with_no_side_symmetry(self) -> None:
        kwargs = {"use_delta_sl": False, "hard_sl_distance_cents": 99, "tp_cents": 99}
        self.assertEqual(self.first(**kwargs).target, LABEL_EXP_100)
        self.assertEqual(self.first(side="NO", **kwargs).target, LABEL_EXP_0)
        self.frame.loc["row-14", ["low", "close"]] = 99990
        self.assertEqual(self.first(**kwargs).target, LABEL_EXP_0)
        self.assertEqual(self.first(side="NO", **kwargs).target, LABEL_EXP_100)
        self.assertEqual(self.first().target, LABEL_HIT_SL)  # default Delta-SL removes proxy EXP_0

    def test_no_side_tp_and_sl_use_opposite_extrema(self) -> None:
        self.frame.loc[:, ["open", "high", "low", "close"]] = 99990.
        self.frame.loc["row-0", ["open", "high"]] = 100000
        self.frame.loc["row-3", "low"] = 99600
        self.assertEqual(self.first(side="NO").target, LABEL_HIT_TP)
        self.frame.loc["row-3", "high"] = 100100
        row = self.first(side="NO")
        self.assertEqual(row.target, LABEL_HIT_SL)
        self.assertTrue(row.ambiguous_barrier)

    def test_entry_already_on_wrong_side_is_immediate_delta_sl(self) -> None:
        self.frame.loc["row-1", ["close", "low"]] = 99990
        row = self.first()
        self.assertEqual(row.target, LABEL_HIT_SL)
        self.assertEqual(row.exit_time_lower, row.decision_ts)
        self.assertEqual(row.label_available_at, row.decision_ts)

    def test_future_volatility_does_not_change_an_earlier_entry_or_path(self) -> None:
        self.frame.loc["row-3", "high"] = 100400
        before = self.first()
        altered = self.frame.copy()
        altered.loc[altered.index[2:], "vol_15m"] = 50
        after = self.first(altered)
        for name in ("entry_price_proxy", "volatility_at_entry", "target", "label_available_at"):
            self.assertEqual(before[name], after[name])

    def test_two_cent_spread_is_used_for_entry_ask_and_exit_bid(self) -> None:
        row = self.first()
        midpoint = compute_binary_option_proxy(100010, 100000, .001, 780 / 900)
        self.assertAlmostEqual(row.entry_price_proxy, midpoint + 1)
        self.assertEqual(row.volatility_at_entry, .001)

    def test_expiry_ties_require_explicit_policy(self) -> None:
        self.frame.loc["row-14", ["low", "close"]] = 100000
        kwargs = {"use_delta_sl": False, "hard_sl_distance_cents": 99, "tp_cents": 99}
        self.assertTrue(generate_multiclass_labels(self.frame, **kwargs).empty)
        self.assertEqual(self.first(settlement_tie="yes", **kwargs).target, LABEL_EXP_100)
        self.assertEqual(self.first(settlement_tie="no", **kwargs).target, LABEL_EXP_0)

    def test_official_strike_and_settlement_are_not_replaced_by_spot(self) -> None:
        record = CyclePrices(self.start, 100020., self.start, "kalshi:test", 100030., self.start + pd.Timedelta(minutes=16))
        row = self.first(cycle_prices={self.start: record}, as_of=self.start + pd.Timedelta(minutes=16),
                         use_delta_sl=False, hard_sl_distance_cents=99, tp_cents=99)
        self.assertEqual(row.cycle_strike, 100020)
        self.assertEqual(row.target, LABEL_EXP_100)  # final spot was 100010, below official strike
        self.assertEqual(row.label_available_at, record.settle_available_at)
        self.assertTrue(generate_multiclass_labels(self.frame, cycle_prices={}).empty)

    def test_official_strike_publication_and_delayed_settlement(self) -> None:
        record = CyclePrices(self.start, 100000., self.start + pd.Timedelta(minutes=5), "kalshi:test",
                             100010., self.start + pd.Timedelta(minutes=17))
        kwargs = {"cycle_prices": {self.start: record}, "use_delta_sl": False,
                  "hard_sl_distance_cents": 99, "tp_cents": 99}
        self.assertTrue(generate_multiclass_labels(self.frame, as_of=self.start + pd.Timedelta(minutes=16), **kwargs).empty)
        result = generate_multiclass_labels(self.frame, as_of=self.start + pd.Timedelta(minutes=17), **kwargs)
        self.assertTrue((result.decision_ts >= record.strike_available_at).all())
        self.assertNotIn("row-1", result.index)

    def test_missing_official_settlement_still_allows_earlier_tp(self) -> None:
        self.frame.loc["row-3", "high"] = 100400
        record = CyclePrices(self.start, 100000., self.start, "kalshi:test")
        self.assertEqual(self.first(cycle_prices={self.start: record}).target, LABEL_HIT_TP)

    def test_late_entry_bar_or_entry_volatility_is_excluded(self) -> None:
        self.frame["available_at"] = self.index + pd.Timedelta(minutes=1)
        self.frame.loc["row-1", "available_at"] += pd.Timedelta(nanoseconds=1)
        result = generate_multiclass_labels(self.frame)
        self.assertNotIn("row-1", result.index)
        self.frame["available_at"] = self.index + pd.Timedelta(minutes=1)
        self.frame["volatility_available_at"] = self.index + pd.Timedelta(minutes=1)
        self.frame.loc["row-1", "volatility_available_at"] += pd.Timedelta(seconds=1)
        self.assertNotIn("row-1", generate_multiclass_labels(self.frame).index)

    def test_delayed_future_bar_delays_label_availability(self) -> None:
        self.frame["available_at"] = self.index + pd.Timedelta(minutes=1)
        self.frame.loc["row-2", "available_at"] = self.start + pd.Timedelta(minutes=18)
        self.frame.loc["row-3", "high"] = 100400
        row = self.first(as_of=self.start + pd.Timedelta(minutes=18))
        self.assertEqual(row.label_available_at, self.start + pd.Timedelta(minutes=18))

    def test_empty_partial_gapped_or_not_yet_closed_cycles_have_typed_empty_result(self) -> None:
        for frame, kwargs in ((self.frame.iloc[:0], {}), (self.frame.iloc[:14], {}),
                              (self.frame.drop("row-5"), {}), (self.frame, {"as_of": self.start + pd.Timedelta(minutes=14)})):
            with self.subTest(rows=len(frame)):
                result = generate_multiclass_labels(frame, **kwargs)
                self.assertTrue(result.empty)
                self.assertTrue(set(OUTPUT_DTYPES) <= set(result.columns))
                self.assertEqual(str(result.target.dtype), "int8")

    def test_unsorted_aware_datetime_index_and_input_are_preserved(self) -> None:
        source = self.frame.copy(deep=True)
        indexed = self.frame.drop(columns="timestamp").set_axis(self.index).iloc[::-1]
        result = generate_multiclass_labels(indexed)
        self.assertEqual(result.index[0], self.index[1])
        self.assertEqual(result.decision_ts.iloc[0], self.index[1] + pd.Timedelta(minutes=1))
        generate_multiclass_labels(self.frame)
        pd.testing.assert_frame_equal(self.frame, source)

    def test_duplicate_invalid_prices_or_naive_times_are_rejected(self) -> None:
        bad = self.frame.copy()
        bad.loc["row-3", "high"] = 1
        with self.assertRaises(ValueError):
            generate_multiclass_labels(bad)
        bad = self.frame.copy()
        bad.loc["row-3", "timestamp"] = bad.loc["row-2", "timestamp"]
        with self.assertRaises(ValueError):
            generate_multiclass_labels(bad)
        with self.assertRaises(ValueError):
            generate_multiclass_labels(self.frame.drop(columns="timestamp").set_axis(self.index.tz_localize(None)))

    def test_derived_volatility_is_causal_and_in_fractional_units(self) -> None:
        index = pd.date_range(self.start - pd.Timedelta(minutes=30), periods=45, freq="min")
        close = 100000 + 10 * np.sin(np.arange(45))
        frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index)
        result = generate_multiclass_labels(frame)
        timestamp = self.start + pd.Timedelta(minutes=1)
        expected = np.std(np.diff(np.log(frame.loc[:timestamp, "close"].iloc[-16:])), ddof=1) * np.sqrt(15)
        self.assertAlmostEqual(result.loc[timestamp, "volatility_at_entry"], expected)
        with self.assertRaises(ValueError):
            generate_multiclass_labels(frame, volatility_unit="usd")

    def test_derived_volatility_retains_nanosecond_publication_precision(self) -> None:
        index = pd.date_range(self.start - pd.Timedelta(minutes=30), periods=45, freq="min")
        frame = pd.DataFrame({"open": 100010., "high": 100011., "low": 100009., "close": 100010.,
                              "available_at": index + pd.Timedelta(minutes=1)}, index=index)
        decision = self.start + pd.Timedelta(minutes=2)
        frame.loc[self.start - pd.Timedelta(minutes=1), "available_at"] = decision + pd.Timedelta(nanoseconds=1)
        result = generate_multiclass_labels(frame)
        self.assertNotIn(self.start + pd.Timedelta(minutes=1), result.index)

    def test_derived_volatility_window_cannot_bridge_missing_minutes(self) -> None:
        index = pd.date_range(self.start - pd.Timedelta(minutes=30), periods=45, freq="min")
        frame = pd.DataFrame({"open": 100010., "high": 100011., "low": 100009., "close": 100010.}, index=index)
        frame = frame.drop(self.start - pd.Timedelta(minutes=5))
        self.assertNotIn(self.start + pd.Timedelta(minutes=1), generate_multiclass_labels(frame).index)

    def test_multiclass_targets_are_rejected_by_binary_trainer_even_if_only_zero_one_present(self) -> None:
        self.frame.loc["row-3", "high"] = 100400
        labels = generate_multiclass_labels(self.frame)
        labels = labels.loc[labels.target.isin([0, 1])].copy()
        self.assertFalse(labels.empty)
        self.assertTrue(set(labels.target) <= {0, 1})
        with self.assertRaisesRegex(ValueError, "triple-barrier"):
            train_model(TrainingDataset(labels, self.frame, {}, "SPOT_PROXY"))
        restored = labels.copy()
        restored.attrs.clear()  # persisted schema column still protects CSV users
        with self.assertRaisesRegex(ValueError, "triple-barrier"):
            train_model(TrainingDataset(restored, self.frame, {}, "SPOT_PROXY"))

    def test_cli_adds_strike_audit_columns_and_replaces_output_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            source, target = Path(directory)/"candles.csv", Path(directory)/"labels.csv"
            self.frame.to_csv(source, index=False)
            with patch("sys.argv", ["labeler", "--candles-csv", str(source), "--output", str(target)]):
                self.assertEqual(main(), 0)
            result = pd.read_csv(target)
            self.assertIn("cycle_strike", result)
            self.assertTrue(result.target_schema.eq("triple-barrier-four-class-v1").all())
            self.assertFalse(list(Path(directory).glob(".labels-*")))


if __name__ == "__main__":
    unittest.main()
