"""Chronological leakage checks, metric edge cases and artifact round trips."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from kalshi_ml.backtest.engine import replay_synthetic, synthetic_snapshots
from kalshi_ml.config import FEATURE_NAMES
from kalshi_ml.data.features import MINUTE
from kalshi_ml.model.dataset import TrainingDataset, build_dataset, load_candles, load_cycle_prices
from kalshi_ml.model.trainer import (
    chronological_split, evaluate, load_artifact, save_artifact, train_model,
)


def fixture_dataset(minutes: int = 360) -> TrainingDataset:
    """Generate oscillating minute data with alternating cycle outcomes."""
    index = pd.date_range("2026-11-01T00:00:00Z", periods=minutes, freq="min")
    close = 100_000 + 50 * np.sin(np.arange(minutes) * np.pi / 15)
    opened = np.r_[close[0] - 1, close[:-1]]
    candles = pd.DataFrame({
        "open": opened, "high": np.maximum(opened, close) + 4,
        "low": np.minimum(opened, close) - 4, "close": close, "volume": 1.0,
    }, index=index)
    candles.attrs["source"] = "test-fixture"
    books = pd.DataFrame({
        "timestamp": index + MINUTE, "received_at": index + MINUTE,
        "bid_volume": 2.0 + np.sin(np.arange(minutes)), "ask_volume": 2.0,
        "sequence": np.arange(minutes), "depth": 10,
    })
    return build_dataset(candles, order_books=books)


class TrainerTests(unittest.TestCase):
    """Use a small reproducible fixture; production CLI uses 3000 candles."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = fixture_dataset()
        cls.result = train_model(cls.dataset, max_iter=5)

    def test_seventy_thirty_cycle_split_and_no_label_overlap(self) -> None:
        train, test = self.result.train, self.result.test
        cycles = self.dataset.samples.cycle_start_ts.nunique()
        self.assertEqual(train.cycle_start_ts.nunique(), int(cycles * 0.7))
        self.assertFalse(set(train.cycle_start_ts) & set(test.cycle_start_ts))
        self.assertLess(train.index[-1], test.index[0])
        self.assertTrue((train.label_available_at < test.index[0]).all())

    def test_late_train_labels_are_purged(self) -> None:
        samples = self.dataset.samples.copy()
        original_train, test = chronological_split(samples)
        late_cycle = original_train.cycle_start_ts.iloc[-1]
        samples.loc[samples.cycle_start_ts == late_cycle, "label_available_at"] = test.index[0]
        train, _ = chronological_split(samples)
        self.assertNotIn(late_cycle, set(train.cycle_start_ts))

    def test_changing_test_targets_cannot_change_fitted_model(self) -> None:
        samples = self.dataset.samples.copy()
        samples.loc[self.result.test.index, "target"] = 1 - samples.loc[self.result.test.index, "target"]
        changed = TrainingDataset(samples, self.dataset.candles, self.dataset.cycles, self.dataset.price_mode)
        fitted = train_model(changed, max_iter=5)
        np.testing.assert_array_equal(self.result.predictions.p_yes, fitted.predictions.p_yes)

    def test_artifact_roundtrip_and_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "latest.joblib"
            save_artifact(self.result, path)
            loaded = load_artifact(path)
            self.assertIsNone(loaded["scaler"])
            self.assertEqual(loaded["metadata"]["feature_names"], list(FEATURE_NAMES))
            self.assertFalse(loaded["model"].early_stopping)
            self.assertTrue(path.with_suffix(".metadata.json").exists())
            self.assertTrue(path.with_suffix(".oos.csv").exists())
            np.testing.assert_array_equal(
                loaded["model"].predict_proba(self.result.test[list(FEATURE_NAMES)]),
                self.result.artifact["model"].predict_proba(self.result.test[list(FEATURE_NAMES)]),
            )

    def test_one_class_metrics_have_explicit_undefined_auc(self) -> None:
        metrics = evaluate(np.array([1, 1]), np.array([0.8, 0.9]))
        self.assertIsNone(metrics["auc_roc"])
        self.assertAlmostEqual(metrics["brier_score"], 0.025)
        self.assertGreater(metrics["log_loss"], 0)

    def test_synthetic_quotes_are_oos_and_two_cents_wide(self) -> None:
        snapshots, _ = synthetic_snapshots(self.dataset, self.result.artifact, path="OLHC")
        self.assertTrue(snapshots)
        cutoff = pd.Timestamp(self.result.artifact["metadata"]["inference_not_before"])
        self.assertTrue(all(snapshot.timestamp >= cutoff for snapshot in snapshots))
        self.assertTrue(all(snapshot.yes_ask_cents - snapshot.yes_bid_cents == 2 for snapshot in snapshots))
        self.assertTrue(all(snapshot.no_ask_cents - snapshot.no_bid_cents == 2 for snapshot in snapshots))
        reports = replay_synthetic(self.dataset, self.result.artifact)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all("SYNTHETIC" in report.scenario for report in reports))

    def test_dataset_marks_proxy_and_excludes_warmup(self) -> None:
        self.assertEqual(self.dataset.price_mode, "SPOT_PROXY")
        self.assertGreaterEqual(self.dataset.samples.index[0], self.dataset.candles.index[0] + 61 * MINUTE)
        self.assertTrue(all(record.source.startswith("SPOT_PROXY:") for record in self.dataset.cycles.values()))

    def test_official_csv_publication_times_and_price_mode(self) -> None:
        """External official records survive CSV loading without proxy substitution."""
        with TemporaryDirectory() as directory:
            candle_path = Path(directory) / "candles.csv"
            cycle_path = Path(directory) / "cycles.csv"
            self.dataset.candles.to_csv(candle_path, index_label="timestamp")
            pd.DataFrame([
                {
                    "cycle_start_ts": record.cycle_start_ts,
                    "strike_open": record.strike_open,
                    "strike_available_at": record.strike_available_at,
                    "cycle_settle": record.cycle_settle,
                    "settle_available_at": record.settle_available_at,
                    "source": "kalshi:official-test-fixture",
                }
                for record in self.dataset.cycles.values()
            ]).to_csv(cycle_path, index=False)
            candles = load_candles(candle_path)
            cycles = load_cycle_prices(cycle_path)
            official = build_dataset(candles, cycles, order_books=self.dataset.order_books)
            self.assertEqual(official.price_mode, "OFFICIAL_PRICES")
            self.assertEqual(len(official.samples), len(self.dataset.samples))
            self.assertEqual(official.samples.target.tolist(), self.dataset.samples.target.tolist())


if __name__ == "__main__":
    unittest.main()
