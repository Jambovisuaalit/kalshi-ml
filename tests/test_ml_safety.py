"""Adversarial OBI causality, calibration isolation and entry-risk gates."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from kalshi_ml.config import FEATURE_NAMES, StrategyConfig
from kalshi_ml.data.collector import MarketDataCollector, MarketDataError
from kalshi_ml.data.features import intracycle_volatility
from kalshi_ml.data.orderbook import OrderBookSnapshot, imbalance_at, snapshot_records, validate_snapshots
from kalshi_ml.model.dataset import TrainingDataset, build_dataset
from kalshi_ml.model.inference import RealtimePredictor
from kalshi_ml.model.safety import VolatilityReference, net_edge_cents
from kalshi_ml.model.trainer import load_artifact, train_model
from kalshi_ml.backtest.engine import synthetic_snapshots
from test_collector import response
from test_trainer import fixture_dataset


class SafetyTests(unittest.TestCase):
    """All prices, snapshots and model targets here are synthetic test fixtures."""

    def test_two_sigma_boundaries_and_degenerate_reference(self) -> None:
        reference = VolatilityReference(0.01, 0.001, 100)
        self.assertTrue(reference.allows(0.012))
        self.assertTrue(reference.allows(0.008))
        self.assertFalse(reference.allows(0.012001))
        self.assertFalse(reference.allows(0.007999))
        self.assertFalse(reference.allows(float("nan")))
        self.assertFalse(reference.allows(0.01, 3))
        self.assertTrue(VolatilityReference(0, 0, 2).allows(0))
        self.assertFalse(VolatilityReference(0, 0, 2).allows(0.00001))

    def test_edge_is_owned_probability_less_price_and_fees(self) -> None:
        strategy = StrategyConfig()
        self.assertEqual(net_edge_cents(0.55, "YES", Decimal("0.66"), strategy), -15)
        self.assertEqual(net_edge_cents(0.55, "YES", Decimal("0.50"), strategy), 1)
        self.assertEqual(net_edge_cents(0.45, "NO", Decimal("0.50"), strategy), 1)
        self.assertEqual(net_edge_cents(0.71, "YES", Decimal("0.66"), strategy), 1)
        for value in (Decimal(0), Decimal("NaN"), Decimal(1)):
            with self.assertRaises(ValueError):
                net_edge_cents(0.55, "YES", value, strategy)
        with self.assertRaises(ValueError):
            replace(strategy, volatility_sigma_limit=Decimal(3))

    def test_obi_asof_never_reads_future_or_delayed_snapshots(self) -> None:
        now = pd.Timestamp("2026-11-01T12:00:00Z")
        records = (
            OrderBookSnapshot(now - pd.Timedelta(seconds=4), now - pd.Timedelta(seconds=3), 3.0, 1.0, 1),
            OrderBookSnapshot(now - pd.Timedelta(seconds=1), now + pd.Timedelta(seconds=1), 1.0, 9.0, 2),
        )
        self.assertEqual(imbalance_at(records, now), 0.5)
        self.assertEqual(imbalance_at(records, now + pd.Timedelta(seconds=1)), -0.8)
        with self.assertRaises(ValueError):
            imbalance_at(records, now + pd.Timedelta(seconds=12))
        with self.assertRaises(ValueError):
            imbalance_at(records, now - pd.Timedelta(seconds=5))
        delayed = (OrderBookSnapshot(now - pd.Timedelta(seconds=20), now, 1.0, 1.0, 1),)
        with self.assertRaises(ValueError):
            imbalance_at(delayed, now)

    def test_obi_validation_rejects_invalid_depth_size_sequence(self) -> None:
        now = pd.Timestamp("2026-11-01T12:00:00Z")
        base = OrderBookSnapshot(now, now, 1.0, 1.0, 1)
        for kwargs in ({"depth": 1}, {"bid_volume": 0}, {"sequence": -1}, {"ask_volume": float("inf")},
                       {"timestamp": now + pd.Timedelta(seconds=1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(base, **kwargs)
        with self.assertRaises(ValueError):
            validate_snapshots(pd.DataFrame([{"sequence": 1.5, "depth": 10}]))

    def test_collector_uses_l2_top_ten_btc_sizes(self) -> None:
        now = pd.Timestamp("2026-11-01T12:00:00Z")
        payload = {"sequence": 1, "time": now.isoformat(),
                   "bids": [[str(100_000 - i), "3" if i < 10 else "1000", 1] for i in range(11)],
                   "asks": [[str(100_001 + i), "1" if i < 10 else "1000", 1] for i in range(11)]}
        session = Mock(spec=requests.Session)
        session.get.return_value = response(payload)
        with MarketDataCollector(session=session, clock=lambda: now.to_pydatetime(), min_request_interval_sec=0) as client:
            book = client.fetch_order_book()
            self.assertEqual(book.imbalance, 0.5)
            self.assertEqual(book.bid_volume, 30)
            self.assertEqual(session.get.call_args.kwargs["params"], {"level": 2})
            for invalid in ({**payload, "time": (now - pd.Timedelta(seconds=11)).isoformat()},
                            {**payload, "asks": []}, {**payload, "time": None}):
                session.get.return_value = response(invalid)
                with self.assertRaises(MarketDataError):
                    client.fetch_order_book()

    def test_intracycle_volatility_detects_jump_and_closed_bar_reversal(self) -> None:
        start = pd.Timestamp("2026-11-01T12:00:00Z")
        now = start + pd.Timedelta(minutes=2, seconds=30)
        frame = pd.DataFrame({"open": 100000., "close": 100000., "high": 100001., "low": 99999., "volume": 1.},
                             index=pd.date_range(start, periods=3, freq="min"))
        normal = intracycle_volatility(frame, 100000, now, start)
        jump = intracycle_volatility(frame, 90000, now, start)
        self.assertGreater(jump, normal * 100)
        frame.loc[start, "low"] = 90000  # recovery before bar close still registers
        self.assertGreater(intracycle_volatility(frame, 100000, now, start), normal * 100)
        before = intracycle_volatility(frame, 100000, now, start)
        frame.loc[start + pd.Timedelta(minutes=2), "high"] = 1e9  # forming bar is unavailable
        self.assertEqual(intracycle_volatility(frame, 100000, now, start), before)

    def test_missing_historical_obi_cannot_be_synthesized(self) -> None:
        dataset = fixture_dataset()
        with self.assertRaisesRegex(ValueError, "historical Coinbase"):
            build_dataset(dataset.candles)
        with self.assertRaises(ValueError):
            load_artifact(Path("models/hgb_btc15m_prod.joblib"))

    def test_inference_applies_calibrator_instead_of_raw_model(self) -> None:
        model, calibrator = Mock(), Mock()
        calibrator.classes_ = np.array([0, 1])
        calibrator.predict_proba.return_value = np.array([[0.4, 0.6]])
        predictor = RealtimePredictor(model, calibrator=calibrator)
        self.assertEqual(predictor.predict_yes(dict.fromkeys(FEATURE_NAMES, 0.0)), 0.6)
        model.predict_proba.assert_not_called()
        with self.assertRaises(ValueError):
            RealtimePredictor(model).predict_yes(dict.fromkeys(FEATURE_NAMES, 0.0))

    def test_calibration_and_volatility_reference_are_oos_isolated(self) -> None:
        dataset = fixture_dataset()
        result = train_model(dataset, max_iter=5)
        metadata = result.artifact["metadata"]
        split = metadata["calibration"]
        self.assertLess(pd.Timestamp(split["fit_label_latest"]), pd.Timestamp(split["start"]))
        self.assertLess(pd.Timestamp(split["label_latest"]), result.test.index[0])
        self.assertEqual(metadata["volatility_reference"]["count"], split["fit_samples"])
        changed = dataset.samples.copy()
        changed.loc[result.test.index, "cycle_volatility"] *= 1000
        changed.loc[result.test.index, "target"] = 1 - changed.loc[result.test.index, "target"]
        changed.loc[result.test.index, "obi"] = -changed.loc[result.test.index, "obi"]
        rerun = train_model(TrainingDataset(changed, dataset.candles, dataset.cycles, dataset.price_mode), max_iter=5)
        self.assertEqual(metadata["volatility_reference"], rerun.artifact["metadata"]["volatility_reference"])
        np.testing.assert_array_equal(result.artifact["calibrator"].predict_proba(result.test[list(FEATURE_NAMES)]),
                                      rerun.artifact["calibrator"].predict_proba(result.test[list(FEATURE_NAMES)]))

    def test_replay_retains_exit_quotes_when_obi_is_missing(self) -> None:
        dataset = fixture_dataset()
        result = train_model(dataset, max_iter=5)
        with patch.object(result.artifact["calibrator"], "predict_proba", side_effect=AssertionError("must not impute OBI")):
            snapshots, _ = synthetic_snapshots(replace(dataset, order_books=()), result.artifact, path="OLHC")
        self.assertTrue(snapshots)
        self.assertTrue(all(snapshot.cycle_volatility is None and snapshot.p_yes == 0.5 for snapshot in snapshots))


if __name__ == "__main__":
    unittest.main()
