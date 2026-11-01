"""Chronological HGB training: python -m kalshi_ml.model.trainer --candles 3000.

Default downloads use explicitly labelled spot-price proxy outcomes. Supply
--cycle-prices for official strike/settlement records. No random split,
automatic random early-stopping holdout, or refit on OOS samples is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from kalshi_ml.config import FEATURE_NAMES, load_settings
from kalshi_ml.data.collector import MarketDataCollector, RetryPolicy, utc_timestamp
from kalshi_ml.data.features import MINUTE
from kalshi_ml.model.dataset import TrainingDataset, build_dataset, load_candles, load_cycle_prices

ARTIFACT_VERSION = 1
FEATURE_VERSION = "closed-logreturns-v1"
DEFAULT_MODEL_PATH = Path("models/hgb_btc15m_latest.joblib")


@dataclass(frozen=True)
class TrainingResult:
    """Holdout metrics and split indices for reproducible independent replay."""

    artifact: dict[str, Any]
    train: pd.DataFrame
    test: pd.DataFrame
    predictions: pd.DataFrame


def chronological_split(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the newest 30% of whole cycles and purge unavailable train labels.

    TimeSeriesSplit's last expanding split yields the 70/30 holdout. Earlier
    splits are not used for tuning. Ratios are approximate due to whole-cycle
    rounding and label purging; actual sample/cycle counts are recorded.
    """
    if not samples.index.is_monotonic_increasing or samples.index.has_duplicates:
        raise ValueError("samples must be strictly chronological")
    cycles = pd.Index(samples.cycle_start_ts.unique()).sort_values()
    if len(cycles) < 6:
        raise ValueError("at least six usable cycles are required")
    test_size = len(cycles) - int(len(cycles) * 0.70)
    splits = list(TimeSeriesSplit(n_splits=2, test_size=test_size).split(cycles))
    train_cycles, test_cycles = splits[-1]
    train = samples.loc[samples.cycle_start_ts.isin(cycles[train_cycles])].copy()
    test = samples.loc[samples.cycle_start_ts.isin(cycles[test_cycles])].copy()
    # Fit at the first OOS decision; labels published at that exact instant are
    # conservatively excluded as order-of-events is not independently known.
    train = train.loc[train.label_available_at < test.index[0]]
    if train.empty or test.empty or train.target.nunique() != 2:
        raise ValueError("training split must contain samples from both outcome classes")
    return train, test


def evaluate(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    """Report binary probability quality; AUC is undefined for one-class OOS."""
    return {
        "auc_roc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def train_model(dataset: TrainingDataset, *, max_iter: int = 150, random_state: int = 42) -> TrainingResult:
    """Fit only pre-holdout samples with a fixed, untuned hyperparameter set.

    HGB uses tree thresholds, so no scaler is fitted or needed. An explicit
    scaler=None and preprocessing specification are persisted with the model.
    """
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not set(dataset.samples.target.unique()) <= {0, 1}:
        raise ValueError("targets must be binary 0/1")
    train, test = chronological_split(dataset.samples)
    x_train = train.loc[:, list(FEATURE_NAMES)].astype(float)
    x_test = test.loc[:, list(FEATURE_NAMES)].astype(float)
    if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_test.to_numpy()).all():
        raise ValueError("training and test features must be finite")
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=random_state,
    )
    model.fit(x_train, train.target.to_numpy(dtype=int))
    probabilities = model.predict_proba(x_test)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    metrics = evaluate(test.target.to_numpy(dtype=int), probabilities)
    baseline = np.full(len(test), float(train.target.mean()))
    predictions = test[["cycle_start_ts", "target", "label_available_at"]].copy()
    predictions["p_yes"] = probabilities
    # Fingerprint both observed features and target publication times.
    digest = hashlib.sha256(dataset.samples.to_csv().encode("utf-8")).hexdigest()
    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION, "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES), "price_mode": dataset.price_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__, "random_state": random_state,
        "preprocessing": "identity; no scaler required for histogram trees",
        "split": "whole-cycle chronological 70/30, purged label availability",
        "train_samples": len(train), "test_samples": len(test),
        "train_cycles": int(train.cycle_start_ts.nunique()), "test_cycles": int(test.cycle_start_ts.nunique()),
        "train_start": train.index[0].isoformat(), "train_end": train.index[-1].isoformat(),
        "train_label_latest": train.label_available_at.max().isoformat(),
        "inference_not_before": test.index[0].isoformat(), "test_end": test.index[-1].isoformat(),
        "metrics": metrics, "constant_train_prior_metrics": evaluate(test.target.to_numpy(dtype=int), baseline),
        "model_parameters": model.get_params(), "dataset_sha256": digest,
        "outcome_rule": "settle > strike; exact ties excluded; verify contract comparator",
        "candles_source": dataset.candles.attrs.get("source", "unknown"),
    }
    return TrainingResult({"model": model, "scaler": None, "metadata": metadata}, train, test, predictions)


def save_artifact(result: TrainingResult, path: Path = DEFAULT_MODEL_PATH) -> None:
    """Atomically publish one compressed joblib bundle, plus readable audit files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(result.artifact, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(result.artifact["metadata"], indent=2, allow_nan=False), encoding="utf-8",
    )
    result.predictions.to_csv(path.with_suffix(".oos.csv"), index_label="timestamp")


def load_artifact(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load ONLY a trusted local artifact: joblib/pickle can execute arbitrary code."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError("invalid model artifact")
    metadata = artifact["metadata"]
    if (
        metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("feature_version") != FEATURE_VERSION
        or metadata.get("feature_names") != list(FEATURE_NAMES)
        or metadata.get("sklearn_version") != sklearn.__version__
        or not isinstance(artifact.get("model"), HistGradientBoostingClassifier)
        or artifact.get("scaler") is not None
    ):
        raise ValueError("incompatible model, feature schema, preprocessing or sklearn version")
    return artifact


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared CLI input schema for training and historical replay."""
    parser.add_argument("--candles", type=int, default=3000, help="number of closed 1m candles to download")
    parser.add_argument("--candles-csv", type=Path, help="use local candle CSV instead of downloading")
    parser.add_argument("--cycle-prices", type=Path, help="official cycle price/publication CSV; otherwise SPOT_PROXY")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--end", help="exclusive UTC-aware minute boundary for downloads, e.g. 2026-10-30T19:55:00Z")


def dataset_from_arguments(args: argparse.Namespace) -> TrainingDataset:
    """Fetch/read candles and persist raw input used by the default CLI run."""
    if args.candles < 150:
        raise ValueError("--candles must be at least 150")
    if args.candles_csv is not None:
        candles = load_candles(args.candles_csv)
    else:
        settings = load_settings(args.env_file)
        # Leave a minute for public candle publication; do not request a bar
        # that has just closed and may not yet exist on the REST endpoint.
        end = utc_timestamp(pd.Timestamp(args.end)) if args.end else pd.Timestamp.now(tz="UTC").floor("min") - MINUTE
        policy = RetryPolicy(max_retries=settings.http_max_retries, timeout_sec=float(settings.http_timeout_sec))
        with MarketDataCollector(retry=policy) as collector:
            candles = collector.fetch_candles(end - args.candles * MINUTE, end)
        destination = Path("data/raw/btc_usd_1m_latest.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        candles.to_csv(destination, index_label="timestamp")
    cycles = load_cycle_prices(args.cycle_prices) if args.cycle_prices else None
    return build_dataset(candles, cycles)


def main() -> int:
    """Train, print holdout metrics, save the bundle and replay the holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()
    try:
        dataset = dataset_from_arguments(args)
        result = train_model(dataset, max_iter=args.max_iter)
        save_artifact(result, args.output)
        metadata = result.artifact["metadata"]
        print(f"Data mode: {dataset.price_mode}; train={len(result.train)}, OOS={len(result.test)}")
        print(json.dumps(metadata["metrics"], indent=2, allow_nan=False))
        if metadata["metrics"]["auc_roc"] is None:
            print("AUC-ROC unavailable: holdout contains only one outcome class.")
        print(f"Saved: {args.output}")
        if not args.skip_backtest:
            from kalshi_ml.backtest.engine import ExecutionConfig, replay_synthetic

            strategy = load_settings(args.env_file).strategy
            reports = replay_synthetic(dataset, result.artifact, strategy=strategy)
            for report in reports:
                print(report.format_report())
            header = json.dumps({"strategy": asdict(strategy), "execution": asdict(ExecutionConfig())}, default=str, indent=2)
            args.output.with_suffix(".backtest.txt").write_text(
                header + "\n\n" + "\n\n".join(report.format_report() for report in reports) + "\n",
                encoding="utf-8",
            )
            trade_rows = [{"scenario": report.scenario, **asdict(trade)} for report in reports for trade in report.trades]
            pd.DataFrame(trade_rows, columns=[
                "scenario", "cycle_start_ts", "side", "entry_ts", "exit_ts", "entry_cents", "exit_cents",
                "contracts", "total_fee_cents", "net_pnl_cents", "reason",
            ]).to_csv(args.output.with_suffix(".trades.csv"), index=False)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
