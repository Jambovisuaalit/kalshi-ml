"""Validated, reusable real-time inference with the training feature schema."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from kalshi_ml.config import FEATURE_NAMES

PACKAGED_MODEL_PATH = Path("models/hgb_btc15m_prod.joblib")


def default_model_path() -> Path:
    """Prefer the working copy, then the model installed with a wheel."""
    if PACKAGED_MODEL_PATH.is_file():
        return PACKAGED_MODEL_PATH
    installed = Path(sys.prefix) / "share/kalshi-ml" / PACKAGED_MODEL_PATH
    return installed if installed.is_file() else PACKAGED_MODEL_PATH


class RealtimePredictor:
    """Load once; validate each feature vector and probability distribution.

    No calibration, scaler fit or model mutation occurs during inference.
    Only load trusted joblib files, which can execute Python during loading.
    """

    def __init__(self, model: Any, metadata: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.metadata = dict(metadata or {})
        if self.metadata.get("feature_names", list(FEATURE_NAMES)) != list(FEATURE_NAMES):
            raise ValueError("incompatible feature order")

    @classmethod
    def from_file(cls, path: Path | None = None) -> RealtimePredictor:
        """Read and version-check a trusted artifact using the training loader."""
        from kalshi_ml.model.trainer import load_artifact

        artifact = load_artifact(default_model_path() if path is None else path)
        return cls(artifact["model"], artifact["metadata"])

    def predict_batch(self, frame: pd.DataFrame, *, as_of: datetime | None = None) -> np.ndarray:
        """Return P(YES) for finite rows in exactly the declared column order."""
        if list(frame.columns) != list(FEATURE_NAMES) or frame.empty:
            raise ValueError("expected a nonempty frame with exactly eight ordered features")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("features must be finite")
        if as_of is not None:
            timestamp = pd.Timestamp(as_of)
            if timestamp.tzinfo is None:
                raise ValueError("as_of must be timezone-aware")
            cutoff = self.metadata.get("inference_not_before")
            if cutoff is not None and timestamp < pd.Timestamp(cutoff):
                raise ValueError("inference precedes the model's OOS cutoff")
        classes = np.asarray(self.model.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("model must expose exactly the binary classes 0 and 1")
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)
        if (probabilities.shape != (len(frame), 2) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)):
            raise ValueError("model returned an invalid probability distribution")
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]

    def predict_yes(self, features: Mapping[str, float], *, as_of: datetime | None = None) -> float:
        """Evaluate one current decision vector in canonical training order."""
        if set(features) != set(FEATURE_NAMES):
            raise ValueError("feature names must match the eight training features")
        frame = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=list(FEATURE_NAMES))
        return float(self.predict_batch(frame, as_of=as_of)[0])
