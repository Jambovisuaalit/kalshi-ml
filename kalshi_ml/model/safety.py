"""Training-only volatility reference and conservative executable-price edge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import math
from typing import Any, Mapping

import numpy as np

from kalshi_ml.config import StrategyConfig


@dataclass(frozen=True, slots=True)
class VolatilityReference:
    """Mean/sample sigma of intracycle range/return RMS per square-root minute."""

    mean: float
    std: float
    count: int

    def __post_init__(self) -> None:
        if (not all(math.isfinite(x) and x >= 0 for x in (self.mean, self.std))
                or type(self.count) is not int or self.count < 2):
            raise ValueError("invalid training volatility reference")

    @classmethod
    def fit(cls, values: np.ndarray) -> VolatilityReference:
        """Called only on base-model fit data, never calibration or test data."""
        data = np.asarray(values, dtype=float)
        if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("invalid training volatility samples")
        return cls(float(data.mean()), float(data.std(ddof=1)), len(data))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> VolatilityReference:
        """Missing/incompatible risk metadata must fail closed."""
        record = metadata.get("volatility_reference")
        if not isinstance(record, dict) or record.get("statistic") != "intracycle-range-return-rms-per-sqrt-minute-v1":
            raise ValueError("model lacks a compatible volatility reference")
        return cls(record["mean"], record["std"], record["count"])

    def metadata(self) -> dict[str, Any]:
        """Serializable reference, including its exact statistical definition."""
        return {**asdict(self), "statistic": "intracycle-range-return-rms-per-sqrt-minute-v1"}

    def allows(self, volatility: float, sigma_limit: float = 2.0) -> bool:
        """Inclusive two-sided band; degenerate distributions allow equality only."""
        if not math.isfinite(volatility) or volatility < 0 or not 0 < sigma_limit <= 2:
            return False
        tolerance = max(1e-15, abs(self.mean) * 1e-12)
        return abs(volatility - self.mean) <= sigma_limit * self.std + tolerance


def net_edge_cents(probability: float, side: str, ask_dollars: Decimal, strategy: StrategyConfig) -> Decimal:
    """Estimated settlement EV after reserved entry/exit fee budgets.

    This is model-based expected value, not a guarantee of predictive accuracy
    or of the TP/SL policy's path-dependent return. The executable limit must
    include rounding/slippage; fee budgets must cover the actual account fees.
    """
    if side not in {"YES", "NO"} or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid binary probability/side")
    if not ask_dollars.is_finite() or not 0 < ask_dollars < 1:
        raise ValueError("invalid executable ask")
    yes = Decimal(str(probability))
    owned = yes if side == "YES" else 1 - yes
    return owned * 100 - ask_dollars * 100 - strategy.entry_fee_buffer_cents - strategy.exit_fee_buffer_cents
