"""Causal KXBTC15M features and separate, explicitly sourced cycle outcomes.

Candles use minute-open UTC timestamps. A candle's close is usable only at
open + 60 seconds. An optional ``available_at`` column can encode a later
actual receipt time for point-in-time replay. Without it the caller assumes
completed candles are available immediately at their theoretical close.

Official Kalshi strike/settlement values must be supplied by the caller via
``bind_cycle_prices``. Coinbase/Kraken candle opens and closes are only spot
proxies and are never silently substituted for official contract values.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
import pandas as pd

from kalshi_ml.config import FEATURE_NAMES
from kalshi_ml.data.collector import utc_timestamp

CYCLE = pd.Timedelta(minutes=15)
MINUTE = pd.Timedelta(minutes=1)
PRICE_FLOOR_USD = 0.01


class FeatureDataError(ValueError):
    """A feature cannot be calculated from trustworthy point-in-time inputs."""


def cycle_start(timestamp: datetime | pd.Timestamp) -> pd.Timestamp:
    """Return the UTC quarter-hour start (minutes 00, 15, 30, or 45)."""
    return utc_timestamp(timestamp).floor("15min")


@dataclass(frozen=True, slots=True)
class CyclePrices:
    """Official contract prices and the times they became observable.

    ``source`` identifies the official market metadata/result provenance.
    Supply a record for the exact contract; the API adapter that retrieves
    these values is intentionally outside the spot-data collector.
    Settlement is optional until the official result has been published.
    """

    cycle_start_ts: pd.Timestamp
    strike_open: float
    strike_available_at: pd.Timestamp
    source: str
    cycle_settle: float | None = None
    settle_available_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        """Normalize timestamps and reject inconsistent official price records."""
        start = utc_timestamp(self.cycle_start_ts)
        strike_at = utc_timestamp(self.strike_available_at)
        if cycle_start(start) != start:
            raise ValueError("cycle_start_ts must be a UTC quarter-hour boundary")
        if not self.source.strip() or not math.isfinite(self.strike_open) or self.strike_open <= 0:
            raise ValueError("official strike and provenance are required")
        if strike_at < start:
            raise ValueError("strike cannot be observed before cycle start")
        if (self.cycle_settle is None) != (self.settle_available_at is None):
            raise ValueError("settlement price and availability must be supplied together")
        object.__setattr__(self, "cycle_start_ts", start)
        object.__setattr__(self, "strike_available_at", strike_at)
        if self.cycle_settle is not None:
            if not math.isfinite(self.cycle_settle) or self.cycle_settle <= 0:
                raise ValueError("settlement must be finite and positive")
            settled_at = utc_timestamp(self.settle_available_at)
            if settled_at < start + CYCLE:
                raise ValueError("settlement cannot be observed before cycle end")
            object.__setattr__(self, "settle_available_at", settled_at)

    def settlement_delta(self, as_of: datetime | pd.Timestamp) -> float:
        """Return official settle minus strike only after result availability.

        This is an outcome for labels/evaluation, never an input feature.
        Contract tie and rounding rules must be handled by the label module.
        """
        if self.cycle_settle is None or self.settle_available_at is None:
            raise FeatureDataError("official settlement has not been supplied")
        if utc_timestamp(as_of) < self.settle_available_at:
            raise FeatureDataError("official settlement was not yet available")
        return self.cycle_settle - self.strike_open


def bind_cycle_prices(df: pd.DataFrame, prices: Iterable[CyclePrices]) -> pd.DataFrame:
    """Return a copy with official records in attrs, separate from OHLCV columns.

    DataFrame attrs are an in-memory contract, not a persistence format. Persist
    official records separately and rebind them after CSV/Parquet loading.
    """
    records: dict[pd.Timestamp, CyclePrices] = {}
    for record in prices:
        if record.cycle_start_ts in records:
            raise ValueError("duplicate official cycle record")
        records[record.cycle_start_ts] = record
    result = df.copy()
    result.attrs["cycle_prices"] = records
    return result


def _closed_history(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Select available completed bars BEFORE inspecting their price contents."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise FeatureDataError("candles require a timezone-aware DatetimeIndex")
    if df.index.hasnans:
        raise FeatureDataError("candle timestamps must not contain NaT")
    result = df.copy()
    result.index = result.index.tz_convert("UTC")
    result = result.loc[result.index + MINUTE <= now]
    if "available_at" in result:
        try:
            available = result["available_at"].map(utc_timestamp)
        except (TypeError, ValueError) as exc:
            raise FeatureDataError("available_at must contain aware timestamps") from exc
        if (available < result.index + MINUTE).any():
            raise FeatureDataError("a completed candle cannot be available before its close")
        result = result.loc[available <= now]
    if result.index.has_duplicates:
        raise FeatureDataError("duplicate completed candle timestamps")
    if not (result.index == result.index.floor("min")).all():
        raise FeatureDataError("candles must use whole-minute open timestamps")
    return result.sort_index()


def aggregate_15m(
    df: pd.DataFrame, as_of: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate complete 15-row cycles into spot OHLCV proxies.

    Incomplete cycles and cycles containing gaps are excluded. Columns are
    ``spot_open``, ``spot_high``, ``spot_low``, ``spot_close``, ``volume``.
    No official strike, settlement or contract outcome is inferred here.
    """
    now = utc_timestamp(as_of)
    history = _closed_history(df, now)
    columns = ["spot_open", "spot_high", "spot_low", "spot_close", "volume"]
    rows: list[dict[str, float]] = []
    starts: list[pd.Timestamp] = []
    for start, group in history.groupby(history.index.floor("15min")):
        expected = pd.date_range(start, periods=15, freq="min")
        if start + CYCLE > now or not group.index.equals(expected):
            continue
        try:
            values = group[["open", "high", "low", "close", "volume"]].astype(float)
        except (KeyError, ValueError, TypeError) as exc:
            raise FeatureDataError("invalid cycle OHLCV") from exc
        _validate_ohlcv(values)
        starts.append(start)
        rows.append({
            "spot_open": float(values.open.iloc[0]),
            "spot_high": float(values.high.max()),
            "spot_low": float(values.low.min()),
            "spot_close": float(values.close.iloc[-1]),
            "volume": float(values.volume.sum()),
        })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(starts, tz="UTC", name="cycle_start_ts"), columns=columns)


def _validate_ohlcv(values: pd.DataFrame) -> None:
    prices = values[["open", "high", "low", "close"]]
    if (
        not np.isfinite(values.to_numpy()).all()
        or (prices <= 0).any().any() or (values.volume < 0).any()
        or (values.low > prices.min(axis=1)).any()
        or (values.high < prices.max(axis=1)).any()
    ):
        raise FeatureDataError("nonfinite or inconsistent OHLCV")


def compute_features(
    df: pd.DataFrame,
    current_spot: float,
    current_ts: datetime | pd.Timestamp,
    cycle_start_ts: datetime | pd.Timestamp,
) -> dict[str, float]:
    """Calculate exactly eight causal features in the configured model order.

    Requires 61 consecutive completed minute bars ending at the last closed
    minute, and a CyclePrices record attached with bind_cycle_prices. A missing
    or not-yet-observable official strike raises FeatureDataError.

    Formulas (returns are fractional, not percentage points):
      delta_usd = current_spot - official strike_open
      remaining_sec = cycle_end - current_ts
      vol_15m = sample_std(last 15 one-minute log returns) * sqrt(15)
      vol_60m = sample_std(last 60 one-minute log returns) * sqrt(60)
      vol_ratio = vol_15m / vol_60m (1 for completely flat history)
      z_score = delta_usd / max(0.01 USD, spot * std_15 * sqrt(remaining/60))
      mom_Nm = spot / last closed price available at current_ts - N minutes - 1

    The z-score's one-cent denominator floor handles zero realized volatility.
    Momentum uses the minute-close step function at non-minute timestamps.
    The caller must supply a spot observation known at current_ts (for live
    data use SpotTicker.timestamp and enforce freshness). Never supply a
    cycle's eventual close as its earlier current_spot during training.
    Settlement fields are not read by this function.
    """
    now, start = utc_timestamp(current_ts), utc_timestamp(cycle_start_ts)
    if start != cycle_start(now):
        raise FeatureDataError("cycle_start_ts must match the active UTC quarter-hour")
    if not math.isfinite(current_spot) or current_spot <= 0:
        raise FeatureDataError("current_spot must be finite and positive")
    records = df.attrs.get("cycle_prices", {})
    if not isinstance(records, dict):
        raise FeatureDataError("invalid official cycle records")
    record = records.get(start)
    if not isinstance(record, CyclePrices) or record.strike_available_at > now:
        raise FeatureDataError("official cycle strike is missing or not yet observable")
    history = _closed_history(df, now)
    expected = pd.date_range(now.floor("min") - 61 * MINUTE, periods=61, freq="min")
    window = history.loc[(history.index >= expected[0]) & (history.index <= expected[-1])]
    if not window.index.equals(expected):
        raise FeatureDataError("require 61 consecutive available completed minute candles")
    try:
        values = window[["open", "high", "low", "close", "volume"]].astype(float)
    except (KeyError, ValueError, TypeError) as exc:
        raise FeatureDataError("invalid feature-window OHLCV") from exc
    _validate_ohlcv(values)
    close = values.close.to_numpy()
    returns = np.diff(np.log(close))
    std15 = float(np.std(returns[-15:], ddof=1))
    std60 = float(np.std(returns, ddof=1))
    vol15, vol60 = std15 * math.sqrt(15), std60 * math.sqrt(60)
    delta = float(current_spot) - record.strike_open
    remaining = (start + CYCLE - now).total_seconds()

    def momentum(minutes: int) -> float:
        # Bar at t-MINUTE has its close at t; use only closes <= cutoff.
        cutoff = (now - minutes * MINUTE).floor("min") - MINUTE
        return float(current_spot) / float(values.loc[cutoff, "close"]) - 1.0

    result = {
        "delta_usd": delta,
        "remaining_sec": remaining,
        "z_score": delta / max(PRICE_FLOOR_USD, float(current_spot) * std15 * math.sqrt(remaining / 60)),
        "vol_15m": vol15,
        "vol_60m": vol60,
        "vol_ratio": vol15 / vol60 if vol60 > 0 else 1.0,
        "mom_1m": momentum(1),
        "mom_3m": momentum(3),
    }
    if tuple(result) != FEATURE_NAMES or not all(math.isfinite(value) for value in result.values()):
        raise FeatureDataError("feature schema mismatch or nonfinite output")
    return result
