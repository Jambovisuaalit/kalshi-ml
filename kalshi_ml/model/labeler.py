"""Four-class, side-conditioned triple-barrier RESEARCH labels from minute bars.

OHLCV cannot recover Kalshi bid/ask paths or the ordering/time of intrabar
extrema. These labels use a declared Gaussian spot/quote proxy, frozen entry
volatility, a two-cent spread and SL-first ambiguity resolution. They are not
observed execution outcomes. Supply CyclePrices for actual strike/settlement;
otherwise the first open/final close are explicitly marked SPOT_PROXY.

The binary P(YES) trainer and daemon must not consume these four targets.
Each sample is a hypothetical entry, not a portfolio backtest or a fill.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Literal

import numpy as np
import pandas as pd

from kalshi_ml.data.collector import utc_timestamp
from kalshi_ml.data.features import CYCLE, MINUTE, CyclePrices

LABEL_HIT_SL = 0
LABEL_HIT_TP = 1
LABEL_EXP_0 = 2
LABEL_EXP_100 = 3
LABEL_NAMES = {0: "HIT_SL", 1: "HIT_TP", 2: "EXP_0", 3: "EXP_100"}
Side = Literal["YES", "NO"]
VolatilityUnit = Literal["fractional", "usd"]
TiePolicy = Literal["exclude", "yes", "no"]

OUTPUT_DTYPES: dict[str, str] = {
    "target": "int8", "entry_price_proxy": "float64", "cycle_strike": "float64",
    "cycle_id": "int64", "cycle_start_ts": "datetime64[ns, UTC]",
    "decision_ts": "datetime64[ns, UTC]", "remaining_sec": "float64",
    "volatility_at_entry": "float64", "exit_price_proxy": "float64",
    "exit_time_lower": "datetime64[ns, UTC]", "exit_time_upper": "datetime64[ns, UTC]",
    "label_available_at": "datetime64[ns, UTC]", "ambiguous_barrier": "bool",
    "side": "object", "exit_reason": "object", "label_mode": "object",
    "target_schema": "object",
}


def compute_binary_option_proxy(
    spot: float, strike: float, vol_15m: float, rem_fraction: float, *,
    volatility_unit: VolatilityUnit = "fractional", tie_result: bool | None = None,
) -> float:
    """Return a Gaussian arithmetic-price YES midpoint proxy, in cents.

    Fractional volatility matches compute_features: std(log returns)*sqrt(15).
    Convert to USD with the fixed strike reference; USD input is also supported
    explicitly. Remaining time is a fraction of fifteen minutes. The one-cent
    USD denominator floor is numerical regularization, not a volatility fit.
    No artificial early expiry cutoff or future-cycle volatility is used.
    At actual expiry ties require an explicit contract result.
    """
    if not all(math.isfinite(value) for value in (spot, strike, vol_15m, rem_fraction)):
        raise ValueError("proxy inputs must be finite")
    if spot <= 0 or strike <= 0 or vol_15m < 0 or not 0 <= rem_fraction <= 1:
        raise ValueError("invalid spot/strike/volatility or remaining fraction")
    if volatility_unit not in {"fractional", "usd"}:
        raise ValueError("volatility_unit must be fractional or usd")
    if tie_result is not None and type(tie_result) is not bool:
        raise ValueError("tie_result must be an explicit boolean")
    if rem_fraction == 0:
        if spot == strike:
            if tie_result is None:
                raise ValueError("expiry tie requires the contract's explicit result")
            return 100.0 if tie_result else 0.0
        return 100.0 if spot > strike else 0.0
    usd_vol = strike * vol_15m if volatility_unit == "fractional" else vol_15m
    if not math.isfinite(usd_vol):
        raise ValueError("USD volatility overflow")
    sigma = max(usd_vol * math.sqrt(rem_fraction), 0.01)
    return max(0.0, min(100.0, 50.0 * (1 + math.erf((spot - strike) / sigma / math.sqrt(2)))))


@dataclass(frozen=True, slots=True)
class _Exit:
    """Proxy outcome and the interval/publication time supporting its label."""

    target: int
    price: float
    lower: pd.Timestamp
    upper: pd.Timestamp
    available_at: pd.Timestamp
    reason: str
    ambiguous: bool = False


def _timestamps(values: pd.Series) -> pd.DatetimeIndex:
    """Numeric timestamps mean Unix SECONDS; text/datetime must be aware."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("nonfinite candle timestamps")
        index = pd.DatetimeIndex(pd.to_datetime(values, unit="s", utc=True))
    else:
        index = pd.DatetimeIndex([utc_timestamp(pd.Timestamp(value)) for value in values])
    if index.hasnans or not (index == index.floor("min")).all():
        raise ValueError("candles require whole-minute UTC open timestamps")
    return index


def _normalize(df: pd.DataFrame, as_of: datetime | pd.Timestamp | None) -> pd.DataFrame:
    """Keep source row positions, validate bars and derive causal rolling vol."""
    if not df.index.is_unique:
        raise ValueError("source row index must be unique")
    if not df.columns.is_unique:
        raise ValueError("source column names must be unique")
    required = ["open", "high", "low", "close"]
    if not set(required) <= set(df.columns):
        raise ValueError("candles require open/high/low/close")
    if "timestamp" in df:
        index = _timestamps(df.timestamp)
    elif isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        index = _timestamps(pd.Series(df.index))
    else:
        raise ValueError("provide Unix-second timestamp column or aware DatetimeIndex")
    frame = pd.DataFrame(df[required].to_numpy(), index=index, columns=required)
    frame["_position"] = np.arange(len(df))
    closed = index + MINUTE
    for column in ("available_at", "volatility_available_at"):
        values = [utc_timestamp(pd.Timestamp(value)) for value in df[column]] if column in df else closed
        frame[column] = pd.DatetimeIndex(values)
        if (frame[column] < closed).any() and column == "available_at":
            raise ValueError("a completed candle cannot arrive before its close")
    if "volatility_available_at" not in df:
        frame["volatility_available_at"] = frame.available_at
    if "vol_15m" in df:
        frame["vol_15m"] = df.vol_15m.to_numpy(dtype=float)
    if frame.empty:
        return frame
    horizon = utc_timestamp(as_of) if as_of is not None else max(closed.max(), frame.available_at.max())
    frame = frame.loc[(frame.index + MINUTE <= horizon) & (frame.available_at <= horizon)].sort_index()
    if frame.index.has_duplicates:
        raise ValueError("duplicate candle timestamps")
    prices = frame[required].astype(float)
    if (not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
            or (prices.low > prices.min(axis=1)).any() or (prices.high < prices.max(axis=1)).any()):
        raise ValueError("nonfinite or inconsistent candle OHLC")
    frame[required] = prices
    if "vol_15m" not in frame:
        returns = np.log(frame.close).diff()
        # A rolling window straddling a missing minute is not trustworthy.
        consecutive = frame.index.to_series().diff().eq(MINUTE).rolling(15).sum().eq(15)
        frame["vol_15m"] = returns.rolling(15, min_periods=15).std(ddof=1).mul(math.sqrt(15)).where(consecutive)
        # Derived volatility is observable only once ALL its source bars arrive.
        # Preserve nanosecond precision: float rolling maxima can round a
        # receipt just AFTER a decision back onto the decision timestamp.
        receipt_ns = frame.available_at.array.asi8
        latest = receipt_ns.copy()
        for lag in range(1, 16):
            latest[lag:] = np.maximum(latest[lag:], receipt_ns[:-lag])
        latest[:15] = np.iinfo(np.int64).min
        frame["volatility_available_at"] = pd.to_datetime(latest, unit="ns", utc=True)
    nonmissing = frame.vol_15m.dropna().to_numpy(dtype=float)
    if not np.isfinite(nonmissing).all() or (nonmissing < 0).any():
        raise ValueError("vol_15m must be nonnegative and finite (warmup NaN is allowed)")
    frame.attrs["as_of"] = horizon
    return frame


def _outcome(
    future: pd.DataFrame, *, start: pd.Timestamp, strike: float, volatility: float,
    entry_ask: float, entry_spot: float, decision: pd.Timestamp, side: Side,
    tp: int, hard_sl: int, delta_sl: bool, spread: float, unit: VolatilityUnit,
    settle: float | None, settled_at: pd.Timestamp | None, tie: TiePolicy,
) -> _Exit | None:
    """Evaluate known opening gaps first, then SL-first intraminute envelopes."""
    end = start + CYCLE

    def bid(spot: float, at: pd.Timestamp) -> float:
        yes = compute_binary_option_proxy(spot, strike, volatility, (end - at).total_seconds() / 900,
                                          volatility_unit=unit)
        return max(0.0, (yes if side == "YES" else 100 - yes) - spread / 2)

    def reason(spot: float, price: float) -> str | None:
        if delta_sl and (spot <= strike if side == "YES" else spot >= strike):
            return "Delta-SL"
        if price <= entry_ask - hard_sl:
            return "Hard-SL"
        return None

    immediate_bid = bid(entry_spot, decision)
    stop = reason(entry_spot, immediate_bid)
    if stop or immediate_bid >= tp:
        return _Exit(LABEL_HIT_SL if stop else LABEL_HIT_TP, immediate_bid, decision, decision, decision, stop or "TP")
    known_at = decision
    for opened, bar in future.iterrows():
        known_at = max(known_at, bar.available_at)
        open_bid = bid(float(bar.open), opened)
        stop = reason(float(bar.open), open_bid)
        if stop or open_bid >= tp:
            return _Exit(LABEL_HIT_SL if stop else LABEL_HIT_TP, open_bid, opened, opened, known_at, stop or "TP")
        # Freeze clock at bar OPEN; high/low occurrence times are unobservable.
        # This is an explicit quote-proxy convention, not an exact price path.
        adverse = float(bar.low if side == "YES" else bar.high)
        favorable = float(bar.high if side == "YES" else bar.low)
        low_bid, high_bid = bid(adverse, opened), bid(favorable, opened)
        stop = reason(adverse, low_bid)
        tp_hit = high_bid >= tp
        if stop or tp_hit:
            return _Exit(LABEL_HIT_SL if stop else LABEL_HIT_TP, low_bid if stop else float(tp),
                         opened, opened + MINUTE, known_at, stop or "TP", bool(stop and tp_hit))
    if settle is None or settled_at is None:
        return None
    if settle == strike:
        if tie == "exclude":
            return None
        yes_wins = tie == "yes"
    else:
        yes_wins = settle > strike
    wins = yes_wins if side == "YES" else not yes_wins
    return _Exit(LABEL_EXP_100 if wins else LABEL_EXP_0, 100.0 if wins else 0.0,
                 end, end, max(known_at, settled_at), "Settlement")


def generate_multiclass_labels(
    df: pd.DataFrame, tp_cents: int = 88, hard_sl_distance_cents: int = 14,
    use_delta_sl: bool = True, *, side: Side = "YES",
    cycle_prices: Mapping[pd.Timestamp, CyclePrices] | None = None,
    volatility_unit: VolatilityUnit = "fractional", spread_cents: float = 2.0,
    entry_min_remaining_sec: int = 180, entry_max_remaining_sec: int = 800,
    settlement_tie: TiePolicy = "exclude", as_of: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Label hypothetical YES or NO entries at each observable minute CLOSE.

    A timestamp denotes candle OPEN: row m is usable at start+(m+1)*60s.
    Thus default 180..800s remaining permits rows 1..11, not rows 2..11.
    Require exactly fifteen consecutive closed bars per cycle. Late entry
    candles/volatility are skipped: full next-bar OHLC cannot describe the
    post-entry path if entry occurs inside that next bar. Delayed FUTURE bars
    instead delay label_available_at, for downstream purged time-series splits.

    vol_15m is frozen from the entry row; if absent it is derived causally
    from fifteen past minute log returns (sixteen consecutive closes). Supplied
    volatility must also be computed causally by the caller. No whole-cycle
    median or future volatility is consulted. Derived volatility is fractional.

    In official mode a missing strike/record is never replaced by a candle
    open. Missing settlement still permits an earlier observable barrier label.
    Equality at expiry is excluded unless the caller supplies the applicable
    YES tie result. Intrabar SL wins when both barriers are possible; known
    opening gaps take priority. Exit price on an SL range hit is an adverse
    proxy bound, not a historical fill. Fees do not change these gross barriers.

    Returned rows retain their source index and all input columns. New audit
    columns, including cycle_strike and label_available_at, are added explicitly.
    Labels/exit/settlement audit columns MUST NOT be used as predictor features.
    Empty or incomplete input produces an empty, consistently typed result.
    """
    if (side not in {"YES", "NO"} or volatility_unit not in {"fractional", "usd"}
            or settlement_tie not in {"exclude", "yes", "no"}):
        raise ValueError("invalid side, volatility unit or settlement tie rule")
    if type(use_delta_sl) is not bool:
        raise ValueError("use_delta_sl must be boolean")
    for name, value in (("TP", tp_cents), ("hard SL", hard_sl_distance_cents)):
        if type(value) is not int or not 1 <= value <= 99:
            raise ValueError(f"{name} must be an integer in 1..99 cents")
    if (not math.isfinite(spread_cents) or not 0 <= spread_cents < 100
            or type(entry_min_remaining_sec) is not int or type(entry_max_remaining_sec) is not int
            or not 1 <= entry_min_remaining_sec <= entry_max_remaining_sec < 900):
        raise ValueError("invalid spread or entry time bounds")
    if "vol_15m" not in df and volatility_unit != "fractional":
        raise ValueError("automatically derived vol_15m is fractional, not USD")
    frame = _normalize(df, as_of)
    rows: list[dict[str, object]] = []
    positions: list[int] = []
    if not frame.empty:
        horizon = frame.attrs["as_of"]
        for start, group in frame.groupby(frame.index.floor("15min")):
            if not group.index.equals(pd.date_range(start, periods=15, freq="min")):
                continue
            record = None if cycle_prices is None else cycle_prices.get(start)
            if cycle_prices is not None and (record is None or record.cycle_start_ts != start):
                continue
            strike = float(group.open.iloc[0]) if record is None else record.strike_open
            strike_at = group.available_at.iloc[0] if record is None else record.strike_available_at
            settle = float(group.close.iloc[-1]) if record is None else record.cycle_settle
            settled_at = group.available_at.iloc[-1] if record is None else record.settle_available_at
            mode = "SPOT_AND_QUOTE_PROXY" if record is None else "SUPPLIED_CYCLE_PRICES_QUOTE_PROXY"
            for offset, (opened, bar) in enumerate(group.iterrows()):
                decision = opened + MINUTE
                remaining = (start + CYCLE - decision).total_seconds()
                if not entry_min_remaining_sec <= remaining <= entry_max_remaining_sec:
                    continue
                if (bar.available_at > decision or pd.isna(bar.volatility_available_at)
                        or bar.volatility_available_at > decision or strike_at > decision or pd.isna(bar.vol_15m)):
                    continue
                volatility = float(bar.vol_15m)
                yes = compute_binary_option_proxy(float(bar.close), strike, volatility, remaining / 900,
                                                  volatility_unit=volatility_unit)
                ask = min(100.0, (yes if side == "YES" else 100 - yes) + spread_cents / 2)
                outcome = _outcome(
                    group.iloc[offset + 1:], start=start, strike=strike, volatility=volatility,
                    entry_ask=ask, entry_spot=float(bar.close), decision=decision, side=side,
                    tp=tp_cents, hard_sl=hard_sl_distance_cents, delta_sl=use_delta_sl,
                    spread=spread_cents, unit=volatility_unit, settle=settle,
                    settled_at=settled_at, tie=settlement_tie,
                )
                if outcome is None or outcome.available_at > horizon:
                    continue
                positions.append(int(bar["_position"]))
                rows.append({
                    "target": outcome.target, "entry_price_proxy": ask, "cycle_strike": strike,
                    "cycle_id": int(start.timestamp()), "cycle_start_ts": start,
                    "decision_ts": decision, "remaining_sec": remaining, "volatility_at_entry": volatility,
                    "exit_price_proxy": outcome.price, "exit_time_lower": outcome.lower,
                    "exit_time_upper": outcome.upper, "label_available_at": outcome.available_at,
                    "ambiguous_barrier": outcome.ambiguous, "side": side,
                    "exit_reason": outcome.reason, "label_mode": mode,
                    "target_schema": "triple-barrier-four-class-v1",
                })
    result = df.iloc[positions].copy()
    audit = pd.DataFrame(rows, index=result.index)
    for column, dtype in OUTPUT_DTYPES.items():
        result[column] = audit[column].astype(dtype) if rows else pd.Series(index=result.index, dtype=dtype)
    result.attrs["target_schema"] = "triple-barrier-four-class-v1"
    return result


def main() -> int:
    """Write an atomic research-label CSV; never train a model or send orders."""
    import argparse
    import os
    from pathlib import Path
    import tempfile

    from kalshi_ml.model.dataset import load_cycle_prices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles-csv", type=Path, required=True)
    parser.add_argument("--cycle-prices", type=Path, help="supplied official strike/settlement CSV")
    parser.add_argument("--output", type=Path, default=Path("data/processed/multiclass_labels.csv"))
    parser.add_argument("--side", choices=("YES", "NO"), default="YES")
    parser.add_argument("--volatility-unit", choices=("fractional", "usd"), default="fractional")
    parser.add_argument("--as-of", help="UTC-aware observation horizon")
    args = parser.parse_args()
    try:
        result = generate_multiclass_labels(
            pd.read_csv(args.candles_csv), side=args.side, volatility_unit=args.volatility_unit,
            cycle_prices=load_cycle_prices(args.cycle_prices) if args.cycle_prices else None,
            as_of=utc_timestamp(pd.Timestamp(args.as_of)) if args.as_of else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".labels-", dir=args.output.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            result.to_csv(temporary, index_label="source_index")
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Research QUOTE_PROXY labels: {len(result)}; side={args.side}; not observed Kalshi exits")
        print({name: int((result.target == code).sum()) for code, name in LABEL_NAMES.items()})
        print(args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Label generation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
