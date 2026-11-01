"""Build point-in-time training samples without using outcomes as features."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from kalshi_ml.data.collector import MarketDataCollector, utc_timestamp
from kalshi_ml.data.features import (
    CYCLE, MINUTE, CyclePrices, FeatureDataError, aggregate_15m,
    bind_cycle_prices, compute_features, cycle_start, intracycle_volatility,
)
from kalshi_ml.data.orderbook import OrderBookSnapshot, imbalance_at, snapshot_records


@dataclass(frozen=True)
class TrainingDataset:
    """Samples plus the price lineage needed for independent OOS replay."""

    samples: pd.DataFrame
    candles: pd.DataFrame
    cycles: dict[pd.Timestamp, CyclePrices]
    price_mode: str
    order_books: tuple[OrderBookSnapshot, ...] = ()


def load_candles(path: Path) -> pd.DataFrame:
    """Read timestamp,open,high,low,close,volume CSV; timestamps must be aware."""
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError("candle CSV requires timestamp")
    index = pd.DatetimeIndex([utc_timestamp(pd.Timestamp(t)) for t in frame.pop("timestamp")], name="timestamp")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("candle CSV timestamps must be unique and increasing")
    frame.index = index
    if "available_at" in frame:
        frame["available_at"] = frame.available_at.map(lambda t: utc_timestamp(pd.Timestamp(t)))
    if not len(frame):
        raise ValueError("empty candle CSV")
    # Reuse the collector's structural/price checks without discarding gaps.
    rows = [[t.timestamp(), *row] for t, row in zip(index, frame[["open", "high", "low", "close", "volume"]].to_numpy())]
    checked = MarketDataCollector._frame(rows, index[0], index[-1] + MINUTE, "csv")
    if "available_at" in frame:
        checked["available_at"] = frame.available_at
    return checked


def load_cycle_prices(path: Path) -> dict[pd.Timestamp, CyclePrices]:
    """Read official cycle records with explicit publication times and source.

    Columns: cycle_start_ts,strike_open,strike_available_at,cycle_settle,
    settle_available_at,source. A settlement equal to strike is excluded from
    training/replay: its contract-specific tie result is not inferred.
    """
    frame = pd.read_csv(path)
    records: dict[pd.Timestamp, CyclePrices] = {}
    for row in frame.to_dict("records"):
        record = CyclePrices(
            cycle_start_ts=utc_timestamp(pd.Timestamp(row["cycle_start_ts"])),
            strike_open=float(row["strike_open"]),
            strike_available_at=utc_timestamp(pd.Timestamp(row["strike_available_at"])),
            cycle_settle=None if pd.isna(row.get("cycle_settle")) else float(row["cycle_settle"]),
            settle_available_at=None if pd.isna(row.get("settle_available_at")) else utc_timestamp(pd.Timestamp(row["settle_available_at"])),
            source=str(row["source"]),
        )
        if record.cycle_start_ts in records:
            raise ValueError("duplicate official cycle record")
        records[record.cycle_start_ts] = record
    return records


def proxy_cycle_prices(candles: pd.DataFrame, as_of: pd.Timestamp) -> dict[pd.Timestamp, CyclePrices]:
    """Explicit research proxy, NEVER official Kalshi settlement data.

    Opening proxy becomes available only when the first minute is closed
    (or later if available_at says so). Outcome is the last minute close.
    """
    cycles: dict[pd.Timestamp, CyclePrices] = {}
    for start, row in aggregate_15m(candles, as_of).iterrows():
        strike_at = start + MINUTE
        settle_at = start + CYCLE
        if "available_at" in candles:
            strike_at = max(strike_at, utc_timestamp(candles.loc[start, "available_at"]))
            settle_at = max(settle_at, utc_timestamp(candles.loc[start + CYCLE - MINUTE, "available_at"]))
        cycles[start] = CyclePrices(
            start, float(row.spot_open), strike_at,
            f"SPOT_PROXY:{candles.attrs.get('source', 'csv')}",
            float(row.spot_close), settle_at,
        )
    return cycles


def build_dataset(
    candles: pd.DataFrame,
    cycles: dict[pd.Timestamp, CyclePrices] | None = None,
    *,
    as_of: pd.Timestamp | None = None,
    order_books: pd.DataFrame | Sequence[OrderBookSnapshot] | None = None,
) -> TrainingDataset:
    """Sample once per completed minute, binding only the active strike record.

    Each sample uses that minute's observed close as current_spot at its actual
    availability time. Outcomes use settle > strike, with ties excluded. A
    missing cycles argument explicitly selects documented SPOT_PROXY mode.
    """
    if not len(candles):
        raise ValueError("no candles supplied")
    books = order_books if order_books is not None else candles.attrs.get("order_books")
    if not isinstance(books, (pd.DataFrame, tuple)):
        raise ValueError("real historical Coinbase L2 snapshots required; supply --obi-csv")
    candles = candles.copy()
    recorded_books = snapshot_records(books)
    candles.attrs.pop("order_books", None)
    horizon = utc_timestamp(as_of) if as_of is not None else candles.index[-1] + MINUTE
    records = proxy_cycle_prices(candles, horizon) if cycles is None else cycles
    mode = "SPOT_PROXY" if cycles is None else "OFFICIAL_PRICES"
    rows: list[dict[str, object]] = []
    for timestamp, candle in candles.iterrows():
        decision = timestamp + MINUTE
        if "available_at" in candles:
            decision = max(decision, utc_timestamp(candle.available_at))
        if decision > horizon:
            continue
        start = cycle_start(decision)
        record = records.get(start)
        if record is None or record.cycle_settle is None or record.settle_available_at is None:
            continue
        if record.settle_available_at > horizon or record.cycle_settle == record.strike_open:
            continue
        # A candle that arrived late is not a current tick for a later cycle.
        if decision >= timestamp + 2 * MINUTE:
            continue
        history = candles.loc[(candles.index >= decision.floor("min") - 61 * MINUTE) & (candles.index < decision.floor("min"))]
        try:
            imbalance = imbalance_at(recorded_books, decision)
            features = compute_features(bind_cycle_prices(history, [record]), float(candle.close), decision, start, order_book_imbalance=imbalance)
            volatility = intracycle_volatility(history, float(candle.close), decision, start)
        except (FeatureDataError, ValueError):
            continue
        rows.append({
            "timestamp": decision, "cycle_start_ts": start,
            "label_available_at": record.settle_available_at,
            "target": int(record.cycle_settle > record.strike_open), **features,
            "cycle_volatility": volatility,
        })
    if not rows:
        raise ValueError("no usable samples: require warmup and completed, observable cycle records")
    samples = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if samples.index.has_duplicates:
        raise ValueError("ambiguous duplicate decision times")
    return TrainingDataset(samples, candles, records, mode, recorded_books)
