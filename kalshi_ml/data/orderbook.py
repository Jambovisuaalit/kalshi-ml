"""Point-in-time Coinbase top-ten-level imbalance, with explicit receipt times.

OHLCV cannot reconstruct this feature. Collect and retain actual snapshots;
never backfill historical decisions with today's order book or zero values.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Sequence
import csv
from dataclasses import asdict, dataclass
import fcntl
import math
import os
from pathlib import Path
import time
from typing import TextIO

import pandas as pd

DEPTH = 10
MAX_AGE_SEC = 10.0


def aware(value: object) -> pd.Timestamp:
    """Require an actual timezone-aware instant; never infer a timezone."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("order-book times must be timezone-aware")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Aggregated BTC sizes at the best ten price levels on each side."""

    timestamp: pd.Timestamp
    received_at: pd.Timestamp
    bid_volume: float
    ask_volume: float
    sequence: int
    depth: int = DEPTH

    def __post_init__(self) -> None:
        timestamp, received = aware(self.timestamp), aware(self.received_at)
        if timestamp > received or self.depth != DEPTH:
            raise ValueError("future exchange timestamp or unsupported OBI depth")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("invalid order-book sequence")
        if not all(math.isfinite(x) and x > 0 for x in (self.bid_volume, self.ask_volume, self.bid_volume + self.ask_volume)):
            raise ValueError("both order-book sizes must be finite and positive")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "received_at", received)

    @property
    def imbalance(self) -> float:
        """(Bid BTC size - ask BTC size) / total BTC size, in [-1, 1]."""
        ratio = self.bid_volume / (self.bid_volume + self.ask_volume)
        return 2 * ratio - 1


def validate_snapshots(frame: pd.DataFrame | Sequence[OrderBookSnapshot]) -> pd.DataFrame:
    """Validate recorded snapshots and preserve their actual availability."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame([asdict(snapshot) for snapshot in frame])
    for name in ("sequence", "depth"):
        if name not in frame or not all(float(x).is_integer() for x in frame[name]):
            raise ValueError("OBI sequence and depth must be integers")
    rows = [asdict(OrderBookSnapshot(
        aware(row["timestamp"]), aware(row["received_at"]), float(row["bid_volume"]),
        float(row["ask_volume"]), int(row["sequence"]), int(row["depth"]),
    )) for row in frame.to_dict("records")]
    if not rows:
        raise ValueError("historical Coinbase OBI snapshots are required")
    result = pd.DataFrame(rows).sort_values("received_at")
    if result.received_at.duplicated().any():
        raise ValueError("duplicate OBI receipt times")
    if not result.sequence.is_monotonic_increasing:
        raise ValueError("Coinbase sequence regressed; inspect recorded data")
    return result.reset_index(drop=True)


def snapshot_records(frame: pd.DataFrame | Sequence[OrderBookSnapshot]) -> tuple[OrderBookSnapshot, ...]:
    """Immutable records are safe in pandas attrs, including concat/copy."""
    return tuple(OrderBookSnapshot(**row) for row in validate_snapshots(frame).to_dict("records"))


def imbalance_at(frame: pd.DataFrame | Sequence[OrderBookSnapshot], decision: pd.Timestamp) -> float:
    """Backward as-of lookup with freshness on BOTH exchange and receipt time."""
    now = aware(decision)
    records = snapshot_records(frame) if isinstance(frame, pd.DataFrame) else frame
    offset = bisect_right(records, now, key=lambda row: row.received_at) - 1
    if offset < 0:
        raise ValueError("no observable Coinbase OBI snapshot")
    row = records[offset]
    if max((now - row.received_at).total_seconds(), (now - row.timestamp).total_seconds()) > MAX_AGE_SEC:
        raise ValueError("Coinbase OBI snapshot is stale")
    return row.imbalance


def _resume_tail(stream: TextIO) -> OrderBookSnapshot | None:
    """Validate an existing CSV before appending; return its latest snapshot."""
    stream.seek(0)
    reader = csv.DictReader(stream)
    expected = list(OrderBookSnapshot.__dataclass_fields__)
    if reader.fieldnames is None:
        return None
    if reader.fieldnames != expected:
        raise ValueError("incompatible existing OBI CSV header")
    last: OrderBookSnapshot | None = None
    for row in reader:
        if set(row) != set(expected) or any(value is None for value in row.values()):
            raise ValueError("incomplete existing OBI CSV row")
        try:
            snapshot = OrderBookSnapshot(
                aware(row["timestamp"]), aware(row["received_at"]),
                float(row["bid_volume"]), float(row["ask_volume"]),
                int(row["sequence"]), int(row["depth"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid existing OBI CSV row") from exc
        if last is not None and (
            snapshot.sequence <= last.sequence or snapshot.timestamp < last.timestamp
            or snapshot.received_at <= last.received_at
        ):
            raise ValueError("existing OBI observations are out of order")
        last = snapshot
    return last


def main() -> int:
    """Record public Coinbase books to CSV; never submit exchange orders."""
    from kalshi_ml.data.collector import MarketDataCollector, MarketDataError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/coinbase_obi.csv"))
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=0, help="0: continue until Ctrl+C")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1 or args.samples < 0:
        parser.error("interval must be >=1 second; samples must be >=0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a+", newline="") as stream, MarketDataCollector() as collector:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            expected = list(OrderBookSnapshot.__dataclass_fields__)
            last = _resume_tail(stream)
            stream.seek(0, 2)
            writer = csv.DictWriter(stream, fieldnames=expected)
            if stream.tell() == 0:
                writer.writeheader()
                stream.flush()
                os.fsync(stream.fileno())
            count = 0
            while args.samples == 0 or count < args.samples:
                begun = time.monotonic()
                try:
                    snapshot = collector.fetch_order_book()
                    if last is not None and (
                        snapshot.sequence <= last.sequence or snapshot.timestamp < last.timestamp
                        or snapshot.received_at <= last.received_at
                    ):
                        print("Coinbase book sequence/time did not advance; snapshot skipped", flush=True)
                    else:
                        writer.writerow(asdict(snapshot))
                        stream.flush()
                        os.fsync(stream.fileno())
                        last = snapshot
                        count += 1
                except MarketDataError:
                    print("Coinbase book unavailable; no synthetic snapshot written", flush=True)
                time.sleep(max(0, args.interval - (time.monotonic() - begun)))
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"OBI recording failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
