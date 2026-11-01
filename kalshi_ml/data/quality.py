"""Point-in-time quality audit for recorded Coinbase OBI and Kalshi outcomes.

The audit reads local observations only. It never requests prices, places
orders, backfills missing books, or treats visible quotes as executed fills.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import pandas as pd

from kalshi_ml.data.kalshi_observer import _book_quote, _official_result
from kalshi_ml.data.orderbook import MAX_AGE_SEC, OrderBookSnapshot, aware, snapshot_records
from kalshi_ml.execution.kalshi_client import BASE_URLS, API_ROOT, InvalidResponseError
from kalshi_ml.config import KalshiEnvironment


def _utc(value: object, field: str) -> pd.Timestamp:
    try:
        return aware(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid UTC-aware {field}") from None


def _longest_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def audit_obi(
    path: Path, *, as_of: pd.Timestamp, minutes: int = 3000,
    max_age_sec: float = MAX_AGE_SEC,
) -> dict[str, Any]:
    """Evaluate backward as-of book availability at every minute boundary.

    Both Coinbase's exchange timestamp and local receipt must precede the
    decision and be no older than max_age_sec. The newest received snapshot is
    selected, just as inference/training would see it; later rows are excluded.
    """
    if type(minutes) is not int or minutes < 1:
        raise ValueError("minutes must be a positive integer")
    if not math.isfinite(max_age_sec) or max_age_sec <= 0:
        raise ValueError("max_age_sec must be finite and positive")
    payload = path.read_bytes()
    partial_tail = bool(payload) and not payload.endswith(b"\n")
    if partial_tail:
        payload = payload.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in payload else b""
    if not payload:
        raise ValueError("OBI CSV has no complete header")
    frame = pd.read_csv(BytesIO(payload))
    books: tuple[OrderBookSnapshot, ...] = snapshot_records(frame) if len(frame) else ()
    end = _utc(as_of, "as_of").floor("min")
    decisions = pd.date_range(end=end, periods=minutes, freq="min")
    receipt_times = [book.received_at for book in books]
    status: list[str] = []
    ages: list[float] = []
    for decision in decisions:
        offset = bisect_right(receipt_times, decision) - 1
        if offset < 0:
            status.append("missing")
            continue
        book = books[offset]
        age = max((decision - book.timestamp).total_seconds(),
                  (decision - book.received_at).total_seconds())
        if age > max_age_sec:
            status.append("stale")
            continue
        status.append("fresh")
        ages.append(age)
    count = status.count("fresh")
    observed = [book for book in books if book.received_at <= end]
    gaps = [(right.received_at - left.received_at).total_seconds()
            for left, right in zip(observed, observed[1:])]
    return {
        "decision_start": decisions[0].isoformat(),
        "decision_end": decisions[-1].isoformat(),
        "decision_minutes": minutes,
        "recorded_snapshots": len(observed),
        "partial_tail_ignored": partial_tail,
        "first_snapshot_received_at": observed[0].received_at.isoformat() if observed else None,
        "last_snapshot_received_at": observed[-1].received_at.isoformat() if observed else None,
        "fresh_minutes": count,
        "missing_minutes": status.count("missing"),
        "stale_minutes": status.count("stale"),
        "coverage": count / minutes,
        "longest_unusable_run_minutes": _longest_run([item != "fresh" for item in status]),
        "max_snapshot_gap_sec": max(gaps) if gaps else None,
        "fresh_age_p95_sec": float(pd.Series(ages).quantile(0.95)) if ages else None,
        "max_age_sec": max_age_sec,
    }


@dataclass(slots=True)
class _EventStream:
    """Bounded streaming file snapshot; retain no raw pages after validation."""

    path: Path
    count: int = 0
    partial_tail_ignored: bool = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            while stream.tell() < size:
                line = stream.readline(size - stream.tell())
                if not line.endswith(b"\n"):
                    self.partial_tail_ignored = True
                    break
                self.count += 1
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ValueError(f"invalid Kalshi journal JSON at line {self.count}") from None
                if not isinstance(event, dict):
                    raise ValueError(f"invalid Kalshi journal event at line {self.count}")
                event["_line"] = self.count
                yield event


def _dollars(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError("invalid quote price") from None
    if not result.is_finite() or not 0 <= result <= 1:
        raise ValueError("quote price outside [0, 1]")
    return result


def audit_kalshi(path: Path, *, as_of: pd.Timestamp) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Join pre-settlement orderbook observations to official binary outcomes.

    The join is on ticker, with receipt-time and official settlement-time
    guards. Separate orderbook and market GETs remain separate observations.
    A duplicate or contradictory official result fails closed.
    """
    horizon = _utc(as_of, "as_of")
    events = _EventStream(path)
    settlements: dict[str, dict[str, object]] = {}
    quotes: list[dict[str, object]] = []
    environments: set[str] = set()
    market_events = 0
    for event in events:
        kind, ticker, environment = event.get("kind"), event.get("ticker"), event.get("environment")
        if kind not in {"market", "orderbook", "settlement"} or not isinstance(ticker, str) or not ticker.startswith("KXBTC15M-"):
            raise ValueError(f"invalid Kalshi event identity at line {event['_line']}")
        if environment not in {member.value for member in KalshiEnvironment}:
            raise ValueError("invalid Kalshi observation environment")
        environments.add(environment)
        started = _utc(event.get("request_started_at"), "request start")
        received = _utc(event.get("received_at"), "receipt")
        if started > received:
            raise ValueError("Kalshi response precedes its request")
        source = BASE_URLS[KalshiEnvironment(environment)] + f"{API_ROOT}/markets/{ticker}"
        if kind == "orderbook":
            source += "/orderbook"
        if event.get("source_url") != source or event.get("schema") != "kalshi-public-observation-v1":
            raise ValueError("Kalshi source URL/schema mismatch")
        if received > horizon:
            continue  # Future evidence is unavailable to this as-of audit.
        raw = event.get("raw_response")
        if not isinstance(raw, dict):
            raise ValueError("missing raw Kalshi response")
        if kind == "market":
            if not isinstance(raw.get("market"), dict) or raw["market"].get("ticker") != ticker:
                raise ValueError("market response ticker mismatch")
            market_events += 1
        elif kind == "orderbook":
            parsed = _book_quote(raw)
            if parsed != event.get("top_of_book"):
                raise ValueError("stored Kalshi quote differs from raw orderbook")
            for name in ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars"):
                _dollars(parsed[name])
            quotes.append({
                "ticker": ticker, "quote_received_at": received,
                "quote_source_url": source, "quote_journal_line": event["_line"],
                **{name: parsed[name] for name in ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars")},
            })
        else:
            row = raw.get("market")
            if not isinstance(row, dict) or row.get("ticker") != ticker:
                raise ValueError("settlement response ticker mismatch")
            official = _official_result(row)
            if official is None or official != (event.get("yes_payout_dollars"), event.get("result")):
                raise ValueError("stored settlement differs from raw official response")
            official_ts = row.get("settlement_ts")
            if official_ts is None or official_ts != event.get("official_settlement_ts"):
                raise ValueError("official settlement timestamp is missing or inconsistent")
            settled_at = _utc(official_ts, "official settlement timestamp")
            if settled_at > received:
                raise ValueError("official settlement timestamp follows local receipt")
            payout = _dollars(official[0])
            if payout not in {Decimal(0), Decimal(1)}:
                raise ValueError("KXBTC15M settlement is not binary")
            record = {"yes_payout_dollars": str(payout), "settlement_received_at": received,
                      "official_settlement_ts": settled_at, "settlement_journal_line": event["_line"]}
            if ticker in settlements and settlements[ticker] != record:
                raise ValueError("duplicate/contradictory Kalshi settlement")
            settlements[ticker] = record
    if len(environments) > 1:
        raise ValueError("mixed DEMO/PROD Kalshi journal")
    joined: list[dict[str, object]] = []
    late_quotes = 0
    for quote in quotes:
        settlement = settlements.get(str(quote["ticker"]))
        if settlement is None:
            continue
        if quote["quote_received_at"] >= settlement["official_settlement_ts"]:
            late_quotes += 1
            continue
        joined.append({**quote, **settlement})
    joined.sort(key=lambda row: (row["quote_received_at"], row["ticker"]))
    contracts = sorted({str(row["ticker"]) for row in joined},
                       key=lambda ticker: (settlements[ticker]["official_settlement_ts"], ticker))
    train_size = int(len(contracts) * 0.70)
    train, oos = contracts[:train_size], contracts[train_size:]
    complete = [row for row in joined if all(row[name] is not None for name in
                ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars"))]
    return {
        "environment": next(iter(environments)) if environments else None,
        "journal_events_in_snapshot": events.count, "partial_tail_ignored": events.partial_tail_ignored,
        "market_observations": market_events, "orderbook_observations": len(quotes),
        "official_settlements": len(settlements),
        "joined_pre_settlement_quotes": len(joined),
        "joined_contracts": len(contracts),
        "quotes_after_settlement_excluded": late_quotes,
        "unsettled_quote_contracts": len({str(row["ticker"]) for row in quotes} - settlements.keys()),
        "complete_joined_quotes": len(complete),
        "complete_joined_contracts": len({str(row["ticker"]) for row in complete}),
        "chronological_contract_split": {
            "train_70pct": len(train), "out_of_sample_30pct": len(oos),
            "train_yes": sum(Decimal(str(settlements[ticker]["yes_payout_dollars"])) == 1 for ticker in train),
            "out_of_sample_yes": sum(Decimal(str(settlements[ticker]["yes_payout_dollars"])) == 1 for ticker in oos),
            "first_oos_settlement_ts": settlements[oos[0]]["official_settlement_ts"].isoformat() if oos else None,
        },
        "quote_fill_assumption": "none; displayed prices are observations only",
    }, joined


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".quality-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Write a reproducible JSON report and joined quote/settlement CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obi-csv", type=Path, default=Path("data/raw/coinbase_obi.csv"))
    parser.add_argument("--kalshi-jsonl", type=Path, default=Path("data/raw/kalshi/kalshi_observations.jsonl"))
    parser.add_argument("--minutes", type=int, default=3000)
    parser.add_argument("--as-of", help="UTC-aware cutoff; defaults to current UTC minute")
    parser.add_argument("--report", type=Path, default=Path("data/processed/quality_report.json"))
    parser.add_argument("--joined-csv", type=Path, default=Path("data/processed/kalshi_joined_quotes.csv"))
    args = parser.parse_args(argv)
    try:
        horizon = _utc(args.as_of if args.as_of else datetime.now(timezone.utc), "as_of")
        obi = audit_obi(args.obi_csv, as_of=horizon, minutes=args.minutes)
        kalshi, joined = audit_kalshi(args.kalshi_jsonl, as_of=horizon)
        window_start = pd.Timestamp(obi["decision_start"])
        window_end = pd.Timestamp(obi["decision_end"])
        aligned_contracts = {str(row["ticker"]) for row in joined
            if window_start <= row["quote_received_at"] <= window_end
            and all(row[name] is not None for name in
                    ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars"))}
        kalshi["complete_joined_contracts_in_obi_window"] = len(aligned_contracts)
        report = {"schema": "market-data-quality-v1", "as_of": horizon.isoformat(),
                  "obi": obi, "kalshi": kalshi,
                  "collection_ready": args.minutes >= 3000 and obi["coverage"] >= 0.95
                    and len(aligned_contracts) >= 100,
                  "collection_gate": "OBI >=95% of >=3000 minute decisions and >=100 fully quoted joined settled contracts; not a model-performance claim"}
        fields = ("ticker", "quote_received_at", "quote_source_url", "quote_journal_line",
                  "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars",
                  "yes_payout_dollars", "official_settlement_ts", "settlement_received_at", "settlement_journal_line")
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in joined:
            writer.writerow({name: str(row[name]) if row[name] is not None else "" for name in fields})
        _atomic_text(args.joined_csv, buffer.getvalue())
        _atomic_text(args.report, json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"collection_ready": report["collection_ready"],
                          "obi_coverage": obi["coverage"], "joined_contracts": kalshi["joined_contracts"],
                          "report": str(args.report), "joined_csv": str(args.joined_csv)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, InvalidResponseError, pd.errors.ParserError) as exc:
        parser.exit(2, f"quality audit failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
