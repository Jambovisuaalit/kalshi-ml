"""Read-only, append-only Kalshi KXBTC15M quote and settlement observer.

Each HTTP response is recorded with its own request/receipt interval and exact
source URL. A market response and an orderbook response are separate events:
they are never presented as one atomic exchange snapshot. No order methods,
credentials, model predictions, or inferred settlements are used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import fcntl
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping

from kalshi_ml.config import KalshiEnvironment
from kalshi_ml.execution.kalshi_client import (
    API_ROOT, BASE_URLS, InvalidResponseError, KalshiClient, KalshiError,
)


def _stamp(now: Callable[[], datetime]) -> str:
    value = now()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observer clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _number(value: object, name: str, *, upper: Decimal | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InvalidResponseError(f"invalid {name}")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise InvalidResponseError(f"invalid {name}") from None
    if not result.is_finite() or result < 0 or (upper is not None and result > upper):
        raise InvalidResponseError(f"out-of-range {name}")
    return result


def _best_bid(levels: object, *, fixed: bool) -> tuple[Decimal | None, Decimal]:
    if not isinstance(levels, list):
        raise InvalidResponseError("missing orderbook side")
    best: Decimal | None = None
    size = Decimal(0)
    seen: set[Decimal] = set()
    for level in levels:
        if not isinstance(level, list) or len(level) != 2:
            raise InvalidResponseError("invalid orderbook level")
        price = _number(level[0], "book price", upper=Decimal(1) if fixed else Decimal(100))
        price = price if fixed else price / 100
        quantity = _number(level[1], "book size")
        if price in seen:
            raise InvalidResponseError("duplicate orderbook level")
        seen.add(price)
        if quantity > 0 and (best is None or price > best):
            best, size = price, quantity
    return best, size


def _book_quote(body: Mapping[str, Any]) -> dict[str, str | None]:
    fixed = "orderbook_fp" in body
    book = body.get("orderbook_fp" if fixed else "orderbook")
    if not isinstance(book, dict):
        raise InvalidResponseError("missing orderbook")
    yes, yes_size = _best_bid(book.get("yes_dollars" if fixed else "yes"), fixed=fixed)
    no, no_size = _best_bid(book.get("no_dollars" if fixed else "no"), fixed=fixed)
    if yes is not None and no is not None and yes + no > 1:
        raise InvalidResponseError("crossed single-book")
    def out(value: Decimal | None) -> str | None:
        return str(value) if value is not None else None
    return {
        "yes_bid_dollars": out(yes),
        "yes_ask_dollars": out(1 - no) if no is not None else None,
        "no_bid_dollars": out(no),
        "no_ask_dollars": out(1 - yes) if yes is not None else None,
        "yes_bid_size": str(yes_size),
        "no_bid_size": str(no_size),
        "quote_basis": "single_book_best_bid_and_complementary_ask",
    }


def _official_result(row: Mapping[str, Any]) -> tuple[str, str | None] | None:
    status = row.get("status")
    if status not in {"settled", "finalized"}:
        return None
    result = row.get("result")
    value = row.get("settlement_value_dollars")
    payout = _number(value, "settlement value", upper=Decimal(1)) if value is not None else None
    if result not in {None, "", "yes", "no"}:
        raise InvalidResponseError("invalid official result")
    if payout is None and result not in {"yes", "no"}:
        return None
    if payout is None:
        payout = Decimal(int(result == "yes"))
    if result in {"yes", "no"} and payout != Decimal(int(result == "yes")):
        raise InvalidResponseError("conflicting settlement fields")
    return str(payout), result if result in {"yes", "no"} else None


@dataclass(frozen=True, slots=True)
class ObservationJournal:
    """Local JSONL audit trail; each line is one complete fsynced event."""

    directory: Path

    @property
    def path(self) -> Path:
        return self.directory / "kalshi_observations.jsonl"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Allow one writer per directory for the whole collector lifetime."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.directory / "kalshi_observations.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another observer already owns this journal") from None
            yield
        finally:
            os.close(fd)

    def append(self, event: Mapping[str, object]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def pending_tickers(self, environment: KalshiEnvironment, series_ticker: str) -> set[str]:
        """Resume only same-environment/series contracts after a restart."""
        if not self.path.exists():
            return set()
        observed: set[str] = set()
        settled: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("ticker"), str):
                    raise ValueError("invalid observation journal")
                if event.get("schema") != "kalshi-public-observation-v1" or event.get("environment") != environment.value:
                    raise ValueError("journal schema/environment mismatch; use another output directory")
                if not event["ticker"].startswith(series_ticker + "-"):
                    raise ValueError("journal series mismatch; use another output directory")
                if event.get("kind") == "market":
                    observed.add(event["ticker"])
                elif event.get("kind") == "settlement":
                    settled.add(event["ticker"])
        return observed - settled


class KalshiObserver:
    """Collect public orderbook evidence and official settled-market results."""

    def __init__(
        self, client: KalshiClient, journal: ObservationJournal, *,
        series_ticker: str = "KXBTC15M",
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not series_ticker or not series_ticker.isalnum():
            raise ValueError("invalid series ticker")
        self.client, self.journal, self.series_ticker, self.now = client, journal, series_ticker, now
        self.pending = journal.pending_tickers(client.environment, series_ticker)
        self.base_url = BASE_URLS[client.environment]

    def _read(self, ticker: str, kind: str) -> tuple[dict[str, Any], dict[str, object]]:
        path = f"{API_ROOT}/markets/{ticker}" + ("/orderbook" if kind == "orderbook" else "")
        started = _stamp(self.now)
        body = (self.client.get_public_orderbook_response(ticker) if kind == "orderbook"
                else self.client.get_public_market_response(ticker))
        received = _stamp(self.now)
        if received < started:
            raise ValueError("observer clock moved backwards during an API request")
        return body, {
            "schema": "kalshi-public-observation-v1", "kind": kind, "ticker": ticker,
            "request_started_at": started, "received_at": received,
            "source_url": self.base_url + path, "environment": self.client.environment.value,
            "raw_response": body,
        }

    def observe_once(self) -> tuple[int, int]:
        """Observe the active book, then poll pending contracts for settlement."""
        quotes = settlements = 0
        active = self.client.get_active_15m_market(self.series_ticker)
        verified_open_ticker: str | None = None
        if active is not None:
            ticker = active.ticker
            if not ticker.startswith(self.series_ticker + "-"):
                raise InvalidResponseError("wrong series from market discovery")
            body, event = self._read(ticker, "market")
            row = body.get("market")
            if not isinstance(row, dict) or row.get("ticker") != ticker:
                raise InvalidResponseError("wrong market response")
            if row.get("exchange_index", active.exchange_index) != active.exchange_index:
                raise InvalidResponseError("market shard changed")
            if row.get("status") in {"open", "active"} and row.get("result") in {None, ""}:
                verified_open_ticker = ticker
                self.journal.append(event)
                self.pending.add(ticker)
                book, book_event = self._read(ticker, "orderbook")
                book_event["top_of_book"] = _book_quote(book)
                self.journal.append(book_event)
                quotes += 1
        for ticker in sorted(self.pending):
            if ticker == verified_open_ticker:
                continue  # The same response just confirmed this contract is still open.
            body, event = self._read(ticker, "market")
            row = body.get("market")
            if not isinstance(row, dict) or row.get("ticker") != ticker:
                raise InvalidResponseError("wrong settlement market response")
            result = _official_result(row)
            if result is None:
                continue
            event["kind"] = "settlement"
            event["yes_payout_dollars"], event["result"] = result
            settlement_ts = row.get("settlement_ts")
            if settlement_ts is not None:
                if not isinstance(settlement_ts, str):
                    raise InvalidResponseError("invalid official settlement timestamp")
                try:
                    parsed = datetime.fromisoformat(settlement_ts.replace("Z", "+00:00"))
                except ValueError:
                    raise InvalidResponseError("invalid official settlement timestamp") from None
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise InvalidResponseError("official settlement timestamp lacks timezone")
            event["official_settlement_ts"] = settlement_ts
            self.journal.append(event)
            self.pending.remove(ticker)
            settlements += 1
        return quotes, settlements


def main(argv: list[str] | None = None) -> int:
    """Run a bounded or continuous read-only observer; Ctrl-C closes cleanly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("PROD", "DEMO"), default="PROD")
    parser.add_argument("--output", type=Path, default=Path("data/raw/kalshi"))
    parser.add_argument("--poll-sec", type=float, default=4.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.poll_sec <= 3600:
        parser.error("--poll-sec must be between 1 and 3600")
    try:
        journal = ObservationJournal(args.output)
        with journal.exclusive(), KalshiClient(environment=KalshiEnvironment(args.environment)) as client:
            observer = KalshiObserver(client, journal)
            while True:
                quotes, settlements = observer.observe_once()
                print(f"quotes={quotes} settlements={settlements} pending={len(observer.pending)}", flush=True)
                if args.once:
                    return 0
                time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        return 0
    except (KalshiError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"observer stopped: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
