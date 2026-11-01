"""Public BTC/USD market data with bounded retries and explicit source lineage.

Coinbase candles: https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles
Kraken OHLC: https://docs.kraken.com/api-reference/market-data/get-ohlc-data

All frames use a UTC DatetimeIndex named ``timestamp`` representing minute
OPEN times. Only completed candles are returned. Missing minutes are errors,
never forward-filled. A fallback replaces the entire requested window rather
than silently mixing exchange prices within one feature window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
import logging
import math
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from kalshi_ml.data.orderbook import DEPTH, OrderBookSnapshot

LOGGER = logging.getLogger(__name__)
COINBASE_URL = "https://api.exchange.coinbase.com"
KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
PAGE_MINUTES = 250
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataError(RuntimeError):
    """A source failed, returned invalid data, or cannot cover the time range."""


class DataCoverageError(MarketDataError):
    """At least one requested completed minute is unavailable."""


class _TransientError(MarketDataError):
    """An exchange-level error that can be retried safely for GET requests."""


def utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Require an unambiguous timezone-aware timestamp and normalize to UTC."""
    if not isinstance(value, (datetime, pd.Timestamp)):
        raise TypeError("timestamp must be a timezone-aware datetime or Timestamp")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("timestamp must be finite and timezone-aware")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """GET retry budget; max_retries excludes the initial attempt.

    Retry-After longer than max_delay_sec aborts instead of retrying too early.
    Exponential delays include up to 25% jitter and remain capped.
    """

    max_retries: int = 3
    base_delay_sec: float = 0.5
    max_delay_sec: float = 30.0
    timeout_sec: float = 10.0

    def __post_init__(self) -> None:
        """Reject unusable retry budgets and nonfinite timing values."""
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be an integer in [0, 10]")
        for name in ("base_delay_sec", "max_delay_sec", "timeout_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.base_delay_sec > self.max_delay_sec:
            raise ValueError("base_delay_sec exceeds max_delay_sec")


@dataclass(frozen=True, slots=True)
class SpotTicker:
    """Coinbase last-trade price and separate exchange/receipt timestamps."""

    price: Decimal
    timestamp: pd.Timestamp
    received_at: pd.Timestamp
    source: str = "coinbase"


class MarketDataCollector:
    """Synchronous public REST client; inject session, clock and sleep for tests.

    Instances own only sessions they create and are not intended to be shared
    across threads. Requests are paced per instance; callers sharing an IP
    should also coordinate their aggregate request rate.
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        min_request_interval_sec: float = 0.35,
    ) -> None:
        if not math.isfinite(min_request_interval_sec) or min_request_interval_sec < 0:
            raise ValueError("min_request_interval_sec must be finite and nonnegative")
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self.retry = retry if retry is not None else RetryPolicy()
        self._sleep, self._clock = sleep, clock
        self._monotonic, self._jitter = monotonic, jitter
        self._min_interval = min_request_interval_sec
        self._last_request: float | None = None

    def __enter__(self) -> MarketDataCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close a client-owned session; leave injected sessions untouched."""
        if self._owns_session:
            self.session.close()

    def _retry_after(self, header: str | None) -> float:
        if header is None:
            return 0.0
        try:
            delay = float(header)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(header) - self._clock()).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return 0.0
        return max(0.0, delay) if math.isfinite(delay) else self.retry.max_delay_sec + 1

    def _get_json(self, url: str, params: dict[str, str | int] | None = None) -> Any:
        """Retry network failures, 408/429/5xx and Kraken throttling envelopes."""
        for attempt in range(self.retry.max_retries + 1):
            retry_after = 0.0
            try:
                now = self._monotonic()
                if self._last_request is not None:
                    self._sleep(max(0.0, self._min_interval - (now - self._last_request)))
                self._last_request = self._monotonic()
                with self.session.get(
                    url, params=params, timeout=self.retry.timeout_sec,
                    headers={"Accept": "application/json", "User-Agent": "kalshi-ml/0.1"},
                ) as response:
                    if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                        retry_after = self._retry_after(response.headers.get("Retry-After"))
                        raise _TransientError(f"transient HTTP status {response.status_code}")
                    if not 200 <= response.status_code < 300:
                        raise MarketDataError(f"nonretryable HTTP status {response.status_code}")
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise _TransientError("invalid JSON response") from exc
                    if isinstance(payload, dict) and payload.get("error"):
                        errors = payload["error"]
                        if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
                            raise MarketDataError("invalid Kraken error envelope")
                        if any("Rate limit" in e or "Throttled" in e or "Unavailable" in e for e in errors):
                            for error in errors:
                                if "Throttled:" in error:
                                    try:
                                        retry_after = max(retry_after, float(error.rsplit(":", 1)[1]) - self._clock().timestamp())
                                    except ValueError:
                                        pass
                            raise _TransientError("Kraken temporarily unavailable or throttled")
                        raise MarketDataError("Kraken rejected public data request")
                    return payload
            except (requests.Timeout, requests.ConnectionError, _TransientError) as exc:
                if attempt == self.retry.max_retries or retry_after > self.retry.max_delay_sec:
                    raise MarketDataError("GET retry budget exhausted or Retry-After exceeds budget") from exc
                delay = min(
                    self.retry.max_delay_sec,
                    self.retry.base_delay_sec * 2**attempt * (1 + 0.25 * self._jitter()),
                )
                self._sleep(max(delay, retry_after))
            except requests.RequestException as exc:
                raise MarketDataError("public data HTTP request failed") from exc
        raise AssertionError("unreachable retry state")

    def fetch_candles(
        self, start: datetime | pd.Timestamp, end: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return complete one-minute candles in the half-open range [start, end).

        Bounds must lie on whole minutes, and end must be at or before the
        current minute. Coinbase uses exact 250-minute pages (a shorter final
        page is allowed). If any page or coverage check fails, request the whole
        window from Kraken. Kraken only exposes the latest 720 OHLC entries;
        coverage checks reject unsupported historical requests explicitly.
        """
        start, end = utc_timestamp(start), utc_timestamp(end)
        if start != start.floor("min") or end != end.floor("min") or start >= end:
            raise ValueError("require ordered whole-minute bounds")
        if end > utc_timestamp(self._clock()).floor("min"):
            raise ValueError("end must exclude the currently forming candle")
        try:
            return self._coinbase_candles(start, end)
        except MarketDataError as primary:
            LOGGER.warning("Coinbase window unavailable; attempting Kraken: %s", type(primary).__name__)
            try:
                return self._kraken_candles(start, end)
            except MarketDataError as fallback:
                raise DataCoverageError(
                    f"both sources failed: Coinbase={type(primary).__name__}, "
                    f"Kraken={type(fallback).__name__}; window not returned"
                ) from fallback

    def _coinbase_candles(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows: list[list[object]] = []
        cursor = start
        while cursor < end:
            page_end = min(cursor + pd.Timedelta(minutes=PAGE_MINUTES), end)
            payload = self._get_json(
                f"{COINBASE_URL}/products/BTC-USD/candles",
                {"start": cursor.isoformat(), "end": page_end.isoformat(), "granularity": 60},
            )
            if not isinstance(payload, list):
                raise MarketDataError("Coinbase candle payload must be an array")
            for row in payload:
                if not isinstance(row, list) or len(row) != 6:
                    raise MarketDataError("invalid Coinbase candle row")
                # Exchange ordering is time, low, high, open, close, volume.
                rows.append([row[0], row[3], row[2], row[1], row[4], row[5]])
            cursor = page_end
        return self._frame(rows, start, end, "coinbase")

    def _kraken_candles(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        payload = self._get_json(KRAKEN_URL, {"pair": "XBTUSD", "interval": 1, "since": int(start.timestamp())})
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            raise MarketDataError("invalid Kraken OHLC payload")
        pairs = [key for key in payload["result"] if key != "last"]
        if len(pairs) != 1:
            raise MarketDataError("unexpected Kraken pair count")
        candles = payload["result"][pairs[0]]
        if not isinstance(candles, list):
            raise MarketDataError("invalid Kraken candle array")
        rows: list[list[object]] = []
        # The last Kraken row is ALWAYS uncommitted, even for historical since.
        for row in candles[:-1]:
            if not isinstance(row, list) or len(row) != 8:
                raise MarketDataError("invalid Kraken candle row")
            rows.append([row[0], row[1], row[2], row[3], row[4], row[6]])
        return self._frame(rows, start, end, "kraken")

    @staticmethod
    def _frame(
        rows: list[list[object]], start: pd.Timestamp, end: pd.Timestamp, source: str,
    ) -> pd.DataFrame:
        if not rows:
            raise DataCoverageError("empty candle response")
        try:
            frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
            seconds = pd.to_numeric(frame.pop("timestamp"), errors="raise")
            frame.index = pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", utc=True), name="timestamp")
            frame = frame.loc[(frame.index >= start) & (frame.index < end)].astype(float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise MarketDataError("invalid candle values") from exc
        # Identical page-boundary duplicates are harmless; conflicting data is not.
        if frame.index.has_duplicates:
            for _, group in frame.groupby(level=0):
                if len(group.drop_duplicates()) > 1:
                    raise MarketDataError("conflicting duplicate candles")
            frame = frame.loc[~frame.index.duplicated()]
        frame = frame.sort_index()
        expected = pd.date_range(start, end, freq="min", inclusive="left", name="timestamp")
        if not frame.index.equals(expected):
            raise DataCoverageError("missing or misaligned one-minute candles")
        prices = frame[["open", "high", "low", "close"]]
        if (
            not np.isfinite(frame.to_numpy()).all()
            or (prices <= 0).any().any() or (frame.volume < 0).any()
            or (frame.low > prices.min(axis=1)).any()
            or (frame.high < prices.max(axis=1)).any()
        ):
            raise MarketDataError("nonfinite or inconsistent OHLCV values")
        frame.attrs["source"] = source
        frame.attrs["timestamp_semantics"] = "minute_open_utc"
        return frame

    def fetch_order_book(self, *, max_age_sec: float = 10.0) -> OrderBookSnapshot:
        """Read Coinbase L2 and sum BTC size at ten best levels on each side.

        Both sides must provide ten unique, uncrossed levels. A stale exchange
        timestamp, missing time, malformed size or transport error blocks OBI;
        Kraken candles cannot substitute for Coinbase microstructure.
        """
        if not math.isfinite(max_age_sec) or max_age_sec <= 0:
            raise ValueError("max_age_sec must be positive")
        payload = self._get_json(f"{COINBASE_URL}/products/BTC-USD/book", {"level": 2})
        received = utc_timestamp(self._clock())
        try:
            levels: dict[str, list[tuple[Decimal, Decimal]]] = {}
            for side in ("bids", "asks"):
                entries = [(Decimal(str(row[0])), Decimal(str(row[1]))) for row in payload[side]]
                if (len(entries) < DEPTH or len({price for price, _ in entries}) != len(entries)
                        or not all(price.is_finite() and size.is_finite() and price > 0 and size > 0 for price, size in entries)):
                    raise ValueError("invalid depth/price/size")
                levels[side] = sorted(entries, reverse=side == "bids")[:DEPTH]
            if levels["bids"][0][0] >= levels["asks"][0][0]:
                raise ValueError("crossed Coinbase book")
            snapshot = OrderBookSnapshot(
                utc_timestamp(pd.Timestamp(payload["time"])), received,
                float(sum(size for _, size in levels["bids"])),
                float(sum(size for _, size in levels["asks"])), payload["sequence"],
            )
            if (received - snapshot.timestamp).total_seconds() > max_age_sec:
                raise ValueError("stale Coinbase book")
        except (KeyError, IndexError, TypeError, ValueError, InvalidOperation, ArithmeticError) as exc:
            raise MarketDataError("invalid or stale Coinbase order book") from exc
        return snapshot

    def fetch_ticker(self, *, max_age_sec: float = 10.0) -> SpotTicker:
        """Fetch Coinbase last trade; reject stale or >2s future exchange times."""
        if not math.isfinite(max_age_sec) or max_age_sec <= 0:
            raise ValueError("max_age_sec must be finite and positive")
        payload = self._get_json(f"{COINBASE_URL}/products/BTC-USD/ticker")
        try:
            price = Decimal(str(payload["price"]))
            timestamp = utc_timestamp(pd.Timestamp(payload["time"]))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise MarketDataError("invalid Coinbase ticker") from exc
        received = utc_timestamp(self._clock())
        age = (received - timestamp).total_seconds()
        if not price.is_finite() or price <= 0 or not -2 <= age <= max_age_sec:
            raise MarketDataError("nonpositive price or stale/future ticker")
        return SpotTicker(price=price, timestamp=timestamp, received_at=received)
