"""REST contracts and retry/fallback behavior without network access."""

from decimal import Decimal
import unittest
from unittest.mock import Mock

import pandas as pd
import requests

from kalshi_ml.data.collector import (
    DataCoverageError, MarketDataCollector, MarketDataError, RetryPolicy,
)


def response(payload: object, status: int = 200, headers: dict[str, str] | None = None) -> Mock:
    """Build a requests-like response supporting explicit resource cleanup."""
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = payload
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class CollectorTests(unittest.TestCase):
    """Drive REST responses through injected sessions and a fixed clock."""

    def setUp(self) -> None:
        self.now = pd.Timestamp("2026-11-01T12:00:10Z")
        self.end = self.now.floor("min")
        self.session = Mock(spec=requests.Session)
        self.sleeps: list[float] = []
        self.client = MarketDataCollector(
            session=self.session, clock=lambda: self.now.to_pydatetime(),
            sleep=self.sleeps.append, jitter=lambda: 0,
            min_request_interval_sec=0,
        )

    @staticmethod
    def coinbase_rows(start: pd.Timestamp, end: pd.Timestamp) -> list[list[float]]:
        """Return reverse-ordered arrays in Coinbase's native column order."""
        return [[int(t.timestamp()), 99, 102, 100, 101, 2] for t in reversed(pd.date_range(start, end, freq="min", inclusive="left"))]

    def test_exact_250_minute_pages_and_boundary_deduplication(self) -> None:
        """Out-of-range rows and identical boundary overlaps do not corrupt pages."""
        start = self.end - pd.Timedelta(minutes=501)

        def get(url: str, **kwargs: object) -> Mock:
            params = kwargs["params"]
            left, right = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
            # Simulate an inclusive end and data before the requested start.
            return response(self.coinbase_rows(left - pd.Timedelta(minutes=1), right + pd.Timedelta(minutes=1)))

        self.session.get.side_effect = get
        frame = self.client.fetch_candles(start, self.end)
        self.assertEqual(len(frame), 501)
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertEqual(frame.attrs["source"], "coinbase")
        self.assertEqual(frame.iloc[0].to_dict(), {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 2})
        durations = [
            (pd.Timestamp(call.kwargs["params"]["end"]) - pd.Timestamp(call.kwargs["params"]["start"])).total_seconds() / 60
            for call in self.session.get.call_args_list
        ]
        self.assertEqual(durations, [250, 250, 1])

    def test_kraken_fallback_replaces_window_and_discards_last(self) -> None:
        """Parse Kraken OHLC indices and omit the uncommitted row."""
        start = self.end - pd.Timedelta(minutes=2)
        rows = [[int(t.timestamp()), "100", "102", "99", "101", "100.5", "3", 4] for t in pd.date_range(start, self.end, freq="min")]
        self.session.get.side_effect = [response([], 400), response({"error": [], "result": {"XXBTZUSD": rows, "last": 123}})]
        frame = self.client.fetch_candles(start, self.end)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.attrs["source"], "kraken")
        self.assertEqual(frame.volume.tolist(), [3, 3])
        self.assertNotIn(self.end, frame.index)

    def test_both_sources_missing_history_fail_closed(self) -> None:
        """Kraken's history limit cannot silently truncate a training request."""
        self.session.get.side_effect = [response([]), response({"error": [], "result": {"XXBTZUSD": [], "last": 1}})]
        with self.assertRaises(DataCoverageError):
            self.client.fetch_candles(
                self.end - pd.Timedelta(days=2),
                self.end - pd.Timedelta(days=2) + pd.Timedelta(minutes=2),
            )

    def test_rate_limit_and_network_exponential_backoff(self) -> None:
        """Respect Retry-After, then exponentiate on a later transport timeout."""
        self.session.get.side_effect = [
            response({}, 429, {"Retry-After": "2"}), requests.Timeout(),
            response({"price": "100000.25", "time": self.now.isoformat()}),
        ]
        ticker = self.client.fetch_ticker()
        self.assertEqual(ticker.price, Decimal("100000.25"))
        self.assertEqual([delay for delay in self.sleeps if delay > 0], [2, 1])
        self.assertEqual(self.session.get.call_count, 3)

    def test_retry_after_http_date_and_long_delay_budget(self) -> None:
        """HTTP-date is honored, and a long cooldown is not shortened."""
        self.session.get.side_effect = [
            response({}, 429, {"Retry-After": "Sun, 01 Nov 2026 12:00:15 GMT"}),
            response({"price": "100", "time": self.now.isoformat()}),
        ]
        self.client.fetch_ticker()
        self.assertIn(5.0, self.sleeps)
        self.session.get.reset_mock()
        self.session.get.side_effect = [response({}, 429, {"Retry-After": "600"})]
        with self.assertRaises(MarketDataError):
            self.client.fetch_ticker()
        self.assertEqual(self.session.get.call_count, 1)

    def test_kraken_http_200_rate_limit_is_retried(self) -> None:
        """Kraken can report rate limits in a successful HTTP envelope."""
        self.session.get.side_effect = [
            response({"error": ["EAPI:Rate limit exceeded"]}),
            response({"price": "100", "time": self.now.isoformat()}),
        ]
        self.assertEqual(self.client.fetch_ticker().price, 100)
        self.assertIn(0.5, self.sleeps)

    def test_ticker_stale_and_invalid_values(self) -> None:
        """Never return stale, future, nonfinite or nonpositive spot ticks."""
        for price, timestamp in (
            ("NaN", self.now), ("-1", self.now),
            ("100", self.now - pd.Timedelta(seconds=11)),
            ("100", self.now + pd.Timedelta(seconds=3)),
        ):
            self.session.get.return_value = response({"price": price, "time": timestamp.isoformat()})
            with self.subTest(price=price, timestamp=timestamp), self.assertRaises(MarketDataError):
                self.client.fetch_ticker()

    def test_permanent_error_is_not_retried(self) -> None:
        self.session.get.return_value = response({}, 404)
        with self.assertRaises(MarketDataError):
            self.client.fetch_ticker()
        self.assertEqual(self.session.get.call_count, 1)

    def test_invalid_range_does_not_contact_exchange(self) -> None:
        with self.assertRaises(ValueError):
            self.client.fetch_candles(self.end, self.end + pd.Timedelta(minutes=1))
        self.session.get.assert_not_called()

    def test_retry_configuration_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(max_retries=11)


if __name__ == "__main__":
    unittest.main()
