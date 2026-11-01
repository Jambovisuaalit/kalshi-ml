"""Focused provenance, parity, settlement and restart tests for public collection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from kalshi_ml.config import KalshiEnvironment
from kalshi_ml.data.kalshi_observer import KalshiObserver, ObservationJournal, _book_quote, _official_result
from kalshi_ml.execution.kalshi_client import InvalidResponseError


class FakeClient:
    environment = KalshiEnvironment.PROD

    def __init__(self) -> None:
        self.active = SimpleNamespace(ticker="KXBTC15M-TEST", exchange_index=2)
        self.status = "open"
        self.result: str | None = None

    def get_active_15m_market(self, series_ticker: str) -> object:
        assert series_ticker == "KXBTC15M"
        return self.active

    def get_public_market_response(self, ticker: str) -> dict:
        return {"market": {"ticker": ticker, "exchange_index": 2, "status": self.status,
                           "result": self.result, "settlement_ts": "2026-11-01T12:15:00Z"}}

    def get_public_orderbook_response(self, ticker: str) -> dict:
        return {"orderbook_fp": {"yes_dollars": [["0.48", "3.00"]],
                                 "no_dollars": [["0.50", "4.00"]]}}


class ObserverTests(unittest.TestCase):
    def test_quote_provenance_and_restart_settlement(self) -> None:
        with TemporaryDirectory() as directory:
            client = FakeClient()
            now = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
            calls = 0

            def clock() -> datetime:
                nonlocal calls
                calls += 1
                return now + timedelta(milliseconds=calls)

            journal = ObservationJournal(Path(directory))
            observer = KalshiObserver(client, journal, now=clock)  # type: ignore[arg-type]
            self.assertEqual(observer.observe_once(), (1, 0))
            self.assertEqual(journal.pending_tickers(), {"KXBTC15M-TEST"})
            events = [__import__("json").loads(line) for line in journal.path.read_text().splitlines()]
            self.assertEqual([event["kind"] for event in events], ["market", "orderbook"])
            self.assertEqual(events[1]["top_of_book"]["yes_ask_dollars"], "0.50")
            self.assertEqual(events[1]["top_of_book"]["no_ask_dollars"], "0.52")
            self.assertTrue(events[0]["source_url"].endswith("/markets/KXBTC15M-TEST"))
            self.assertLess(events[0]["request_started_at"], events[0]["received_at"])
            client.active = None
            client.status, client.result = "settled", "yes"
            observer = KalshiObserver(client, journal, now=clock)  # type: ignore[arg-type]
            self.assertEqual(observer.observe_once(), (0, 1))
            self.assertEqual(observer.observe_once(), (0, 0))
            self.assertEqual(journal.pending_tickers(), set())
            events = [__import__("json").loads(line) for line in journal.path.read_text().splitlines()]
            self.assertEqual(events[-1]["yes_payout_dollars"], "1")
            self.assertEqual(events[-1]["official_settlement_ts"], "2026-11-01T12:15:00Z")

    def test_invalid_book_is_not_recorded_as_quote(self) -> None:
        with TemporaryDirectory() as directory:
            client = FakeClient()
            client.get_public_orderbook_response = lambda _: {"orderbook_fp": {"yes_dollars": [["0.7", "1"]], "no_dollars": [["0.5", "1"]]}}  # type: ignore[method-assign]
            journal = ObservationJournal(Path(directory))
            observer = KalshiObserver(client, journal)  # type: ignore[arg-type]
            with self.assertRaises(InvalidResponseError):
                observer.observe_once()
            self.assertEqual(journal.pending_tickers(), {"KXBTC15M-TEST"})
            self.assertEqual(len(journal.path.read_text().splitlines()), 1)

    def test_no_fabricated_settlement(self) -> None:
        self.assertIsNone(_official_result({"status": "closed", "result": "yes"}))
        self.assertIsNone(_official_result({"status": "settled"}))
        with self.assertRaises(InvalidResponseError):
            _official_result({"status": "settled", "result": "yes", "settlement_value_dollars": "0"})

    def test_empty_and_legacy_book(self) -> None:
        self.assertIsNone(_book_quote({"orderbook": {"yes": [], "no": []}})["yes_ask_dollars"])
        self.assertEqual(_book_quote({"orderbook": {"yes": [[48, 2]], "no": [[50, 1]]}})["yes_bid_dollars"], "0.48")


if __name__ == "__main__":
    unittest.main()
