"""Focused provenance, parity, settlement and restart tests for public collection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kalshi_ml.config import KalshiEnvironment
from kalshi_ml.data.kalshi_observer import KalshiObserver, ObservationJournal, _book_quote, _official_result
from kalshi_ml.execution.kalshi_client import InvalidResponseError


class FakeClient:
    environment = KalshiEnvironment.PROD

    def __init__(self) -> None:
        self.active = SimpleNamespace(ticker="KXBTC15M-TEST", exchange_index=2)
        self.status = "open"
        self.result: str | None = None
        self.market_reads = 0

    def get_active_15m_market(self, series_ticker: str) -> object:
        assert series_ticker == "KXBTC15M"
        return self.active

    def get_public_market_response(self, ticker: str) -> dict:
        self.market_reads += 1
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
            self.assertEqual(client.market_reads, 1)
            self.assertEqual(journal.pending_tickers(KalshiEnvironment.PROD, "KXBTC15M"), {"KXBTC15M-TEST"})
            events = [json.loads(line) for line in journal.path.read_text().splitlines()]
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
            self.assertEqual(journal.pending_tickers(KalshiEnvironment.PROD, "KXBTC15M"), set())
            events = [json.loads(line) for line in journal.path.read_text().splitlines()]
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
            self.assertEqual(journal.pending_tickers(KalshiEnvironment.PROD, "KXBTC15M"), {"KXBTC15M-TEST"})
            self.assertEqual(len(journal.path.read_text().splitlines()), 1)

    def test_no_fabricated_settlement(self) -> None:
        self.assertIsNone(_official_result({"status": "closed", "result": "yes"}))
        self.assertIsNone(_official_result({"status": "settled"}))
        with self.assertRaises(InvalidResponseError):
            _official_result({"status": "settled", "result": "yes", "settlement_value_dollars": "0"})

    def test_bad_official_timestamp_stops_without_settlement_record(self) -> None:
        with TemporaryDirectory() as directory:
            client = FakeClient()
            client.status, client.result, client.active = "settled", "yes", None
            journal = ObservationJournal(Path(directory))
            journal.append({"schema": "kalshi-public-observation-v1", "environment": "PROD",
                            "ticker": "KXBTC15M-TEST", "kind": "market"})
            original = client.get_public_market_response
            def response(ticker: str) -> dict:
                data = original(ticker)
                data["market"]["settlement_ts"] = "2026-11-01T12:15:00"
                return data
            client.get_public_market_response = response  # type: ignore[method-assign]
            observer = KalshiObserver(client, journal)  # type: ignore[arg-type]
            with self.assertRaisesRegex(InvalidResponseError, "lacks timezone"):
                observer.observe_once()
            self.assertEqual(journal.pending_tickers(KalshiEnvironment.PROD, "KXBTC15M"), {"KXBTC15M-TEST"})

    def test_empty_and_legacy_book(self) -> None:
        self.assertIsNone(_book_quote({"orderbook": {"yes": [], "no": []}})["yes_ask_dollars"])
        self.assertEqual(_book_quote({"orderbook": {"yes": [[48, 2]], "no": [[50, 1]]}})["yes_bid_dollars"], "0.48")

    def test_journal_rejects_second_writer(self) -> None:
        with TemporaryDirectory() as directory:
            journal = ObservationJournal(Path(directory))
            with journal.exclusive():
                with self.assertRaisesRegex(RuntimeError, "another observer"):
                    with journal.exclusive():
                        pass

    def test_journal_rejects_mixed_environments(self) -> None:
        with TemporaryDirectory() as directory:
            journal = ObservationJournal(Path(directory))
            journal.append({"schema": "kalshi-public-observation-v1", "environment": "PROD",
                            "ticker": "KXBTC15M-TEST", "kind": "market"})
            with self.assertRaisesRegex(ValueError, "environment mismatch"):
                journal.pending_tickers(KalshiEnvironment.DEMO, "KXBTC15M")

    def test_public_raw_readers_never_authenticate_or_mutate(self) -> None:
        from kalshi_ml.execution.kalshi_client import KalshiClient
        client = KalshiClient(environment=KalshiEnvironment.PROD)
        with patch.object(client, "_request", return_value={"market": {"ticker": "TEST"}}) as request:
            client.get_public_market_response("TEST")
            self.assertEqual(request.call_args.args, ("GET", "/trade-api/v2/markets/TEST"))
            self.assertFalse(request.call_args.kwargs["authenticated"])


if __name__ == "__main__":
    unittest.main()
