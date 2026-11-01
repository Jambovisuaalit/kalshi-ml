"""Offline causal coverage and official-outcome join tests."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from kalshi_ml.data.kalshi_observer import ObservationJournal, _book_quote
from kalshi_ml.data.quality import audit_kalshi, audit_obi, main


TICKER = "KXBTC15M-TEST"
BASE = "https://external-api.kalshi.com/trade-api/v2/markets/" + TICKER


def event(kind: str, received: str, raw: dict) -> dict:
    return {
        "schema": "kalshi-public-observation-v1", "kind": kind,
        "environment": "PROD", "ticker": TICKER,
        "request_started_at": received, "received_at": received,
        "source_url": BASE + ("/orderbook" if kind == "orderbook" else ""),
        "raw_response": raw,
    }


class DataQualityTests(unittest.TestCase):
    def test_obi_minute_coverage_requires_fresh_received_exchange_time(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "obi.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["timestamp", "received_at", "bid_volume", "ask_volume", "sequence", "depth"])
                for minute, sequence in ((1, 1), (2, 2), (4, 3)):
                    stamp = f"2026-11-01T12:0{minute}:00Z"
                    writer.writerow([stamp, stamp, 1, 1, sequence, 10])
            report = audit_obi(path, as_of=pd.Timestamp("2026-11-01T12:04:00Z"), minutes=5)
            self.assertEqual((report["fresh_minutes"], report["missing_minutes"], report["stale_minutes"]), (3, 1, 1))
            self.assertEqual(report["coverage"], 0.6)
            self.assertEqual(report["longest_unusable_run_minutes"], 1)
            self.assertEqual(report["max_snapshot_gap_sec"], 120)
            earlier = audit_obi(path, as_of=pd.Timestamp("2026-11-01T12:01:00Z"), minutes=2)
            self.assertEqual(earlier["recorded_snapshots"], 1)
            self.assertEqual(earlier["fresh_minutes"], 1)

    def test_obi_torn_final_row_is_ignored_but_complete_bad_row_fails(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "obi.csv"
            valid = ("timestamp,received_at,bid_volume,ask_volume,sequence,depth\n"
                     "2026-11-01T12:00:00Z,2026-11-01T12:00:00Z,1,1,1,10\n")
            path.write_text(valid + "2026-11-01T12:01")
            report = audit_obi(path, as_of=pd.Timestamp("2026-11-01T12:00:00Z"), minutes=1)
            self.assertTrue(report["partial_tail_ignored"])
            self.assertEqual(report["fresh_minutes"], 1)
            path.write_text(valid + "incomplete,row\n")
            with self.assertRaises((ValueError, TypeError)):
                audit_obi(path, as_of=pd.Timestamp("2026-11-01T12:00:00Z"), minutes=1)

    def test_kalshi_join_excludes_post_settlement_and_preserves_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            journal = ObservationJournal(Path(directory))
            book = {"orderbook_fp": {"yes_dollars": [["0.48", "2"]], "no_dollars": [["0.50", "3"]]}}
            before = event("orderbook", "2026-11-01T12:10:00Z", book)
            before["top_of_book"] = _book_quote(book)
            after = event("orderbook", "2026-11-01T12:16:00Z", book)
            after["top_of_book"] = _book_quote(book)
            settled_raw = {"market": {"ticker": TICKER, "status": "settled", "result": "yes",
                                      "settlement_value_dollars": "1.0000", "settlement_ts": "2026-11-01T12:15:00Z"}}
            settled = event("settlement", "2026-11-01T12:16:30Z", settled_raw)
            settled.update(yes_payout_dollars="1.0000", result="yes",
                           official_settlement_ts="2026-11-01T12:15:00Z")
            for item in (before, after, settled):
                journal.append(item)
            early, early_rows = audit_kalshi(journal.path, as_of=pd.Timestamp("2026-11-01T12:16:00Z"))
            self.assertEqual((early["official_settlements"], len(early_rows)), (0, 0))
            report, rows = audit_kalshi(journal.path, as_of=pd.Timestamp("2026-11-01T12:17:00Z"))
            self.assertEqual(report["joined_pre_settlement_quotes"], 1)
            self.assertEqual(report["quotes_after_settlement_excluded"], 1)
            self.assertEqual(report["joined_contracts"], 1)
            self.assertEqual(rows[0]["quote_journal_line"], 1)
            self.assertEqual(rows[0]["yes_ask_dollars"], "0.50")
            self.assertEqual(rows[0]["yes_payout_dollars"], "1.0000")

    def test_rejects_modified_raw_quote_and_conflicting_settlement(self) -> None:
        with TemporaryDirectory() as directory:
            journal = ObservationJournal(Path(directory))
            book = {"orderbook_fp": {"yes_dollars": [["0.48", "2"]], "no_dollars": [["0.50", "3"]]}}
            item = event("orderbook", "2026-11-01T12:10:00Z", book)
            item["top_of_book"] = {**_book_quote(book), "yes_ask_dollars": "0.49"}
            journal.append(item)
            with self.assertRaisesRegex(ValueError, "differs from raw"):
                audit_kalshi(journal.path, as_of=pd.Timestamp("2026-11-01T12:20:00Z"))

        with TemporaryDirectory() as directory:
            journal = ObservationJournal(Path(directory))
            for result, payout, received in (("yes", "1.0000", "2026-11-01T12:16:00Z"),
                                             ("no", "0.0000", "2026-11-01T12:17:00Z")):
                raw = {"market": {"ticker": TICKER, "status": "settled", "result": result,
                                  "settlement_value_dollars": payout, "settlement_ts": "2026-11-01T12:15:00Z"}}
                item = event("settlement", received, raw)
                item.update(yes_payout_dollars=payout, result=result,
                            official_settlement_ts="2026-11-01T12:15:00Z")
                journal.append(item)
            with self.assertRaisesRegex(ValueError, "contradictory"):
                audit_kalshi(journal.path, as_of=pd.Timestamp("2026-11-01T12:20:00Z"))

    def test_torn_final_line_is_reported_and_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            book = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
            item = event("orderbook", "2026-11-01T12:10:00Z", book)
            item["top_of_book"] = _book_quote(book)
            path.write_text(json.dumps(item) + "\n{" , encoding="utf-8")
            report, rows = audit_kalshi(path, as_of=pd.Timestamp("2026-11-01T12:20:00Z"))
            self.assertTrue(report["partial_tail_ignored"])
            self.assertEqual(report["orderbook_observations"], 1)
            self.assertFalse(rows)

    def test_cli_writes_local_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            obi = root / "obi.csv"
            obi.write_text("timestamp,received_at,bid_volume,ask_volume,sequence,depth\n"
                           "2026-11-01T12:00:00Z,2026-11-01T12:00:00Z,1,1,1,10\n")
            journal = root / "kalshi.jsonl"
            journal.write_text("")
            report, joined = root / "report.json", root / "joined.csv"
            self.assertEqual(main(["--obi-csv", str(obi), "--kalshi-jsonl", str(journal),
                                   "--as-of", "2026-11-01T12:00:00Z", "--minutes", "1",
                                   "--report", str(report), "--joined-csv", str(joined)]), 0)
            self.assertEqual(json.loads(report.read_text())["obi"]["coverage"], 1.0)
            self.assertEqual(json.loads(report.read_text())["kalshi"]["complete_joined_contracts_in_obi_window"], 0)
            self.assertFalse(json.loads(report.read_text())["collection_ready"])
            self.assertEqual(len(joined.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
