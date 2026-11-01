"""Regression tests for Kalshi historical audit price units and interval joins."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "join_kalshi_real_data.py"

class KalshiJoinTests(unittest.TestCase):
    def test_dollar_price_conversion_and_candle_end_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            market = {
                "ticker":"KXBTC15M-TEST", "result":"yes",
                "open_time":"2025-12-30T12:00:00Z",
                "close_time":"2025-12-30T12:15:00Z"}
            candle = {"end_period_ts":1767096060,
                      "yes_bid":{"close":"0.4100"},
                      "yes_ask":{"close":"0.4500"}}
            trade = {"ticker":"KXBTC15M-TEST",
                     "trade_id":"t1","created_time":"2025-12-30T12:00:30Z",
                     "count_fp":"3.00"}
            for filename, data in [
                ("settled_markets.jsonl",market),
                ("raw_candles.jsonl",{"payload":{"ticker":"KXBTC15M-TEST","candlesticks":[candle]}}),
                ("raw_trades.jsonl",{"payload":{"trades":[trade]}}),
            ]:
                (base/filename).write_text(json.dumps(data)+chr(10),encoding="utf-8")
            result=subprocess.run([sys.executable,str(SCRIPT),"--input",str(base)],
                                  capture_output=True,text=True,check=True)
            audit=json.loads(result.stdout)
            self.assertEqual(audit["markets"],1)
            self.assertEqual(audit["candle_rows"],1)
            self.assertEqual(audit["trade_observations_joined_by_candle_interval"],1)
            self.assertEqual(audit["yes_spread_p50_cents"],4)
            import pandas as pd
            df=pd.read_csv(base/"kalshi_joined_audit.csv")
            self.assertEqual(df.iloc[0].yes_bid_close_cents,41)
            self.assertEqual(df.iloc[0].yes_ask_close_cents,45)
            self.assertEqual(df.iloc[0].trade_count_in_interval,1)
            self.assertTrue(pd.isna(df.iloc[0].no_bid_close_cents))
            self.assertTrue(pd.isna(df.iloc[0].model_p_yes))

if __name__=="__main__":
    unittest.main()
