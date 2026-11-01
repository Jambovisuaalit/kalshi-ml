"""Pure offline validation tests: no API access, credentials or trading."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"validate_kalshi_market.py"
spec=importlib.util.spec_from_file_location("validate_kalshi_market",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def sample():
    return pd.DataFrame([
      dict(ticker="KXBTC15M-A",observed_at="2025-12-10T12:00:00Z",settled_at="2025-12-10T12:15:00Z",
           yes_bid_cents=50,yes_ask_cents=52,no_bid_cents=48,no_ask_cents=50,
           model_p_yes=.70,settlement_yes=1),
      dict(ticker="KXBTC15M-A",observed_at="2025-12-10T12:01:00Z",settled_at="2025-12-10T12:15:00Z",
           yes_bid_cents=52,yes_ask_cents=54,no_bid_cents=46,no_ask_cents=48,
           model_p_yes=.80,settlement_yes=1),
      dict(ticker="KXBTC15M-B",observed_at="2025-12-10T12:15:00Z",settled_at="2025-12-10T12:30:00Z",
           yes_bid_cents=45,yes_ask_cents=47,no_bid_cents=53,no_ask_cents=55,
           model_p_yes=.65,settlement_yes=0),
    ])

class MarketValidationTests(unittest.TestCase):
    def validate(self,df):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"snapshots.csv"
            df.to_csv(path,index=False)
            return mod.load(path)
    def test_valid_one_decision_per_contract(self):
        data=self.validate(sample())
        report,bins,decisions=mod.evaluate(data,2,1,3,5)
        self.assertEqual(report["distinct_contracts"],2)
        self.assertEqual(report["quote_snapshots"],3)
        self.assertEqual(len(decisions),2)
        self.assertEqual(int(bins.contracts.sum()),2)
        self.assertEqual(decisions.iloc[0].observed_at,pd.Timestamp("2025-12-10T12:00Z"))
    def test_reject_future_observation(self):
        data=sample()
        data.loc[0,"observed_at"]="2025-12-10T12:15:00Z"
        with self.assertRaisesRegex(ValueError,"look-ahead"):self.validate(data)
    def test_reject_conflicting_outcome(self):
        data=sample();data.loc[1,"settlement_yes"]=0
        with self.assertRaisesRegex(ValueError,"contradictory"):self.validate(data)
    def test_reject_complementary_cross(self):
        data=sample();data.loc[0,"no_bid_cents"]=55
        with self.assertRaisesRegex(ValueError,"inconsistent"):self.validate(data)
    def test_reject_duplicate_snapshot(self):
        data=sample();data.loc[1,"observed_at"]=data.loc[0,"observed_at"]
        with self.assertRaisesRegex(ValueError,"duplicate"):self.validate(data)
    def test_reject_missing_probability(self):
        data=sample();data.loc[0,"model_p_yes"]=float("nan")
        with self.assertRaisesRegex(ValueError,"missing"):self.validate(data)
    def test_reject_negative_fee(self):
        data=self.validate(sample())
        with self.assertRaisesRegex(ValueError,"nonnegative"):mod.evaluate(data,-1,0,3,5)

if __name__=="__main__":unittest.main()
