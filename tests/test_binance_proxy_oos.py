"""Time-leakage regressions for the Binance proxy OOS exporter."""
import unittest
import numpy as np
import pandas as pd
from scripts.export_binance_proxy_oos import samples

class ProxyOOSTests(unittest.TestCase):
    @staticmethod
    def market(seed=4):
        rng=np.random.default_rng(seed)
        times=pd.date_range("2025-12-30T00:00:00Z",periods=240,freq="min")
        price=87000+rng.normal(0,5,len(times)).cumsum()
        return pd.DataFrame({"open":price,"close":price+rng.normal(0,.5,len(times)),
                             "high":price+8,"low":price-8,"volume":5.0},index=times)

    def test_features_do_not_use_future_prices(self):
        original=self.market()
        x1,y1,eligible1,ties1,strike1=samples(original)
        changed=original.copy()
        # Future bars and the final settlement of early cycles must not change
        # predictions made before 01:15.
        changed.loc[changed.index>=pd.Timestamp("2025-12-30T01:15Z"),["open","close"]]+=10000
        x2,y2,eligible2,ties2,strike2=samples(changed)
        t=pd.Timestamp("2025-12-30T01:10:00Z")
        self.assertTrue(bool(eligible1[original.index.get_loc(t)]))
        np.testing.assert_allclose(x1.loc[t].to_numpy(float),x2.loc[t].to_numpy(float))
        self.assertEqual(bool(eligible1[original.index.get_loc(t)]),bool(eligible2[original.index.get_loc(t)]))

    def test_inference_eligibility_does_not_filter_tie(self):
        original=self.market()
        # Make one cycle's outcome a tie without touching early input bars.
        baseline,_,eligible,_,strike=samples(original)
        ix=original.index.get_loc(pd.Timestamp("2025-12-30T01:10Z"))
        altered=original.copy()
        cycle=original.index.get_loc(pd.Timestamp("2025-12-30T01:00Z"))
        altered.iloc[cycle+14,altered.columns.get_loc("close")]=float(original.open.iloc[cycle])
        after,_,eligible_after,no_tie,_=samples(altered)
        self.assertTrue(bool(eligible[ix]))
        self.assertTrue(bool(eligible_after[ix]))
        self.assertFalse(bool(no_tie[ix]))
        np.testing.assert_allclose(baseline.iloc[ix].to_numpy(float),after.iloc[ix].to_numpy(float))

if __name__=="__main__":unittest.main()
