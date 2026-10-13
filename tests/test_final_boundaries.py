"""Adversarial inputs that previously escaped happy-path chapter checks."""
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ch19_cointegration import fit_and_test_pair
from ch17_order_events import OrderLedger
from paper_service.engine import Risk, simulate, validate_bars, make_bars

class FinalBoundaryTests(unittest.TestCase):
    def test_pair_rejects_degenerate_or_invalid_series(self):
        b = pd.Series(np.arange(100., 140.))
        for a, other in [(2*b+3,b),(pd.Series([1.]*40),b),(b,pd.Series([1.]*40)),
                         (b.iloc[:10],b.iloc[:10]),(b.where(b.index!=2,np.inf),b),
                         (b.set_axis([0]*40),b),(b.iloc[::-1],b.iloc[::-1])]:
            with self.subTest(a=a.iloc[:2].tolist()):
                with self.assertRaises(ValueError):fit_and_test_pair(a,other)

    def test_pair_aligns_missing_dates_and_keeps_stationary_residual(self):
        rng=np.random.default_rng(7)
        b=pd.Series(100+np.cumsum(rng.normal(size=200)))
        a=2+1.3*b+rng.normal(0,.3,200);a.iloc[5]=np.nan
        result=fit_and_test_pair(a,b)
        self.assertEqual(len(result['spread']),199)
        self.assertTrue(np.isfinite(result['eg_stat']))
        self.assertTrue(result['is_candidate'])

    def test_duplicate_execution_conflict_preserves_accounting(self):
        ledger=OrderLedger(10);ledger.on_fill('a',4)
        ledger.on_fill('a',4)
        with self.assertRaises(ValueError):ledger.on_fill('a',5)
        self.assertEqual(ledger.filled,4)
        for key in ['',None]:
            with self.assertRaises(ValueError):ledger.on_fill(key,1)
        ledger.on_cancelled();ledger.on_fill('b',6)
        self.assertEqual((ledger.filled,ledger.status),(10,'filled'))

    def test_simulation_rejects_bad_shapes_prices_and_time(self):
        bars=make_bars(100).iloc[:3]
        cases=[(bars,np.ones((3,1))),(bars.reset_index(drop=True),[1]*3),
               (bars.iloc[::-1],[1]*3),(bars.assign(open=0),[1]*3),
               (bars.assign(close=np.nan),[1]*3)]
        for frame,targets in cases:
            with self.subTest(index=type(frame.index).__name__):
                with self.assertRaises(ValueError):simulate(frame,targets,Risk())

    def test_bars_require_datetime_and_unique_columns(self):
        bars=make_bars(100)
        with self.assertRaises(ValueError):validate_bars(bars.reset_index(drop=True))
        with self.assertRaises(ValueError):validate_bars(pd.concat([bars,bars[['close']]],axis=1))

    def test_randomized_ledger_reconciles_every_cashflow(self):
        rng=np.random.default_rng(812)
        for seed in range(12):
            bars=make_bars(100,seed).iloc[:25].copy()
            bars['open'] *= rng.uniform(.9,1.1,len(bars))
            risk=Risk(10000,.65,.15,23,17)
            result=simulate(bars,rng.uniform(0,1,len(bars)),risk)
            cash=10000.;quantity=0.;peak=10000.;fee_sum=0.
            for t,row in zip(result['timeline'],bars.itertuples()):
                for fill in result['fills']:
                    if fill['timestamp']==t['timestamp']:
                        cash-=fill['quantity']*fill['price']+fill['fee']
                        quantity+=fill['quantity'];fee_sum+=fill['fee']
                equity=cash+quantity*row.close;peak=max(peak,equity)
                self.assertAlmostEqual(t['cash'],cash)
                self.assertAlmostEqual(t['quantity'],quantity)
                self.assertAlmostEqual(t['equity'],equity)
                self.assertAlmostEqual(t['drawdown'],1-equity/peak)
                self.assertGreaterEqual(cash,-1e-8)
            self.assertAlmostEqual(quantity,0.)
            self.assertAlmostEqual(result['metrics']['fees'],fee_sum)
            self.assertAlmostEqual(result['metrics']['final_equity'],cash)

if __name__=='__main__':unittest.main()
