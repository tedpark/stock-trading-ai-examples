"""Execute selected printed functions with controlled inputs, not their design environment."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verify_book_sources import verify
ROWS=json.loads((ROOT/'book_sources/index.json').read_text())

def extract(edition,chapter,needle,**scope):
    matches=[(ROOT/r['artifact']).read_text() for r in ROWS
             if r['edition']==edition and r['chapter'].startswith(chapter+'-') and r['language']=='python']
    code,=[c for c in matches if needle in c]
    namespace=dict(np=np,**scope)
    exec(code,namespace)
    return namespace

class PrintedSourceTests(unittest.TestCase):
    def test_complete_integrity_and_runnable_mapping(self):
        self.assertEqual(verify()['fenced_blocks'],711)

    def test_missing_non_python_source_is_rejected(self):
        for edition in ['ko','en','ja']:
            index=next(i for i,r in enumerate(ROWS)
                       if r['edition']==edition and r['language']!='python')
            with self.subTest(edition=edition), self.assertRaises(AssertionError):
                verify(ROWS[:index]+ROWS[index+1:])

    def test_validation_objective_costs_and_holdout(self):
        # +1 -> -1 -> flat: turnover 1+2+1, gross .1-.2=-.1.
        # A second case stays long until final liquidation: two one-unit fills.
        for lang in ['ko','en','ja']:
            for forecasts,returns,expected in [([1.,-1.,0.],[.1,.2,.3],-.104),([1.,1.],[.1,.2],.298)]:
                n=len(forecasts)
                scope=extract(lang,'15','thresh = std_e * scale',std_e=1.,
                    e_arr=np.array(forecasts+[999.]),ret_arr=np.array(returns+[-999.]),
                    val_end=n,fee_rate=.001,evaluate_objective=np.sum)
                self.assertAlmostEqual(scope['objective'](SimpleNamespace(suggest_float=lambda *a:.5)),expected)

    def test_kalman_missing_bars_and_no_future_effect(self):
        for lang in ['ko','en','ja']:
            fn=extract(lang,'06','def _kalman_innovations(')['_kalman_innovations']
            np.testing.assert_allclose(fn([0,np.nan,np.nan,10,10],1,1)[3:],[10,2])
            np.testing.assert_allclose(fn([0,1,2]),fn([0,1,2,999])[:3])

    def test_half_life_known_ar1(self):
        for lang in ['ko','en','ja']:
            fn=extract(lang,'11','def calc_halflife(')['calc_halflife']
            x=[10.]
            for _ in range(29):x.append(2+.5*x[-1])
            self.assertAlmostEqual(fn(x),1.)
            self.assertTrue(np.isinf(fn([1.]*30)))

    def test_atr_stop_bounds(self):
        for lang in ['ko','en','ja']:
            for atr,expected in [(1,-.02),(2,-.03),(10,-.05)]:
                fn=extract(lang,'16','def _calculate_atr_stop_loss(',compute_atr=lambda *a,**k:atr)['_calculate_atr_stop_loss']
                self.assertAlmostEqual(fn(None,'TEST',100),expected)

    def test_kelly_and_sortino_boundaries(self):
        for lang in ['ko','en','ja']:
            fn=extract(lang,'09','def fractional_kelly(')['fractional_kelly']
            self.assertAlmostEqual(fn(.01,.04),.0625)
            self.assertEqual(fn(.01,float('nan')),0)
            fn=extract(lang,'16','def compute_sortino(')['compute_sortino']
            self.assertIsNone(fn([0.,0.]))
            self.assertAlmostEqual(fn([-.1,.2],periods_per_year=1),2**-.5)

if __name__=='__main__':unittest.main()
