"""Numeric boundary tests with independent expected results."""
import unittest
from math import isfinite
from ch05_forward import forward_step
from ch08_ensemble import combine_two_layers
from monitor_example import monitor

class NumericalBoundaryTests(unittest.TestCase):
    def test_forward_rejects_invalid_probability_contract(self):
        args=([.5,.5],0.,[[.9,.1],[.1,.9]],[0.,2.],[1.,1.])
        for index,value in [(0,[-.1,1.1]),(0,[.2,.2]),(1,float('nan')),
                            (2,[[1.,0.]]),(2,[[1.,1.],[0.,1.]]),
                            (3,[0.,float('inf')]),(4,[0.,1.]),(4,[-1.,1.])]:
            changed=list(args);changed[index]=value
            with self.subTest(index=index,value=value):
                with self.assertRaises(ValueError):forward_step(*changed)

    def test_forward_extreme_observation_and_structural_zero(self):
        # Equal emissions: posterior equals the predicted prior even far into the tail.
        result=forward_step([.25,.75],1000.,[[1.,0.],[0.,1.]],[0.,0.],[1.,1.])
        self.assertAlmostEqual(result[0],.25);self.assertAlmostEqual(result[1],.75)
        self.assertEqual(forward_step([1.,0.],100.,[[1.,0.],[0.,1.]],[0.,100.],[1.,1.]),[1.,0.])

    def test_ensemble_is_invariant_to_weight_scale(self):
        models=[{'regimes':[0],'p_up':.6,'weight':1e308},
                {'regimes':[0],'p_up':.8,'weight':1e308}]
        p,size=combine_two_layers([1.],models)
        self.assertAlmostEqual(p,.7);self.assertAlmostEqual(size,.2)
        self.assertTrue(isfinite(p))

    def test_monitor_rejects_overflow_before_reporting_metrics(self):
        for pnl,initial in [([1e308],1e308),([1e308],1e-308)]:
            with self.assertRaises(ValueError):monitor(pnl,starting_equity=initial)

if __name__=='__main__':unittest.main()
