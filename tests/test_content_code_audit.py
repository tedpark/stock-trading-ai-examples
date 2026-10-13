"""Boundary regressions for formulas and design fragments printed in the book."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import logging
import sys
import threading
import time
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ch11_practice import estimate_halflife, screen_pair
from ch20_screening import _half_life
from hmm_demo import forward


def snippet(path):
    scope = dict(np=np, time=time, logger=logging.getLogger('audit'))
    from sync_book import contract, source_text
    key=path.removesuffix('.py.txt')
    entry=next(e for e in contract() if e['id']==key)
    exec(source_text(entry), scope)
    return scope


class ContentCodeAuditTests(unittest.TestCase):
    def test_exact_discrete_half_life_agrees_across_chapters(self):
        printed = snippet('11-stat-arb-pairs/python-03.py.txt')['calc_halflife']
        values = 4 + 6 * .5 ** np.arange(30)
        for fn in [estimate_halflife, _half_life, printed]:
            self.assertAlmostEqual(fn(values), 1.)
            for bad in [np.ones(30), [np.nan]*30, [1]*3, np.arange(30.), np.ones((30, 2))]:
                self.assertTrue(np.isinf(fn(bad)))

    def test_pair_screening_rejects_invalid_and_degenerate_inputs(self):
        for a,b in [([1]*19,[1]*19),([1]*20,[1]*21),([0]*20,[1]*20),([np.nan]*20,[1]*20)]:
            with self.assertRaises(ValueError): screen_pair(a,b)
        self.assertIsNone(screen_pair(np.ones(30),np.arange(1,31)))
        self.assertIsNone(screen_pair(np.arange(1,31),np.arange(1,31)))

    def model(self):
        return SimpleNamespace(means_=np.array([[0.],[100.]]),covars_=np.array([[[1.]],[[1.]]]),
                               transmat_=np.eye(2),startprob_=np.array([1.,0.]))

    def test_hmm_impossible_states_remain_impossible(self):
        np.testing.assert_array_equal(forward(self.model(),[[100.],[100.]]),[[1.,0.],[1.,0.]])

    def test_hmm_invalid_inputs_are_rejected(self):
        for x in [[],[1,2],[[np.nan]]]:
            with self.assertRaises(ValueError): forward(self.model(),x)
        m=self.model();m.transmat_[0,0]=.5
        with self.assertRaises(ValueError):forward(m,[[0.]])

    def test_trailing_stop_ratchets_when_tier_widens(self):
        scope=snippet('16-risk-management/python-03.py.txt')
        tier=scope['_get_tiered_trailing_stop']; ratchet=scope['ratchet_long_stop']
        stop=None
        for peak in [102.99,103.,109.99,110.,120.]:
            new=ratchet(stop,peak,tier(None,peak/100-1))
            if stop is not None:self.assertGreaterEqual(new,stop)
            stop=new
        for args in [(None,0,-.01),(None,100,0),(float('nan'),100,-.01)]:
            with self.assertRaises(ValueError):ratchet(*args)

    def test_ibkr_account_initialization_failure_is_not_success(self):
        import asyncio
        from quantlab.broker import BrokerSession, PaperBrokerBackend
        class Broken(PaperBrokerBackend):
            async def accounts(self):raise RuntimeError('unavailable')
        async def check():
            obj=BrokerSession(Broken())
            self.assertFalse(await obj.connect())
            self.assertFalse(obj.connected)
            self.assertFalse(await obj.connect())
        asyncio.run(check())

if __name__=='__main__':unittest.main()
