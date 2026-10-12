"""python -m unittest discover -s tests -v"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm_demo import dummy_bars, features, fit_model, forward
from tiny_env import TinyTradingEnv
from ch21_regime_zscore import decide
from monitor_example import monitor


class ExamplesTest(unittest.TestCase):
    def test_chapter_coverage(self):
        rows = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual({r['chapter'] for r in rows}, {f'{i:02}' for i in range(29)})
        for row in rows:
            self.assertTrue((ROOT / row['file']).is_file())

    def test_features_do_not_see_future(self):
        bars = dummy_bars()
        changed = bars.copy()
        changed.iloc[250:, 0] *= 4
        assert_allclose(features(bars).iloc[:230], features(changed).iloc[:230])
        assert_allclose(bars, dummy_bars())

    def test_hmm_forward_matches_prefix_inference(self):
        x = features(dummy_bars()).to_numpy()
        model, scaler, order = fit_model(x, 200)
        x = scaler.transform(x)
        probability = forward(model, x)
        assert_allclose(probability.sum(axis=1), 1)
        for end in [201, 230, len(x)]:
            assert_allclose(probability[end-1], model.predict_proba(x[:end])[-1], atol=1e-9)
        changed = x.copy()
        changed[250:] += 100
        assert_allclose(probability[:250], forward(model, changed)[:250])

    def test_env_observation_cost_and_terminal(self):
        env = TinyTradingEnv([0.] * 12)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (11,))
        _, first, done, _, _ = env.step([1.])
        self.assertFalse(done)
        _, last, done, _, _ = env.step([-1.])
        self.assertTrue(done)
        self.assertAlmostEqual(first + last, -0.004)  # Entry 1 + reversal 2 + liquidation 1
        with self.assertRaises(RuntimeError):
            env.step([0.])
        with self.assertRaises(ValueError):
            TinyTradingEnv([float('nan')] * 12)

    def test_env_no_future_observation(self):
        a = TinyTradingEnv([0.] * 12)
        b = TinyTradingEnv([0.] * 10 + [0.1, -0.1])
        assert_allclose(a.reset()[0], b.reset()[0])

    def test_regime_gates_and_exits(self):
        self.assertEqual(decide(1.8, [1, 0, 0]), (-1., 'enter'))
        self.assertEqual(decide(100, [0, 1, 0])[0], 0)
        self.assertEqual(decide(100, [0, 0, 1])[0], 0)
        self.assertEqual(decide(-3.2, [1, 0, 0], 1), (0., 'stop'))
        self.assertEqual(decide(-0.2, [1, 0, 0], 1), (0., 'exit'))
        with self.assertRaises(ValueError):
            decide(1, [0.5, 0.5, 0.5])

    def test_monitor_and_http(self):
        from fastapi.testclient import TestClient
        from ch18_lab import app
        self.assertAlmostEqual(monitor([-200, 200])['metrics']['mdd'], 0.2)
        with self.assertRaises(ValueError):
            monitor([-1001])
        with TestClient(app) as client:
            self.assertEqual(client.get('/health').json(), {'status': 'ok'})
            self.assertEqual(client.get('/report').json()['violations'], ['mdd'])

    def test_nonpositive_halflife_rejected(self):
        from ch12_practice import check_entry
        for hl in [0, -1]:
            self.assertFalse(check_entry(-2.8, -3.1, hl, 0.65, 0.95)[0])

    def test_invalid_entry_thresholds_rejected(self):
        from ch12_practice import check_entry
        for threshold in [0, -1, float('nan'), float('inf')]:
            self.assertFalse(check_entry(-2.8, -3.1, 25, 0.65, 0.95, threshold)[0])
        self.assertFalse(check_entry(-2.8, -3.1, 25, 1.5, 0.95)[0])

    def test_accounting_next_open_and_final_costs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from ch03_practice import simulate
        bars = [{'open': 100., 'close': 100.},
                {'open': 110., 'close': 120.},
                {'open': 120., 'close': 130.}]
        result = simulate(bars, [1, 1, 0], fee_bps=10)
        self.assertEqual(result['fills'][0]['bar'], 1)
        self.assertEqual(result['fills'][0]['price'], 110.)
        self.assertEqual(result['fills'][-1]['reason'], 'final_liquidation')
        self.assertAlmostEqual(result['final_equity'], 1197.6)
        self.assertAlmostEqual(sum(result['bar_pnl']), 197.6)

    def test_helpers_match_chapter_definitions(self):
        import ast
        for helper, chapter in [('tiny_env.py', 'ch13_env.py'),
                                ('monitor_example.py', 'ch18_monitor.py')]:
            def definitions(path):
                tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
                return [ast.dump(n) for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            self.assertEqual(definitions(helper), definitions(chapter))
        for path in ROOT.glob('ch*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            names = [n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            self.assertEqual(len(names), len(set(names)), path.name)

    def test_duplicate_fills_and_rejected_overfill(self):
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module('ch17_order_events')
        order = module.OrderLedger(10)
        order.on_fill('a', 4)
        order.on_fill('a', 4)
        self.assertEqual(order.filled, 4)
        with self.assertRaises(ValueError):
            order.on_fill('b', 7)
        self.assertEqual(order.filled, 4)
        order.on_fill('b', 6)
        order.on_cancelled()
        self.assertEqual(order.status, 'filled')


if __name__ == '__main__':
    unittest.main()
