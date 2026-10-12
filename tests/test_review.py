"""Regression checks for the numerical and event-handling review."""
import contextlib
import importlib
import io
import unittest


def load(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module(name)


class ReviewTest(unittest.TestCase):
    def test_cvar_fractional_tail_and_endpoints(self):
        cvar_size = load('ch26_quantile_cvar').cvar_size
        # 37.5% of four atoms = one full atom and half of the second.
        cvar, size = cvar_size([-4, -2, 0, 2], alpha=0.375, loss_budget=1)
        self.assertAlmostEqual(cvar, -10 / 3)
        self.assertAlmostEqual(size, 0.3)
        self.assertEqual(cvar_size([-4, -2, 0, 2], alpha=0.01)[0], -4)
        self.assertEqual(cvar_size([-4, -2, 0, 2], alpha=1)[0], -1)
        self.assertEqual(cvar_size([1, 2], alpha=0.5)[1], 1)

    def test_empirical_cvar_exact_mass_with_ties(self):
        # Three observations tie at the 5% boundary. Include only one of them.
        returns = [-0.04, -0.02, -0.02, -0.02] + [0.01] * 36
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        for name in ['ch03_metrics', 'ch16_metrics']:
            metrics = load(name).compute_risk_metrics(equity)
            self.assertAlmostEqual(metrics['var95_loss'], 0.02)
            self.assertAlmostEqual(metrics['cvar95_loss'], 0.03)
            with self.assertRaises(ValueError):
                load(name).compute_risk_metrics(equity, periods=float('nan'))

    def test_orders_reject_bad_quantities_without_mutating(self):
        Ledger = load('ch17_order_events').OrderLedger
        for quantity in [0, -1, 1.5, float('nan'), float('inf')]:
            with self.subTest(quantity=quantity):
                with self.assertRaises(ValueError):
                    Ledger(quantity)
                order = Ledger(10)
                with self.assertRaises(ValueError):
                    order.on_fill('bad', quantity)
                self.assertEqual(order.filled, 0)
                self.assertEqual(order.executions, set())

    def test_cancel_is_idempotent_and_late_fill_is_accounted(self):
        order = load('ch17_order_events').OrderLedger(10)
        order.request_cancel()
        order.on_cancelled()
        order.request_cancel()
        self.assertEqual(order.status, 'cancelled')
        order.on_fill('late', 4)
        self.assertEqual((order.filled, order.status), (4, 'cancelled'))
        order.on_fill('last', 6)
        order.on_cancelled()
        self.assertEqual(order.status, 'filled')

    def test_invalid_timestamp_preserves_watermark(self):
        validate = load('ch24_validation').validate
        for timestamp in [float('nan'), float('inf'), -float('inf')]:
            self.assertEqual(validate(timestamp, [1, 0, 0], 5), ('invalid_timestamp', 5))
        self.assertEqual(validate(6, [1, 0, 0], 5), ('ok', 6))

    def test_no_entry_at_or_beyond_stop_on_either_side(self):
        decide = load('ch21_regime_zscore').decide
        for z in [-100, -3, 3, 100]:
            self.assertEqual(decide(z, [1, 0, 0]), (0., 'entry beyond stop'))
        self.assertEqual(decide(3, [1, 0, 0], -1), (0., 'stop'))
        self.assertEqual(decide(-3, [1, 0, 0], 1), (0., 'stop'))
        self.assertEqual(decide(1.5, [1, 0, 0]), (-1., 'enter'))
        self.assertEqual(decide(-1.5, [1, 0, 0]), (1., 'enter'))

    def test_filter_rejects_nonfinite_inputs(self):
        filter_prices = load('ch06_kalman').filter_prices
        for bad in [float('nan'), float('inf')]:
            for kwargs in [{'prices': [100, bad]}, {'prices': [100], 'q': bad},
                           {'prices': [100], 'r': bad}]:
                with self.assertRaises(ValueError):
                    filter_prices(**kwargs)

    def test_ensemble_rejects_invalid_weights_and_fraction(self):
        combine = load('ch08_ensemble').combine_two_layers
        for bad in [float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                combine([1.], [{'regimes': [0], 'p_up': 0.7, 'weight': bad}])
            with self.assertRaises(ValueError):
                combine([1.], [], half_kelly=bad)

    def test_dsr_units_and_finite_inputs(self):
        dsr = load('ch03_dsr').dsr_probability
        self.assertEqual(dsr(0, 252, 1, 0.05), 0.5)
        self.assertLess(dsr(0.1, 252, 100, 0.05), dsr(0.1, 252, 1, 0.05))
        for kwargs in [{'sr': float('nan')}, {'n': 3.5}, {'trials': 1.5},
                       {'trial_sr_std': float('inf')}, {'excess_kurtosis': -3}]:
            args = dict(sr=0.1, n=252, trials=10, trial_sr_std=0.05)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                dsr(**args)

    def test_per_uniform_and_full_importance_correction(self):
        replay = load('ch25_per').replay_weights
        probabilities, weights = replay([0, 1, 2], alpha=0)
        self.assertEqual(probabilities, [1/3] * 3)
        self.assertEqual(weights, [1.] * 3)
        probabilities, weights = replay([0.1, 1, 2], beta=1)
        corrected = [p * w for p, w in zip(probabilities, weights)]
        self.assertAlmostEqual(corrected[0], corrected[2])
        for kwargs in [{'alpha': float('nan')}, {'beta': -0.1}, {'beta': 1.1}]:
            with self.assertRaises(ValueError):
                replay([1, 2], **kwargs)

    def test_monitor_limit_validation(self):
        monitor = load('monitor_example').monitor
        for limit in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                monitor([1, 2], mdd_limit=limit)


if __name__ == '__main__':
    unittest.main()
