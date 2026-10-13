"""Independent boundary expectations for the chapter 25/26 calculations."""
import unittest
from ch25_per import replay_weights
from ch26_quantile_cvar import cvar_size


class TailReplayBoundaryTests(unittest.TestCase):
    def test_equal_large_errors_have_uniform_probabilities(self):
        probabilities, weights = replay_weights([1e308, 1e308], alpha=1, beta=1)
        self.assertEqual(probabilities, [.5, .5])
        self.assertEqual(weights, [1., 1.])

    def test_large_error_range_keeps_weights_bounded(self):
        probabilities, weights = replay_weights([0., 1e308], alpha=1, beta=1)
        self.assertGreater(probabilities[0], 0)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertEqual(weights[0], 1)
        self.assertGreater(weights[1], 0)
        self.assertLess(weights[1], 1)

    def test_finite_tail_mean_does_not_overflow(self):
        tail, size = cvar_size([-1e308, -1e308], alpha=1, loss_budget=1e307)
        self.assertEqual(tail, -1e308)
        self.assertAlmostEqual(size, .1)

    def test_subnormal_tail_mass_still_selects_lowest_atom(self):
        tail, size = cvar_size([-.4, .2], alpha=5e-324)
        self.assertEqual(tail, -.4)
        self.assertAlmostEqual(size, .4375)

    def test_fractional_boundary_is_weighted_exactly(self):
        tail, size = cvar_size([-4., -2., 10.], alpha=.5, loss_budget=1)
        self.assertAlmostEqual(tail, -10/3)
        self.assertAlmostEqual(size, .3)


if __name__ == '__main__':
    unittest.main()
