"""Independent edge cases for both implementations printed in the book."""
import sys
import unittest
import torch
from quantlab.risk import return_tail
from quantlab.distribution import QuantileAgent


class FixedAtoms(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float64).view(1, 1, -1)

    def forward(self, state):
        return self.values


class RuntimeTailBoundaries(unittest.TestCase):
    def test_risk_and_agent_agree_with_known_tail_means(self):
        cases = [
            ([-1e308, -1e308], 1, -1e308),
            ([-.4, .2], 5e-324, -.4),
            ([-sys.float_info.max] * 3, 1, -sys.float_info.max),
            ([-1e-308, -1e-308, 1e308], 2/3, -1e-308),
            ([-4., -2., 10.], .5, -10/3),
            ([0., 0., 0.], 1, 0.),
        ]
        for atoms, alpha, expected in cases:
            with self.subTest(atoms=atoms, alpha=alpha):
                actual = return_tail(atoms, alpha)[1]
                prediction = QuantileAgent(FixedAtoms(atoms)).cvar(None, 0, alpha).item()
                if expected:
                    self.assertAlmostEqual(actual / expected, 1.)
                    self.assertAlmostEqual(prediction / expected, 1.)
                else:
                    self.assertEqual(actual, 0.)
                    self.assertEqual(prediction, 0.)


if __name__ == '__main__':
    unittest.main()
