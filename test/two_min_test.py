import unittest
from unittest import TestCase, main

import numpy as np

from bbdob import TwoMin
from bbdob.utils import idx2one_hot


class TestFourPeaks(TestCase):
    def test_optimal(self):
        dim = 6
        objective = TwoMin(dim, minimize=True)
        x = idx2one_hot(objective.y, objective.Cmax)
        self.assertEqual(objective.is_optimum(x), True)
        x = idx2one_hot(np.logical_not(objective.y).astype(np.int), objective.Cmax)
        self.assertEqual(objective.is_optimum(x), True)

    def test_evaluate(self):
        np.random.seed(1)
        dim = 6
        objective = TwoMin(dim, minimize=True)
        # target = [0, 1, 0, 0, 0, 0]

        sample = idx2one_hot(np.array([0, 1, 0, 0, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - objective.optimal_value) < 1e-8), True)

        sample = idx2one_hot(np.array([1, 1, 1, 1, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - 3) < 1e-8), True)

        sample = idx2one_hot(np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - np.array([1, 2, 2])) < 1e-8), True)


if __name__ == "__main__":
    unittest.main()
