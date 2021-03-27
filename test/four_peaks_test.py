import unittest
from unittest import TestCase, main

import numpy as np

from bbdob import FourPeaks
from bbdob.utils import idx2one_hot


class TestFourPeaks(TestCase):
    def test_optimum(self):
        dim = 7
        t = 2
        objective = FourPeaks(dim, t, minimize=True)
        x = np.zeros(dim, dtype=np.int)
        x[:t+1] = 1
        x = idx2one_hot(x, objective.Cmax)
        self.assertEqual(objective.is_optimum(x), True)
        x = np.ones(dim, dtype=np.int)
        x[-t-1:] = 0
        x = idx2one_hot(x, objective.Cmax)
        self.assertEqual(objective.is_optimum(x), True)

    def test_sub_optimum(self):
        dim, t = 7, 2
        objective = FourPeaks(dim, t=t, minimize=True)
        sub_optimum = np.zeros(dim, dtype=np.int)
        evals, _ = objective(idx2one_hot(sub_optimum, 2))
        self.assertEqual(evals.item(), -dim)
        sub_optimum = np.ones(dim, dtype=np.int)
        evals, _ = objective(idx2one_hot(sub_optimum, 2))
        self.assertEqual(evals.item(), -dim)

    def test_evaluation(self):
        dim = 6
        th = 2
        objective = FourPeaks(dim, th, minimize=True)

        sample = idx2one_hot(np.array([1, 1, 1, 0, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - objective.optimal_value) < 1e-8), True)

        sample = idx2one_hot(np.array([1, 1, 1, 1, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + 4) < 1e-8), True)

        sample = idx2one_hot(np.array([[1, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0],
                                       [0, 0, 0, 1, 1, 1]]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([4, 1, 0])) < 1e-8), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            dim = 6
            th = 5
            self.assertRaises(FourPeaks(dim, th))


if __name__ == "__main__":
    unittest.main()
