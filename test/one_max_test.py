import unittest
from unittest import TestCase, main

import numpy as np

from bbdob import OneMax
from bbdob.utils import idx2one_hot


class TestOneMax(TestCase):
    def test_optimum(self):
        dim = 5
        objective = OneMax(dim, minimize=True)
        x = idx2one_hot(np.ones(dim, dtype=np.int), objective.Cmax)
        self.assertEqual(objective.is_optimum(x), True)
        x = idx2one_hot(np.zeros(dim, dtype=np.int), objective.Cmax)
        self.assertEqual(objective.is_optimum(x), False)

    def test_ndarray(self):
        objective = OneMax(3, minimize=False)

        sample = np.array([[[1, 0], [1, 0], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1])), True)

        sample = np.array([[1, 0], [1, 0], [0, 1]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1])), True)

        sample = np.array([[[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1, 2])), True)

        objective = OneMax(3, minimize=True)

        sample = np.array([[[1, 0], [1, 0], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1])), True)

        sample = np.array([[[1, 0], [1, 0], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1])), True)

        sample = np.array([[[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1, 2])), True)

    def test_list(self):
        objective = OneMax(3, minimize=False)

        sample = [[1, 0], [1, 0], [0, 1]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1])), True)

        sample = [[[1, 0], [1, 0], [0, 1]]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1])), True)

        sample = [[[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == np.array([1, 2])), True)

        objective = OneMax(3, minimize=True)

        sample = [[1, 0], [1, 0], [0, 1]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1])), True)

        sample = [[[1, 0], [1, 0], [0, 1]]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1])), True)

        sample = [[[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]]]
        evals, _ = objective(sample)
        self.assertEqual(np.all(evals == -np.array([1, 2])), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            self.assertRaises(OneMax(-1))

        objective = OneMax(3, minimize=False)
        with self.assertRaises(AssertionError):
            sample = [[[1, 0], [0, 1]]]
            self.assertRaises(objective(sample))
        with self.assertRaises(AssertionError):
            sample = [[[1, 0], [0, 1], [1, 0], [0, 1]]]
            self.assertRaises(objective(sample))
        with self.assertRaises(AssertionError):
            sample = [[[[1, 0], [0, 1], [1, 0]]]]
            self.assertRaises(objective(sample))
        with self.assertRaises(AssertionError):
            sample = [[[1, 0, 0], [0, 1, 0], [1, 0, 0]]]
            self.assertRaises(objective(sample))


if __name__ == "__main__":
    unittest.main()
