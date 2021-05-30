import unittest
from unittest import TestCase, main

import numpy as np

from bbdob import NasBench101
from bbdob.utils import idx2one_hot


class TestNasBench101(TestCase):
    def test_ndarray(self):
        np.random.seed(1)
        sample = np.random.randint(0, 2, (10, 26))
        sample[:, 21:] = np.random.randint(0, 3, (10, 5))
        objective = NasBench101()
        evals, info = objective(idx2one_hot(sample, 3))
        print(evals, info)
        print(objective)

if __name__ == "__main__":
    unittest.main()
