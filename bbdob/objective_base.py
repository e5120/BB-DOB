from abc import ABCMeta, abstractmethod

import numpy as np

from bbdob.utils import idx2one_hot


class ObjectiveBase(metaclass=ABCMeta):
    """
    Base class of the Black-box Discrete Optimization Benchmark.
    """
    def __init__(self, dim, minimize=True):
        """
        Parameters
        ----------
        dim : int
            The dimension of the problem.
        minimize : bool, default True
            Whether the problem is a minimization problem or not.
        """
        assert 0 < dim, \
            "Specify a non-negative integer."
        self.dim = dim
        self.minimize = minimize

        self.optimal_value = -np.inf if minimize else np.inf    # optimal value
        self._categories = np.full(dim, 2)   # default: bit-strings

    @property
    def Cmax(self):
        return np.max(self.categories)

    @property
    def categories(self):
        return self._categories

    def __call__(self, c, **kwargs):
        return self.evaluate(c, **kwargs)

    def __str__(self):
        return 'dim: {}\n' \
               'minimize: {}'.format(self.dim, self.minimize)

    @abstractmethod
    def evaluate(self, c):
        """
        Take a vector or a population which is group of vectors as an input, return the evaluation value of each vector.

        Parameters
        ----------
        c : array-like
            A vector or a population.

        Returns
        -------
        numpy.ndarray
            The evaluation value of each vector.
        """
        pass

    def is_optimum(self, x):
        """
        Check whether the vector is optimum or not.

        Parameters
        ----------
        x : numpy.ndarray
            A vector.

        Returns
        -------
        boolean
            Whether to be an optimum or not.
        """
        assert len(x.shape) == 2, \
            "The shape must be (dim, one_hot)." \
            "The shape of the input was {}".format(x.shape)
        evals, _ = self.evaluate(x)
        return evals.item() == self.optimal_value

    def _check_shape(self, c):
        """
        Parameters
        ----------
        c : array-like
            A vector or a population.
            If c is a vector, assume that the shape of c is (dim, one-hot), otherwise (population_size, dim, one-hot).

        Returns
        -------
        numpy.ndarray
            A population after the input c was converted to ndarray.
            The shape is (population_size, dim).
        """
        assert isinstance(c, (list, tuple, np.ndarray)), \
            "Input is required to be of type list, tuple, or numpy.ndarray."
        c = np.array(c)
        assert 2 <= len(c.shape) <= 3, \
            "The shape must be ({0}, {1}) or (population_size, {0}, {1}).\n\t\t" \
            "The shape of the input was {2}".format(self.dim, self.Cmax, c.shape)
        # convert a vector to a population whose size is one.
        if len(c.shape) == 2:
            c = c[np.newaxis]
        _, dim, cardinality = c.shape
        assert dim == self.dim, \
            "The dimension of the vector ({}) does not " \
            "match that of the problem ({}).\n".format(dim, self.dim)
        assert cardinality == self.Cmax, \
            "The cardinality of the vector does not match that of the problem.\n" \
            "Input ({}) and problem({})".format(cardinality, self.Cmax)
        c = np.argmax(c, axis=2)
        return c
