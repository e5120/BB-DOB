import numpy as np

from bbdob import ObjectiveBase


class OneMax(ObjectiveBase):
    """
    A class of One-Max function.

    f(c) = \sum_{i=1}^{D}(c_i),
    where c = (c_1, c_2, ..., c_D) \in {0, 1}^{D}.
    """
    def __init__(self, dim, minimize=True):
        super(OneMax, self).__init__(dim, minimize=minimize)
        self.optimal_value = -dim if minimize else dim

    def evaluate(self, c):
        c = self._check_shape(c)
        evals = np.sum(c, axis=1)
        evals = -evals if self.minimize else evals
        info = {}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(OneMax, self).__str__().replace("\n", "\n    ")
        return 'OneMax(\n' \
               '{}' \
               '\n)'.format(sup_str)
