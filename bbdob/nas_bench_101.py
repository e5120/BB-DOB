import os
import subprocess

import numpy as np
from nasbench import api
from nasbench.lib import graph_util

from bbdob import ObjectiveBase

MAX_EDGES = 9
VERTICES = 7
MATRIX_ELEMENTS = VERTICES * (VERTICES - 1) // 2
OPS = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]


class NasBench101(ObjectiveBase):
    """
    NAS-Bench-101 is a tabular dataset which maps neural network architectures to their trained and evaluated metrics.
    """
    def __init__(self, dim=-1, minimize=True, filename="nasbench_only108.tfrecord"):
        """
        Parameters
        ----------
        dim : int, default -1
            The dimension of the problem.
            In this function, this is a dummy variable, i.e. not used.
        filename : str, default "nasbench_only108.tfrecord"
            "nasbench_full.tfrecord" or "nasbench_only108.tfrecord".
        """
        super(NasBench101, self).__init__(MATRIX_ELEMENTS+VERTICES-2, minimize=minimize)
        self._categories[MATRIX_ELEMENTS:] = 3
        data_dir = "{}/data".format(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists("{}/{}".format(data_dir, filename)):
            os.makedirs(data_dir, exist_ok=True)
            print("downloading data now...")
            subprocess.run("wget -P {} https://storage.googleapis.com/nasbench/{}".format(data_dir, filename), shell=True)
        self.nasbench = api.NASBench('{}/{}'.format(data_dir, filename))
        self.estimated_wall_clock_time = 0
        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def evaluate(self, c):
        c = self._check_shape(c)
        evals = []
        test_error = []
        training_time = []
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for _c in c:
            for i in range(MATRIX_ELEMENTS):
                row = idx[0][i]
                col = idx[1][i]
                matrix[row, col] = _c[i]
            if graph_util.num_edges(matrix) > MAX_EDGES:
                evals.append(1)
                test_error.append(1)
                training_time.append(0)
            else:
                ops = [OPS[i] for i in _c[MATRIX_ELEMENTS:]]
                ops = ["input"] + list(ops) + ["output"]
                model_spec = api.ModelSpec(matrix=matrix, ops=ops)
                try:
                    data = self.nasbench.query(model_spec, epochs=108)
                    evals.append(1 - data["validation_accuracy"])
                    test_error.append(1 - data["test_accuracy"])
                    training_time.append(data["training_time"])
                    self.estimated_wall_clock_time += data["training_time"]
                except api.OutOfDomainError:
                    evals.append(1)
                    test_error.append(1)
                    training_time.append(0)
        evals = np.array(evals)
        evals = evals if self.minimize else -evals
        test_error = np.array(test_error)
        training_time = np.array(training_time)
        info = {"training_time": training_time, "test_error": test_error}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(NasBench101, self).__str__().replace("\n", "\n    ")
        return 'NAS-Bench-101(\n' \
               '{}' \
               '\n)'.format(sup_str)
