import unittest

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from functools import partial
import pandas as pd

from skater.core.explanations import Interpretation
from skater.util import exceptions
from arg_parser import create_parser
from skater.model import InMemoryModel, DeployedModel


class TestModel(unittest.TestCase):

    def setUp(self):
        args = create_parser().parse_args()
        debug = args.debug
        self.seed = args.seed
        self.n = args.n
        self.x1 = np.random.choice(range(10), replace=True, size=self.n)
        self.x2 = np.random.normal(0, 5, size=self.n)
        self.x3 = np.random.choice(['a','b','c'], replace=True, size=self.n)
        self.X = pd.DataFrame({'x1': self.x1, 'x2': self.x2, 'x3': self.x3}).values

        self.y = self.underlying_model_agg(self.X)

        self.y_as_int = np.round(expit(self.y))

        self.y_as_string = np.array([str(i) for i in self.y_as_int])

        self.sample_x = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
        self.sample_y = np.array([-1, -1, -1, 1, 1, 1])
        self.sample_feature_name = [str(i) for i in range(self.sample_x.shape[1])]


    def test_in_memory_gb_regressor_no_args(self):
        model = InMemoryModel(self.underlying_model)


    def underlying_model(self, x1, x2, x3):
        if x3 == 'c' and x1 < 3 and x2 > 0:
            return 1.2 - .7 * x1
        elif x3 == 'c' and x1 < 6 and x2 > 0:
            return 1.2 + .7 * x1
        elif x3 == 'c' and x2 > 0:
            return 5.2 + 1.7 * x1
        elif x3 == 'c':
            return -.2 + 1.3 * x1
        elif x3 == 'b' and x1 < 3 and x2 > 0:
            return 11.2 - 2.7 * x1
        elif x3 == 'b' and x1 < 6 and x2 > 0:
            return 3.2 + 1.7 * x1
        elif x3 == 'b' and x2 > 0:
            return 5.2 - 1.7 * x1
        elif x3 == 'b':
            return 8 + 5.3 * x1
        else:
            return 3.7 * x1 * x2

    def underlying_model_agg(self, x):
        if len(x.shape)==2:
            results = []
            for row in x:
                x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
                results.append(self.underlying_model(x1, x2, x3))
            return np.array(results)
        else:
            x1, x2, x3 = x[0], x[1], x[2]
            result = self.underlying_model(x1, x2, x3)
            return np.array([result])





if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(suite)
