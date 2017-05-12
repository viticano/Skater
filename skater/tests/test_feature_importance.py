import unittest

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from functools import partial

from skater.core.explanations import Interpretation
from skater.util import exceptions
from arg_parser import create_parser
from skater.model import InMemoryModel, DeployedModel


class TestFeatureImportance(unittest.TestCase):

    def setUp(self):
        args = create_parser().parse_args()
        debug = args.debug
        self.seed = args.seed
        self.n = args.n
        self.dim = args.dim
        self.features = [str(i) for i in range(self.dim)]
        self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
        self.B = np.array([-10.1, 2.2, 6.1])
        self.y = np.dot(self.X, self.B)
        self.y_as_int = np.round(expit(self.y))
        self.y_as_string = np.array([str(i) for i in self.y_as_int])
        # example dataset for y = B.X
        # X = array([[ 1.62434536, -0.61175641, -0.52817175], ... [-0.15065961, -1.40002289, -1.30106608]])  (1000 * 3)
        # B = array([-10.1,   2.2,   6.1])
        # y = array([ -2.09736000e+01,  -1.29850618e+00,  -1.73511155e+01, ...]) (1000 * 1)
        # features = ['0', '1', '2']
        ##
        # Other output types:
        # y_as_int = array[ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1., ...]
        # y_as_string = array['0.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', ... ]


        # Another set of input
        # sample data
        self.sample_x = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
        self.sample_y = np.array([-1, -1, -1, 1, 1, 1])
        self.sample_feature_name = [str(i) for i in range(self.sample_x.shape[1])]

        if debug:
            self.interpreter = Interpretation(log_level='DEBUG')
        else:
            self.interpreter = Interpretation()  # default level is 'WARNING'
        self.interpreter.load_data(self.X, feature_names=self.features)

        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = InMemoryModel(self.regressor.predict, examples=self.X)

        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_as_int)
        self.classifier_predict_fn = InMemoryModel(self.classifier.predict, examples=self.X, unique_values=self.classifier.classes_)
        self.classifier_predict_proba_fn = InMemoryModel(self.classifier.predict_proba, examples=self.X)

        self.string_classifier = LogisticRegression()
        self.string_classifier.fit(self.X, self.y_as_string)
        self.string_classifier_predict_fn = InMemoryModel(self.string_classifier.predict_proba, examples=self.X)



    @staticmethod
    def feature_column_name_formatter(columnname):
        return "feature: {}".format(columnname)


    def test_feature_importance(self):
        importances = self.interpreter.feature_importance.feature_importance(self.regressor_predict_fn)
        self.assertEquals(np.isclose(importances.sum(), 1), True)  # default grid resolution is 100


    def test_plot_feature_importance(self):
        self.interpreter.feature_importance.plot_feature_importance(self.regressor_predict_fn)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureImportance)
    unittest.TextTestRunner(verbosity=2).run(suite)
