import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from pyinterpret.data.dataset import DataSet
from pyinterpret.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer


class TestLime(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLime, self).__init__(*args, **kwargs)
        self.setup()

    def setup(self, n=100, dim=3):
        self.dim = dim
        self.n = n
        self.feature_names = ['x{}'.format(i) for i in range(self.dim)]
        self.index = ['{}'.format(i) for i in range(self.n)]
        self.X = np.random.normal(0, 5, size=(self.n, self.dim))
        self.B = np.random.normal(0, 5, size = self.dim)
        self.y = np.dot(self.X, self.B) + np.random.normal(0, 5, size=self.n)
        self.y_for_classifier = np.round(expit(self.y))
        self.example = self.X[0]


    def test_regression_with_feature_names(self):
        regressor = GradientBoostingRegressor()
        regressor.fit(self.X, self.y)
        interpretor = LimeTabularExplainer(self.X, feature_names = self.feature_names)
        interpretor.explain_regressor_instance(self.example, regressor.predict)

    def test_regression_without_feature_names(self):
        regressor = GradientBoostingRegressor()
        regressor.fit(self.X, self.y)
        interpretor = LimeTabularExplainer(self.X)
        interpretor.explain_regressor_instance(self.example, regressor.predict)

    def test_classifier_no_proba_without_feature_names(self):
        classifier = GradientBoostingClassifier()
        classifier.fit(self.X, self.y_for_classifier)
        interpretor = LimeTabularExplainer(self.X)
        interpretor.explain_instance(self.example, classifier.predict)

    def test_classifier_with_proba_without_feature_names(self):
        classifier = GradientBoostingClassifier()
        classifier.fit(self.X, self.y_for_classifier)
        interpretor = LimeTabularExplainer(self.X)
        interpretor.explain_instance(self.example, classifier.predict_proba)

    def test_classifier_no_proba_with_feature_names(self):
        classifier = GradientBoostingClassifier()
        classifier.fit(self.X, self.y_for_classifier)
        interpretor = LimeTabularExplainer(self.X, feature_names= self.feature_names)
        interpretor.explain_instance(self.example, classifier.predict)

    def test_classifier_with_proba_with_feature_names(self):
        classifier = GradientBoostingClassifier()
        classifier.fit(self.X, self.y_for_classifier)
        interpretor = LimeTabularExplainer(self.X, feature_names= self.feature_names)
        interpretor.explain_instance(self.example, classifier.predict_proba)



if __name__ == '__main__':
    unittest.main()
