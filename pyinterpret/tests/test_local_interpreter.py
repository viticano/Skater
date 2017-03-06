import unittest

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from pyinterpret.core.explanations import Interpretation


class TestLocalInterpreter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLocalInterpreter, self).__init__(*args, **kwargs)
        self.build_data()
        self.build_regressor()
        self.build_classifier()

    def build_data(self, n=1000, seed=1, dim=3):
        self.seed = seed
        self.n = n
        self.dim = dim
        self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
        self.B = np.array([-10.1, 2.2, 6.1])
        self.y = np.dot(self.X, self.B)
        self.y_as_prob = expit(self.y)
        self.y_as_ints = np.array([np.random.choice((0, 1), p=(1 - prob, prob)) for prob in self.y_as_prob.reshape(-1)])


    def build_regressor(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = self.regressor.predict
        self.regressor_point = self.X[0]

    def build_classifier(self):
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_as_ints)
        self.classifier_predict_fn = self.classifier.predict_proba
        self.classifier_point = self.X[0]

    def test_lime_regression_coefs_are_close(self, epsilon=1):
        interpreter = Interpretation()
        interpreter.consider(self.X)
        coefs = interpreter.local_interpreter.lime_ds(self.regressor_point, self.regressor_predict_fn)

        coefs_are_close_warning = "Lime coefficients for regressions model are not close to true values for trivial case"
        coefs_are_close = all(abs(coefs - self.B) < epsilon)
        if not coefs_are_close:
            coefs_are_close_warning += "True Coefs: {}".format(self.B)
            coefs_are_close_warning += "Estimated Coefs: {}".format(coefs)
            self.fail(coefs_are_close_warning)

    def test_lime_classifier_coefs_correct_sign(self):
        interpreter = Interpretation()
        interpreter.consider(self.X)
        neg_coefs, pos_coefs = interpreter.local_interpreter.lime_ds(self.classifier_point, self.classifier_predict_fn)

        coefs_are_correct_sign_warning = "Lime coefficients for classifier model are not correct sign for trivial case"
        coefs_are_correct_sign = all(np.sign(pos_coefs) == np.sign(self.B))
        if not coefs_are_correct_sign:
            coefs_are_correct_sign_warning += "True Coefs: {}".format(self.B)
            coefs_are_correct_sign_warning += "Estimated Coefs: {}".format(pos_coefs)
            self.fail(coefs_are_correct_sign_warning)

    def test_initializing_default_lime(self):
        interpreter = Interpretation()
        input_feature_names = ['a', 'b', 'c']
        input_df = pd.DataFrame(np.random.randn(5, 4), columns=['a', 'b', 'c', 'd'])
        input_class_names = [0, 1]
        #explainer = interpreter.local_interpreter.local_explainer(input_df, class_names=input_class_names,
        #                                             feature_names=input_feature_names)
        #explainer = interpreter.local_interpreter.local_explainer()
        #import pdb
        #pdb.set_trace()

    def test_coefs_are_non_zero_for_breast_cancer_dataset(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target
        example = X[0]
        model = RandomForestClassifier()
        model.fit(X, y)
        interpreter = Interpretation()
        interpreter.consider(X)
        lime_coef_ = interpreter.local_interpreter.lime_ds(example, model.predict_proba)
        #TODO : check on this function
        #assert (lime_coef_ != 0).any(), "All coefficients for this are 0, maybe a bad kernel width"

if __name__ == '__main__':
    unittest.main()
