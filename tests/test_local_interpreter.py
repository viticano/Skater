"""Test local interpretations (not lime)"""
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from skater.core.explanations import Interpretation
from arg_parser import arg_parse, create_parser

class TestLocalInterpreter(unittest.TestCase):

    def setUp(self):

        self.parser = create_parser()
        args = self.parser.parse_args()
        debug = args.debug
        self.seed = args.seed
        self.n = args.n
        self.dim = args.dim
        self.features = range(self.dim)
        self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
        self.B = np.array([-10.1, 2.2, 6.1])
        self.y = np.dot(self.X, self.B)
        self.y_as_prob = expit(self.y)
        self.y_as_ints = np.array(
            [np.random.choice((0, 1), p=(1 - prob, prob)) for prob in self.y_as_prob.reshape(-1)]
        )
        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = self.regressor.predict
        self.regressor_point = self.X[0]

        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_as_ints)
        self.classifier_predict_fn = self.classifier.predict_proba
        self.classifier_point = self.X[0]

        if debug:
            self.interpreter = Interpretation(log_level=10)
        else:
            self.interpreter = Interpretation(log_level=30)
        self.interpreter.load_data(self.X)

    def test_lime_regression_coefs_are_close(self, epsilon=1):
        coefs = self.interpreter.local_interpreter._ds_explain(
            self.regressor_point, self.regressor_predict_fn
        )

        coefs_are_close_warning = \
            "Lime coefficients for regressions model " \
            "are not close to true values for trivial case"
        coefs_are_close = all(abs(coefs - self.B) < epsilon)
        if not coefs_are_close:
            coefs_are_close_warning += "True Coefs: {}".format(self.B)
            coefs_are_close_warning += "Estimated Coefs: {}".format(coefs)
            self.fail(coefs_are_close_warning)

    def test_lime_classifier_coefs_correct_sign(self):
        interpreter = Interpretation()
        interpreter.load_data(self.X)
        neg_coefs, pos_coefs = interpreter.local_interpreter._ds_explain(
            self.classifier_point, self.classifier_predict_fn
        )

        coefs_are_correct_sign_warning = \
            "Lime coefficients for classifier model are " \
            "not correct sign for trivial case"
        coefs_are_correct_sign = all(np.sign(pos_coefs) == np.sign(self.B))
        if not coefs_are_correct_sign:
            coefs_are_correct_sign_warning += "True Coefs: {}".format(self.B)
            coefs_are_correct_sign_warning += "Estimated Coefs: {}".format(pos_coefs)
            self.fail(coefs_are_correct_sign_warning)

    # def test_initializing_default_lime(self):
    #     interpreter = Interpretation()
    #     input_feature_names = ['a', 'b', 'c']
    #     input_df = pd.DataFrame(np.random.randn(5, 4), columns=['a', 'b', 'c', 'd'])
    #     input_target_names = [0, 1]
    #     explainer = interpreter.local_interpreter.local_explainer(input_df, target_names=input_target_names,
    #                                                 feature_names=input_feature_names)
    #     explainer = interpreter.local_interpreter.local_explainer()
    #     import pdb
    #     pdb.set_trace()

    def test_coefs_are_non_zero_for_breast_cancer_dataset(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target
        example = X[0]
        model = RandomForestClassifier()
        model.fit(X, y)
        self.interpreter.load_data(X)
        lime_coef_ = self.interpreter.local_interpreter._ds_explain(example, model.predict_proba)

        #TODO : check on this function
        #assert (lime_coef_ != 0).any(), "All coefficients for this are 0, maybe a bad kernel width"

#if __name__ == '__main__':
    #unittest.main()
