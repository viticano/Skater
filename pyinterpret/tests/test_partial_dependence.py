import unittest

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from functools import partial
import sys
import argparse

from pyinterpret.core.explanations import Interpretation
from pyinterpret.util import exceptions

class TestPartialDependence(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()
        args = self.parser.parse_args()
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
        if debug:
            self.interpreter = Interpretation(log_level=10)
        else:
            self.interpreter = Interpretation(log_level=30)
        self.interpreter.load_data(self.X, feature_names=self.features)

        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = self.regressor.predict
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_as_int)
        self.classifier_predict_fn = self.classifier.predict
        self.classifier_predict_proba_fn = self.classifier.predict_proba
        self.string_classifier = LogisticRegression()
        self.string_classifier.fit(self.X, self.y_as_string)

    @staticmethod
    def feature_column_name_formatter(columnname):
        return "val_{}".format(columnname)

    def test_pdp_with_default_sampling(self):
        coefs = self.interpreter.partial_dependence.partial_dependence([self.features[0]],
                                                                       self.regressor_predict_fn,
                                                                       sample=True)

    def test_pdp_regression_coefs_are_close_1d(self, epsilon=1):
        coefs = self.interpreter.partial_dependence.partial_dependence([self.features[0]],
                                                                       self.regressor_predict_fn)
        val_col = self.feature_column_name_formatter(self.features[0])
        y = np.array(coefs['mean'])
        x = np.array(coefs[val_col])[:, np.newaxis]
        pdp_reg = LinearRegression()
        pdp_reg.fit(x, y)
        self.interpreter.logger.debug("PDP coefs: {}".format(pdp_reg.coef_))
        self.interpreter.logger.debug("PDP coef shape: {}".format(pdp_reg.coef_.shape))
        coef = pdp_reg.coef_[0]
        coefs_are_close_warning = "Lime coefficients for regressions model are not " \
                                  "close to true values for trivial case"
        coefs_are_close = abs(coef - self.B[0]) < epsilon
        if not coefs_are_close:
            coefs_are_close_warning += "True Coefs: {}".format(self.B[self.features[0]])
            coefs_are_close_warning += "Estimated Coefs: {}".format(coef)
            self.fail(coefs_are_close_warning)

    def test_2D_pdp(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                       self.regressor_predict_fn,
                                                                       grid_resolution=10,
                                                                       sample=True)
    def test_plot_1D_pdp(self):
        coefs = self.interpreter.partial_dependence.plot_partial_dependence([self.features[0]],
                                                                       self.regressor_predict_fn,
                                                                       grid_resolution=10)

    def test_plot_1D_pdp_with_sampling(self):
        coefs = self.interpreter.partial_dependence.plot_partial_dependence(
            [self.features[0]],
            self.regressor_predict_fn,
            grid_resolution=10,
            sample=True
        )
    def test_plot_2D_pdp(self):
        coefs = self.interpreter.partial_dependence.plot_partial_dependence(self.features[:2],
                                                                       self.regressor_predict_fn,
                                                                       grid_resolution=10,
                                                                       sample=False)
    def test_plot_2D_pdp_with_sampling(self):
        coefs = self.interpreter.partial_dependence.plot_partial_dependence(self.features[:2],
                                                                       self.regressor_predict_fn,
                                                                       grid_resolution=10,
                                                                       sample=True)
    def test_fail_when_grid_range_is_outside_0_and_1(self):
        pdp_func = partial(self.interpreter.partial_dependence.partial_dependence,
                           *[[self.features[0]], self.regressor_predict_fn],
                           **{'grid_range':(.01, 1.01)})
        self.assertRaises(exceptions.MalformedGridRangeError, pdp_func)

    def test_pdp_1d_classifier_no_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                       self.classifier_predict_fn,
                                                                       grid_resolution=10)

    def test_pdp_2d_classifier_no_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                            self.classifier_predict_fn,
                                                                            grid_resolution=10)

    def test_pdp_1d_classifier_with_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                            self.classifier_predict_proba_fn,
                                                                            grid_resolution=10)

    def test_pdp_2d_classifier_with_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                            self.classifier_predict_proba_fn,
                                                                            grid_resolution=10)

    def test_pdp_1d_string_classifier_no_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                       self.string_classifier.predict,
                                                                       grid_resolution=10)

    def test_pdp_1d_string_classifier_with_proba(self):
        coefs = self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                       self.string_classifier.predict_proba,
                                                                       grid_resolution=10)

    #TODO: Add tests for various kinds of kwargs like sampling for pdp funcs

def arg_parse(args):
    parser = create_parser()
    return parser.parse_args(args)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n', default=1000, type=int)
    parser.add_argument('--dim', default=3, type=int)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    unit_argv = [sys.argv[0]] + args.debug
    unittest.main(argv=unit_argv)
