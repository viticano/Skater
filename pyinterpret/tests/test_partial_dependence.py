import unittest

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from pyinterpret.core.explanations import Interpretation


class TestPartialDependence(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPartialDependence, self).__init__(*args, **kwargs)
        self.build_data()
        self.build_regressor()

    def build_data(self, n=1000, seed=1, dim=3):
        self.seed = seed
        self.n = n
        self.dim = dim
        self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
        self.B = np.array([-10.1, 2.2, 6.1])
        self.y = np.dot(self.X, self.B)

    def build_regressor(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = self.regressor.predict
        self.regressor_feature = 0

    def test_pdp_with_default_sampling(self):
        interpreter = Interpretation()
        interpreter.consider(self.X)
        coefs = interpreter.partial_dependence.partial_dependence([self.regressor_feature], self.regressor_predict_fn,
                                                                  sample=True)

    def test_pdp_regression_coefs_are_close_1d(self, epsilon=1):
        interpreter = Interpretation()
        interpreter.consider(self.X)
        coefs = interpreter.partial_dependence.partial_dependence([self.regressor_feature], self.regressor_predict_fn)

        y = np.array(coefs['means'])
        x = np.array(coefs['vals'])

        pdp_reg = LinearRegression()
        pdp_reg.fit(x, y)
        coef = pdp_reg.coef_[0]
        coefs_are_close_warning = "Lime coefficients for regressions model are not close to true values for trivial case"
        coefs_are_close = abs(coef - self.B[self.regressor_feature]) < epsilon
        if not coefs_are_close:
            coefs_are_close_warning += "True Coefs: {}".format(self.B[self.regressor_feature])
            coefs_are_close_warning += "Estimated Coefs: {}".format(coef)
            self.fail(coefs_are_close_warning)


if __name__ == '__main__':
    unittest.main()
