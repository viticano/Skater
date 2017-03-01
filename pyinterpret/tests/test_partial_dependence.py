import unittest
from pyinterpret.explanations import Interpretation
import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression


class TestPartialDependence(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestPartialDependence, self).__init__(*args, **kwargs)
		self.build_data()
		self.build_regressor()

	def build_data(self, n = 1000, seed = 1, dim = 3):
		self.seed = seed
		self.n = n
		self.dim = dim
		self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
		self.B = np.array([-10.1, 2.2, 6.1])
		self.y = np.dot(self.X, self.B)



	def build_regressor(self):
		self.regressor = LinearRegression()
		self.regressor.fit(self.X,self.y)
		self.regressor_predict_fn = self.regressor.predict
		self.regressor_feature = 0


	def test_pdp_regression_coefs_are_close(self, epsilon = 1):
		pdp_for_regressor = Interpretation('partial_dependence')
		pdp_for_regressor.consider(self.X)
		coefs = pdp_for_regressor.partial_dependence([self.regressor_feature], self.regressor_predict_fn)


		coef_vals = [(i,coefs[self.regressor_feature][i]['mean']) for i in coefs[self.regressor_feature]]

		x, y = zip(*coef_vals)
		x = np.array(x)[:, np.newaxis]
		y = np.array(y)
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