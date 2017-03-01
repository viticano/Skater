import unittest
from pyinterpret.explanations import Interpretation
import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression


class TestLime(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestLime, self).__init__(*args, **kwargs)
		self.build_data()
		self.build_regressor()
		self.build_classifier()
	def build_data(self, n = 1000, seed = 1, dim = 3):
		self.seed = seed
		self.n = n
		self.dim = dim
		self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
		self.B = np.array([-10.1, 2.2, 6.1])
		self.y = np.dot(self.X, self.B)
		self.y_as_prob = expit(self.y)
		self.y_as_ints = np.array([np.random.choice((0, 1), p=(1-prob,prob)) for prob in self.y_as_prob.reshape(-1)])


	def build_regressor(self):
		self.regressor = LinearRegression()
		self.regressor.fit(self.X,self.y)
		self.regressor_predict_fn = self.regressor.predict
		self.regressor_point = self.X[0]

	def build_classifier(self):
		self.classifier = LogisticRegression()
		self.classifier.fit(self.X,self.y_as_ints)
		self.classifier_predict_fn = self.classifier.predict_proba
		self.classifier_point = self.X[0]

	def test_lime_regression_coefs_are_close(self, epsilon = 1):
		lime_for_regressor = Interpretation('lime')
		lime_for_regressor.consider(self.X)
		coefs = lime_for_regressor.lime_ds(self.regressor_point, self.regressor_predict_fn)

		coefs_are_close_warning = "Lime coefficients  for regressions model are not close to true values for trivial case"
		coefs_are_close = all(abs(coefs - self.B) < epsilon)
		if not coefs_are_close:
			coefs_are_close_warning += "True Coefs: {}".format(self.B)
			coefs_are_close_warning += "Estimated Coefs: {}".format(coefs)
			self.fail(coefs_are_close_warning)

	def test_lime_classifier_coefs_correct_sign(self):
		lime_for_classifier = Interpretation('lime')
		lime_for_classifier.consider(self.X)
		neg_coefs, pos_coefs = lime_for_classifier.lime_ds(self.classifier_point, self.classifier_predict_fn)

		coefs_are_correct_sign_warning = "Lime coefficients for classifier model are not correct sign for trivial case"
		coefs_are_correct_sign = all(np.sign(pos_coefs) ==  np.sign(self.B))
		if not coefs_are_correct_sign:
			coefs_are_correct_sign_warning += "True Coefs: {}".format(self.B)
			coefs_are_correct_sign_warning += "Estimated Coefs: {}".format(pos_coefs)
			self.fail(coefs_are_correct_sign_warning)



if __name__ == '__main__':
	unittest.main()