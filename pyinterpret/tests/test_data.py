
import unittest
from pyinterpret.base import DataSet
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd


class TestData(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestData, self).__init__(*args, **kwargs)
		self.setup()

	def setup(self, n=100, dim=3):
		self.dim = dim
		self.n = n
		self.feature_names = ['x{}'.format(i) for i in range(self.dim)]
		self.index = ['{}'.format(i) for i in range(self.n)]
		self.X = np.random.normal(0, 5, size=(self.n, self.dim))

	def test_dataset_data_returns_original(self):
		data_set = DataSet(self.X, feature_names=self.feature_names, index=self.index)
		assert_array_equal(data_set.data, self.X)

	def test_1D_numpy_array(self):
		array_1d = np.random.normal(0, 1, size=self.n)
		data_set = DataSet(array_1d)


	def test_2D_numpy_array(self):
		array_2d = self.X
		data_set = DataSet(array_2d, feature_names=self.feature_names, index=self.index)

	def test_pandas_dataframe(self):
		X_as_dataframe = pd.DataFrame(self.X, columns=self.feature_names, index=self.index)
		data_set = DataSet(X_as_dataframe)


		feature_names_warning = "Feature Names from DataFrame not loaded properly"
		feature_names_are_correct = data_set.feature_ids == X_as_dataframe.columns.values.tolist()
		if not feature_names_are_correct:
			feature_names_warning +=  "\n dataset feature ids: {}".format(data_set.feature_ids)
			feature_names_warning +=  "\n input columns: {}".format(X_as_dataframe.columns.values.tolist())
			self.fail(feature_names_warning)

		index_warning = "Index from DataFrame not loaded properly"
		index_is_correct = data_set.index == X_as_dataframe.index.values.tolist()
		if not index_is_correct:
			index_warning +=  "\n dataset feature ids: {}".format(data_set.index)
			index_warning +=  "\n input columns: {}".format(X_as_dataframe.index.values.tolist())
			self.fail(index_warning)


	def test_generate_grid_1_variable(self):
		array_2d = self.X
		data_set = DataSet(array_2d, feature_names=self.feature_names, index=self.index)
		grid = data_set.generate_grid(self.feature_names[0:1], grid_resolution=100)
		assert grid.shape == (1, 100)


	def test_generate_grid_2_variables(self):
		array_2d = self.X
		data_set = DataSet(array_2d, feature_names=self.feature_names, index=self.index)
		grid = data_set.generate_grid(self.feature_names[0:2], grid_resolution=100)
		assert grid.shape == (2, 100)

if __name__ == '__main__':
	unittest.main()
