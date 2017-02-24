import unittest
from PyInterpret.data import DataSet
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

class TestData(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TestData, self).__init__(*args, **kwargs)
		self.setup()

	def setup(self, N = 100, dim = 3):
		self.dim = dim
		self.N = N
		self.feature_names = ['x{}'.format(i) for i in range(self.dim)]
		self.index = ['{}'.format(i) for i in range(self.N)]
		self.X = np.random.normal(0, 5, size = (self.N, self.dim))
		

	def test_dataset_data_returns_original(self):
		data_set = DataSet(self.X, feature_names = self.feature_names, index = self.index)		
		assert_array_equal(data_set.data, self.X)


	def test_1D_numpy_array(self):
		array_1d = np.random.normal(0, 1, size = self.N)
		try:
			data_set = DataSet(array_1d)
		except:
			self.fail('Could not load 1D array')


	def test_2D_numpy_array(self):	
		array_2d = self.X
		try:
			data_set = DataSet(array_2d)
		except:
			self.fail('Could not load 2D array')		

	
	def test_pandas_dataframe(self):
		dataframe = pd.DataFrame(self.X, columns = self.feature_names, index = self.index)
		try:
			data_set = DataSet(dataframe)
		except:
			self.fail('Could not load pandas dataframe')				


if __name__ == '__main__':
	unittest.main()