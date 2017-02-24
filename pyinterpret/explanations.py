import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from data import DataSet
import datetime
import abc

class ModelInterpreter(object):
	'''
	Base Interpreter class. Common methods include loading a dataset and type setting.
	'''

	def __init__(self, interpretation_type):
		self.type = interpretation_type
		self.data_set = None

	@staticmethod
	def _types():
		return ['pdp','lime']

	def consider(self, training_data, index = None, feature_names = None):
		
		self.data_set = DataSet(training_data, index = None, feature_names = None)




# Create based on class name:
def interpretation(interpretation_type):
	'''
	Returns an interpretation class.

	Parameters:
	-----------
		interpretation_type(string): pdp, lime

	Returns:
	----------
		interpretation subclass
	'''
	
	class PartialDependency(ModelInterpreter):
			
		def partial_dependence(self, feature_ids, predict_fn, grid = None, grid_resolution = 100, 
						grid_range = (0.03, 0.97), sample = False, n_samples = 5000,
						sampling_strategy = 'uniform-over-similarity-ranks'):

			'''
			Computes partial_dependence of a set of variables. Essentially approximates
			the partial partial_dependence of the predict_fn with respect to the variables
			passed.

			Parameters:
			-----------
			feature_ids(list): 
				the names/ids of the features for which we compute partial dependence. Note that
				the algorithm's complexity scales exponentially with additional features, so generally
				one should only look at one or two features at a time. These feature ids must be avaiable 
				in the class's associated DataSet.

			predict_fn(function):
				machine learning that takes data and returns an output. Acceptable output formats are ????.
				Supports classification, multiclass classification, and regression.

			grid(numpy.ndarray):
				2 dimensional array on which we fix values of features. Note this is determined automatically
				if not given based on the percentiles of the dataset.

			grid_resolution(int):
				how many unique values to include in the grid. If the percentile range is 5% to 95%, then that
				range will be cut into <grid_resolution> equally size bins.

			grid_range(tuple):
				the percentile extrama to consider. 2 element tuple, increasing, bounded between 0 and 1.

			sample(Bool):
				Whether to sample from the original dataset.

			sampling_strategy(string):
				If sampling, which approach to take. See DataSet.generate_sample for details.


			'''
			assert all(feature_id in self.data_set.feature_ids for feature_id in feature_ids), "Pass in a valid ID"  

			
		
			#if you dont pass a grid, build one.
			if not grid :
				grid = self.data_set.generate_grid(feature_ids, grid_resolution = grid_resolution, grid_range = grid_range)			
			
			#make sure data_set module is giving us correct data structure
			self._check_grid(grid)

			#generate data
			X = self.data_set.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)				 
			
			#make sure data_set module is giving us correct data structure
			self._check_X(X)

			#will store [featurename: {val1: {mean:<>, std:<>}, etc...}]
			pdp_vals = {}
			
			#is this a safe operation?
			for feature_id, grid_column in zip(feature_ids, grid):
				#pandas dataframe
				X_mutable = X.copy()
				pdp_vals[feature_id] = {}
				for value in grid_column:
					#fix value of data
					X_mutable[feature_id] = value
					#pass mutated/perturbed data to predict function
					predictions = predict_fn(X_mutable.values)
					#capture stats of the predictions
					pdp_vals[feature_id][value] = {'mean':np.mean(predictions), 'std':np.std(predictions)}

			return pdp_vals

		def partial_dependency_sklearn(self):
			pass

		@staticmethod
		def _check_grid(grid):
			assert isinstance(grid, np.ndarray), "Grid is not a numpy array"
			assert len(grid.shape) == 2, "Grid is not 2D"

		@staticmethod
		def _check_X(X):
			assert isinstance(X, pd.DataFrame)


	class LocalInterpretation(ModelInterpreter):
	 
		def lime_ds(self, data_row, predict_fn, sample = False, 
							n_samples = 5000, sampling_strategy = 'uniform-over-similarity-ranks',
							distance_metric='euclidean', kernel_width = None, 
							explainer_model = None):

			"""
			For performing Local interpretable model explanation. Creates a sample of data around of point,
			passing these points through your prediction function. This gives a linear model of the full model 
			centered around a point.

			Parameters
			----------
				data_row(numpy.ndarray): 
					The observation to explain. 
				predict_fn(function):
					The model to explain
				sample(Bool):
					Whether or not to sample the data. 
				distance_metric(string):
					The measure by which we evaluate distane between a point in the sampled data
					and the data row to explain.
				kernel_width(numeric):
					We pass distances through a kernel function, this specifies the width. Smaller widths
					will give more weight to points neighboring the explained point. Defaults to proportional
					to dimensionality of dataset.
				explainer_model(scikit learn model):
					Model to use to explain the point. Defaults to Linear regression.

			"""

			if kernel_width is None:
				kernel_width = np.sqrt(self.data_set.dim) * .75  

			if explainer_model == None:
				explainer_model = LinearRegression

			explainer_model = explainer_model()
			self._check_explainer_model_pre_train(explainer_model)				


			kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))		

			#data that has been sampled
			neighborhood = self.data_set.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)
			
			self._check_neighborhood(neighborhood)

			distances = sklearn.metrics.pairwise_distances(
				neighborhood,
				data_row.reshape(1, -1),
				metric=distance_metric) \
				.ravel()

			weights = kernel_fn(distances)
			predictions = predict_fn(neighborhood)
			explainer_model.fit(neighborhood, predictions, sample_weight = weights)
			self._check_explainer_model_post_train(explainer_model)

			return explainer_model.coef_		
			
		def lime(self):
			pass

		@staticmethod
		def _check_explainer_model_pre_train(explainer_model):
			assert hasattr(explainer_model, 'fit'), "Model needs to have a fit method "

		@staticmethod
		def _check_explainer_model_post_train(explainer_model):
			assert hasattr(explainer_model, 'coef_'), "Model needs to have coefficients to explain "

		@staticmethod
		def _check_neighborhood(neighborhood):
			assert isinstance(neighborhood, (np.ndarray, pd.DataFrame))

	if interpretation_type == "pdp": 
		return PartialDependency(interpretation_type)
	if interpretation_type == "lime": 
		return LocalInterpretation(interpretation_type)
	else:
		raise KeyError("interpretation_type needs to be element of {}".format(ModelInterpreter._types()))
	