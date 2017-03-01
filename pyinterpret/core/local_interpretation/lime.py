import numpy as np
import pandas as pd
from .base import BaseLocalInterpretation
from sklearn.linear_model import LinearRegression
import sklearn


class Lime(BaseLocalInterpretation):
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
		neighborhood = self.interpreter.data_set.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)
		
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