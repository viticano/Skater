import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from data import DataSet
import datetime

class Explainer(object):
	def __init__(self, training_data, index=None, feature_names=None,
				 categorical_features=None, categorical_names=None, class_names=None):
		
		self.DataSet = DataSet(training_data, index = index, feature_names = feature_names)
		


	def explain_instance(self, data_row, predict_fn, sample = False, 
							n_samples = 5000, sampling_strategy = 'uniform-over-similarity-ranks',
							distance_metric='euclidean', kernel_width = None, 
							explainer_model = None):

		if kernel_width is None:
			kernel_width = np.sqrt(self.DataSet.dim) * .75	

		if explainer_model == None:
			explainer_model = LinearRegression

		kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))		

		neighborhood = self.DataSet.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)
		distances = sklearn.metrics.pairwise_distances(
		neighborhood,
		data_row.reshape(1, -1),
		metric=distance_metric).ravel()
		weights = kernel_fn(distances)
		predictions = predict_fn(neighborhood)
		explainer_model = explainer_model()
		explainer_model.fit(neighborhood, predictions, sample_weight = weights)
		return explainer_model.coef_


		

	def partial_dependence(self, feature_ids, predict_fn, grid = None, grid_resolution = 100, 
					grid_range = (0.03, 0.97), sample = False, n_samples = 5000,
					sampling_strategy = 'uniform-over-similarity-ranks'):
		assert all(feature_id in self.DataSet.feature_ids for feature_id in feature_ids), "Pass in a valid ID"	
		
		if not grid :
			grid = self.DataSet.generate_grid(feature_ids, grid_resolution = grid_resolution, grid_range = grid_range)
		
		X = self.DataSet.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)	
		
		pdp_vals = {}
		
		for feature_id, grid_column in zip(feature_ids, grid):
			X_mutable = X.copy()
			pdp_vals[feature_id] = {}
			for value in grid_column:
				X_mutable[feature_id] = value
				predictions = predict_fn(X_mutable.values)
				pdp_vals[feature_id][value] = {'mean':np.mean(predictions), 'std':np.std(predictions)}

		return pdp_vals


	

def log(current_logs, last, message):
	current =  datetime.datetime.now()
	diff = current - last
	current_logs.append([message, diff])
	return current_logs, current
					