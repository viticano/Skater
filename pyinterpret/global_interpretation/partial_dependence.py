from .base import BaseGlobalInterpretation

class PartialDependence(BaseGlobalInterpretation):
		
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