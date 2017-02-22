from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from scipy import stats
import pandas as pd

class DataSet(object):
	def __init__(self, data, feature_names = None, index = None, n_percentiles = 100):

		assert isinstance(data, np.ndarray), 'Data needs to be a numpy array'

		ndim = len(data.shape) 
		
		if ndim == 1:
			data = data[: np.newaxis]

		elif ndim >= 3:
			raise ValueError("Data needs to be 1 or 2 dimensions, yours is {}".format(ndim))

		self.n, self.dim = data.shape
		if not feature_names:
			feature_names = range(self.dim)
		if not index:
			index = range(self.n)	

		self.feature_ids = feature_names	
		self.index = index
		self.data = pd.DataFrame(data, columns = self.feature_ids, index = self.index)
		self.metastore = None
		


	def generate_grid(self, feature_ids, grid_resolution = None, grid_range = None):
		assert all(i >= 0 and i <= 1 for i in grid_range), "Grid range values must be between 0 and 1"

		grid_range = map(lambda x: x* 100, grid_range)
		bins = np.linspace(*grid_range, num=grid_resolution)
		grid = []
		for feature_id in feature_ids:
			vals = np.percentile(self[feature_id], bins)
			grid.append(vals)
		return np.array(grid)
		


	def build_metastore(self):
		#self.bins = (np.array(range(self.n_percentiles)) / float(self.n_percentiles)) * 100
		#self.percentiles = np.array(map(lambda i: np.percentile(self.data, i, axis=0), self.bins))
		#last = (self.percentiles[-1] - self.percentiles[-2]).reshape(1, self.dim)
		#self._augment = np.concatenate((self.percentiles.copy(), last))[1:]
		medians = np.median(self.data.values, axis = 0).reshape(1, self.dim)
		dists = cosine_distances(self.data.values, Y = medians).reshape(-1)
		dist_percentiles = map(lambda i: int(stats.percentileofscore(dists, i)), dists)
		ranks = pd.Series(self.dists).rank().values
		ranks_rounded = np.array(map(lambda x: round(x, 2), self.ranks / self.ranks.max()))
		return {
			'median':median,
			'dists':dists,
			'dist_percentiles':dist_percentiles,
			'ranks':ranks,
			'ranks_rounded':ranks_rounded,
		}



	def __getitem__(self, key):
		assert key in self.feature_ids, "The key {} is not the set of feature_ids {}".format(*[key, self.feature_ids])
		return self.data.__getitem__(key)

	def __setitem__(self, key, newval):
		self.data.__setitem__(key, newval)
		
	def generate_sample(self, sample = True, n_samples_from_dataset = 1000, strategy = 'random-choice', replace = True, samples_per_bin = 20):

		if not sample:
			return self.data
		
		samples = []

		if strategy == 'random-choice':
			idx = np.random.choice(self.index, size = n_samples_from_dataset, replace = replace)
			return self.data.loc[idx]

		elif strategy == 'uniform-from-percentile':
			raise NotImplemented("We havent coded this yet.")
		
		elif strategy == 'uniform-over-similarity-ranks':
			
			if not self.metastore:
				self.metastore = self.build_metastore()

			data_distance_ranks = self.metastore['ranks_rounded']
	
			
			for i in range(100):
				j = i / 100.
				idx = np.where(data_distance_ranks==j)[0]
				new_samples = np.random.choice(idx, replace = True, size = samples_per_bin)
				samples.extend(self.data.loc[new_samples].values)
			return pd.DataFrame(samples, index = self.index, columns = self.feature_ids)


