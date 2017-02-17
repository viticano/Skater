import numpy as np

class DataSet(object):
	def __init__(self, data, n_percentiles = 100):
		self.n, self.dim = data.shape
		self.data = data
		self.n_percentiles = n_percentiles
		self.bins = (np.array(range(n_percentiles)) / float(n_percentiles)) * 100
		self.percentiles = np.array(map(lambda i: np.percentile(self.data, i, axis=0), self.bins))
		
		#first = (self.percentiles[1] - self.percentiles[0]).reshape(1, self.dim)
		last = (self.percentiles[-1] - self.percentiles[-2]).reshape(1, self.dim)
		self._augment = np.concatenate((self.percentiles.copy(), last))[1:]
		


	def generate_sample(self, samples_per_bin = 10):
		samples = []
		for i in range(self.n_percentiles):
			bins = np.array(zip(self.percentiles,self._augment)[i]).T
			samples.append(np.array(map(lambda (start, stop): np.random.uniform(start, stop, size=samples_per_bin), bins)).T)
		return np.array(samples).reshape(samples_per_bin * self.n_percentiles, self.dim)