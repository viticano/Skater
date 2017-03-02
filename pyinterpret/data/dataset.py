import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


class DataSet(object):
    def __init__(self, data, feature_names=None, index=None):
        """
        The abtraction around using, accessing, sampling data for interpretation purposes.

        Parameters
        ----------
            data: 1D or 2D numpy array.
            feature_names: iterable of feature names
            index: iterable of row names


        """
        assert isinstance(data, (np.ndarray, pd.DataFrame)), 'Data needs to be a numpy array'

        ndim = len(data.shape)

        if ndim == 1:
            data = data[:, np.newaxis]

        elif ndim >= 3:
            raise ValueError("Data needs to be 1 or 2 dimensions, yours is {}".format(ndim))

        self.n, self.dim = data.shape

        if isinstance(data, pd.DataFrame):
            if not feature_names:
                feature_names = list(data.columns.values)
            if not index:
                index = list(data.index.values)
            self.feature_ids = feature_names
            self.index = index
            self.data = pd.DataFrame(data, columns=self.feature_ids, index=self.index)


        elif isinstance(data, np.ndarray):
            if not feature_names:
                feature_names = range(self.dim)
            if not index:
                index = range(self.n)
            self.feature_ids = feature_names
            self.index = index
            self.data = pd.DataFrame(data, columns=self.feature_ids, index=self.index)

        else:
            raise ValueError("Currently we only support pandas dataframes and numpy arrays"
                             "If you would like support for additional data structures let us "
                             "know!")

        self.metastore = None

    def generate_grid(self, feature_ids, grid_resolution=100, grid_range=(.05, .95)):
        """
        Generates a grid of values on which to compute pdp. For each feature xi, for value
        yj of xi, we will fix xi = yj for every observation in X.

        Parameters
        ----------
            feature_ids(list):
                Feature names for which we'll generate a grid. Must be contained
                by self.feature_ids

            grid_resolution(int):
                The number of unique values to choose for each feature.

            grid_range(tuple):
                The percentile bounds of the grid. For instance, (.05, .95) corresponds to
                the 5th and 95th percentiles, respectively.

        Returns
        ----------
        grid(numpy.ndarray): 	There are as many rows as there are feature_ids
                                There are as many columns as specified by grid_resolution
        """

        grid_range_warning = "Grid range values must be between 0 and 1"
        assert all(i >= 0 and i <= 1 for i in grid_range), grid_range_warning

        grid_resolution_warning = "Grid resolute must be a positive integer"
        assert isinstance(grid_resolution, int) and grid_resolution > 0, grid_resolution_warning

        feature_id_warning = "Must pass in feature ids contained in DataSet.feature_ids"
        assert all(feature_id in self.feature_ids for feature_id in feature_ids), feature_id_warning

        grid_range = map(lambda x: x * 100, grid_range)
        bins = np.linspace(*grid_range, num=grid_resolution)
        grid = []
        for feature_id in feature_ids:
            vals = np.percentile(self[feature_id], bins)
            grid.append(vals)
        return np.array(grid)

    def _build_metastore(self, bin_count):

        n = self.data.shape[0]

        medians = np.median(self.data.values, axis=0).reshape(1, self.dim)

        # how far each data point is from the global median
        dists = cosine_distances(self.data.values, Y=medians).reshape(-1)

        # the percentile distance of each datapoint to the global median
        # dist_percentiles = map(lambda i: int(stats.percentileofscore(dists, i)), dists)

        ranks = pd.Series(dists).rank().values
        round_to = n / float(bin_count)
        rounder_func = lambda x: int(round_to * round(float(x) / round_to))
        ranks_rounded = map(rounder_func, ranks)

        ranks_rounded = np.array(map(lambda x: round(x, 2), ranks / ranks.max()))
        return {
            'median': medians,
            'dists': dists,
            'n': n,
            # 'dist_percentiles': dist_percentiles,
            'ranks': ranks,
            'ranks_rounded': ranks_rounded,
            'round_to': round_to
        }

    def __getitem__(self, key):
        assert key in self.feature_ids, "The key {} is not the set of feature_ids {}".format(*[key, self.feature_ids])
        return self.data.__getitem__(key)

    def __setitem__(self, key, newval):
        self.data.__setitem__(key, newval)

    def generate_sample(self, sample=True, strategy='random-choice', n_samples_from_dataset=1000,
                        replace=True, samples_per_bin=10, bin_count=50):
        '''
        Method for generating data from the dataset.

        Parameters:
        -----------
            sample(Bool):
                If False, we'll take the full dataset, otherwise we'll sample.

            n_samples_from_dataset(int):
                Specifies the number of samples to return. Only implemented
                if strategy is "random-choice".

            replace(Bool):
                Bool for sampling with or without replacement

            samples_per_bin(int):
                If strategy is uniform-over-similarity-ranks, then this is the number
                of samples to take from each discrete rank.


        '''

        if not sample:
            return self.data

        if strategy == 'random-choice':
            idx = np.random.choice(self.index, size=n_samples_from_dataset, replace=replace)
            values = self.data.loc[idx].values
            return pd.DataFrame(values, columns=self.feature_ids)

        elif strategy == 'uniform-from-percentile':
            raise NotImplemented("We havent coded this yet.")

        elif strategy == 'uniform-over-similarity-ranks':

            metastore = self._build_metastore(bin_count)

            total_samples = bin_count * samples_per_bin

            data_distance_ranks = metastore['ranks_rounded']
            round_to = metastore['round_to']
            n = metastore['n']

            samples = []
            for i in range(bin_count):
                j = (i * round_to) / n
                idx = np.where(data_distance_ranks == j)[0]
                if idx.any():
                    new_samples = np.random.choice(idx, replace=True, size=samples_per_bin)
                    samples.extend(self.data.loc[new_samples].values)
            return pd.DataFrame(samples, columns=self.feature_ids)
