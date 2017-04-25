"""DataSet object"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from ..util.logger import build_logger
from ..util import exceptions

__all__ = ['DataManager']

class DataManager(object):
    """Module for passing around data to interpretation objects"""

    def __init__(self, data, feature_names=None, index=None, log_level=30):
        """
        The abstraction around using, accessing, sampling data for interpretation purposes.
        Used by interpretation objects to grab data, collect samples, and handle
        feature names and row indices.

        Parameters
        ----------
            data: 1D/2D numpy array, or pandas DataFrame
                raw data
            feature_names: iterable of feature names
                Optional keyword containing names of features.
            index: iterable of row names
                Optional keyword containing names of indexes (rows).

        """

        # create logger
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)

        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            err_msg = 'Invalid Data: expected data to be a numpy array or pandas dataframe but got ' \
                      '{}'.format(type(data))
            raise(exceptions.DataSetError(err_msg))

        ndim = len(data.shape)
        self.logger.debug("__init__ data.shape: {}".format(data.shape))

        if ndim == 1:
            data = data[:, np.newaxis]

        elif ndim >= 3:
            err_msg = "Invalid Data: expected data to be 1 or 2 dimensions, " \
                      "Data.shape: {}".format(ndim)
            raise(exceptions.DataSetError(err_msg))

        self.n_rows, self.dim = data.shape
        self.logger.debug("after transform data.shape: {}".format(data.shape))

        if isinstance(data, pd.DataFrame):
            if feature_names is None:
                feature_names = list(data.columns.values)
            if not index:
                index = list(data.index.values)
            self.feature_ids = feature_names
            self.index = index

        elif isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = range(self.dim)
            if not index:
                index = range(self.n_rows)
            self.feature_ids = feature_names
            self.index = index

        else:
            raise(ValueError("Invalid: currently we only support pandas dataframes and numpy arrays"
                             "If you would like support for additional data structures let us "
                             "know!"))

        self.data = pd.DataFrame(data, columns=self.feature_ids, index=self.index)
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

        if not all(i >= 0 and i <= 1 for i in grid_range):
            err_msg = "Grid range values must be between 0 and 1 but got:" \
                                 "{}".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))

        if not isinstance(grid_resolution, int) and grid_resolution > 0:
            err_msg = "Grid resolution {} is not a positive integer".format(grid_resolution)
            raise(exceptions.MalformedGridRangeError(err_msg))

        if not all(feature_id in self.feature_ids for feature_id in feature_ids):
            missing_features = []
            for feature_id in feature_ids:
                if feature_id not in self.feature_ids:
                    missing_features.append(feature_id)
            err_msg = "Feature ids {} not found in DataManager.feature_ids".format(missing_features)
            raise(KeyError(err_msg))

        grid_range = [x * 100 for x in grid_range]
        bins = np.linspace(*grid_range, num=grid_resolution)
        grid = []
        for feature_id in feature_ids:
            data = self[feature_id]
            uniques = np.unique(data)
            if len(uniques) ==2:
                vals = uniques.copy()
            else:
                vals = np.unique(np.percentile(self[feature_id], bins))
            grid.append(vals)
        grid = np.array(grid)
        grid_shape = [(1, i) for i in [row.shape[0] for row in grid]]
        self.logger.info('Generated grid of shape {}'.format(grid_shape))
        return grid

    def _build_metastore(self, bin_count):

        n_rows = self.data.shape[0]
        medians = np.median(self.data.values, axis=0).reshape(1, self.dim)

        # how far each data point is from the global median
        dists = cosine_distances(self.data.values, Y=medians).reshape(-1)

        # the percentile distance of each datapoint to the global median
        # dist_percentiles = map(lambda i: int(stats.percentileofscore(dists, i)), dists)

        bins = np.linspace(0, 100, num=bin_count + 1)
        unique_dists = np.unique(dists)

        if len(unique_dists) > 1:
            ranks_rounded = pd.qcut(dists, bins / 100, labels=False)
            unique_ranks = np.unique(ranks_rounded)
        else:
            ranks_rounded = np.ones(n_rows)
            unique_ranks = np.ones(1)
        return {
            'median': medians,
            'dists': dists,
            'n_rows': n_rows,
            'unique_ranks': unique_ranks,
            'ranks_rounded': ranks_rounded
        }

    def __getitem__(self, key):

        if not key in self.feature_ids:
            err_msg = "The key {} is not the set of feature_ids {}".format(*[key, self.feature_ids])
            raise(KeyError(err_msg))
        return self.data.__getitem__(key)

    def __setitem__(self, key, newval):
        self.data.__setitem__(key, newval)

    def generate_sample(self, sample=True, strategy='random-choice', n_samples_from_dataset=1000,
                        replace=True, samples_per_bin=10, bin_count=50):
        """
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


        """

        arg_dict = {
            'sample':sample,
            'strategy':strategy,
            'n_samples_from_dataset':n_samples_from_dataset,
            'replace':replace,
            'samples_per_bin':samples_per_bin,
            'bin_count':bin_count
        }
        self.logger.debug("Generating sample with args:\n {}".format(arg_dict))

        if not sample:
            return self.data

        if strategy == 'random-choice':
            idx = np.random.choice(self.index, size=n_samples_from_dataset, replace=replace)
            values = self.data.loc[idx].values
            return pd.DataFrame(values, columns=self.feature_ids)

        elif strategy == 'uniform-from-percentile':
            raise(NotImplementedError("We havent coded this yet."))

        elif strategy == 'uniform-over-similarity-ranks':
            metastore = self._build_metastore(bin_count)
            data_distance_ranks = metastore['ranks_rounded']
            n_rows = metastore['n_rows']
            unique_ranks = metastore['unique_ranks']

            samples = []

            for rank in unique_ranks:
                idx = np.where(data_distance_ranks == rank)[0]
                if idx.any():
                    new_samples = np.random.choice(idx, replace=True, size=samples_per_bin)
                    samples.extend(self.data.loc[new_samples].values)
            return pd.DataFrame(samples, columns=self.feature_ids)

    def generate_column_sample(self, feature_id, n_samples=None, method='random-choice'):
        """Sample a single feature from the data set.

        Parameters
        ----------

        feature_id: hashable
            name of the feature to sample. If no feature names were passed, then
            the features are accessible via their column index.

        n_samples: int
            the number of samples to generate

        method: str
            the sampling method. Currently only random-choice is implemented.


        """
        if method == 'random-choice':
            return self._generate_column_sample_random_choice(feature_id, n_samples=n_samples)
        else:
            raise(NotImplementedError("Currenly we only support random-choice for column \
                                       level sampling "))

    def _generate_column_sample_random_choice(self, feature_id, n_samples=None):
        return np.random.choice(self.__getitem__(feature_id), size=n_samples)

    def _generate_column_sample_stratified(self, feature_id, n_samples=None):
        """
        Tries to capture all relevant regions of space, relative to how many samples are allowed.
        :param feature_id:
        :param n_samples:
        :return:
        """
        pass

    def _generate_column_sample_modeled(self, feature_id, n_samples=None):
        pass
