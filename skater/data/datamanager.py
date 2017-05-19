"""DataSet object"""
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from ..util.logger import build_logger
from ..util import exceptions
from ..util.static_types import StaticTypes
from ..util.dataops import add_column_numpy_array, allocate_samples_to_bins, flatten

__all__ = ['DataManager']


class DataManager(object):
    """Module for passing around data to interpretation objects"""

    # Todo: we can probably remove some of the keys from data_info, and have properties
    # executed as pure functions for easy to access metadata, such as n_rows, etc

    _n_rows = 'n_rows'
    _dim = 'dim'
    _feature_info = 'feature_info'
    _dtypes = 'dtypes'

    __attribute_keys__ = [_n_rows, _dim, _feature_info, _dtypes]
    __datatypes__ = (pd.DataFrame, np.ndarray)


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

        if not isinstance(data, self.__datatypes__):
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

        self.data = data
        self.data_type = type(data)
        self.metastore = None

        self.logger.debug("after transform data.shape: {}".format(self.data.shape))

        if isinstance(self.data, pd.DataFrame):
            if feature_names is None:
                feature_names = list(self.data.columns.values)
            if not index:
                index = list(self.data.index.values)
            self.feature_ids = list(feature_names)
            self.index = index

        elif isinstance(self.data, np.ndarray):
            if feature_names is None:
                feature_names = range(self.data.shape[1])
            if not index:
                index = range(self.data.shape[0])
            self.feature_ids = list(feature_names)
            self.index = index

        else:
            raise(ValueError("Invalid: currently we only support pandas dataframes and numpy arrays"
                             "If you would like support for additional data structures let us "
                             "know!"))

        self.data_info = {attr: None for attr in self.__attribute_keys__}


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
        bins = np.linspace(*grid_range, num=grid_resolution).tolist()
        grid = []
        for feature_id in feature_ids:
            data = self[feature_id]
            info = self.feature_info[feature_id]
            # if a feature is categorical (non numeric) or
            # has a small number of unique values, we'll just
            # supply unique values for the grid
            if info['unique'] < grid_resolution or info['numeric'] is False:
                vals = np.unique(data)
            else:
                vals = np.unique(np.percentile(data, bins))
            grid.append(vals)
        grid = np.array(grid)
        grid_shape = [(1, i) for i in [row.shape[0] for row in grid]]
        self.logger.info('Generated grid of shape {}'.format(grid_shape))
        return grid


    def sync_metadata(self):
        self.data_info[self._n_rows] = self._calculate_n_rows()
        self.data_info[self._dim] = self._calculate_n_rows()
        self.data_info[self._dtypes] = self._calculate_dtypes()
        self.data_info[self._feature_info] = self._calculate_feature_info()


    def _calculate_n_rows(self):
        return self.data.shape[0]


    def _calculate_dim(self):
        return self.data.shape[1]


    def _calculate_dtypes(self):
        return pd.DataFrame(self.data, columns=self.feature_ids, index=self.index).dtypes


    def _calculate_feature_info(self):
        feature_info = {}
        for feature in self.feature_ids:
            x = self[feature]
            samples = self.generate_column_sample(feature, n_samples=10)
            samples_are_numeric = map(StaticTypes.data_types.is_numeric, samples)
            is_numeric = all(samples_are_numeric)
            feature_info[feature] = {
                'type': self.dtypes.loc[feature],
                'unique': len(np.unique(x)),
                'numeric': is_numeric
            }
        return feature_info


    @property
    def n_rows(self):
        if self.data_info[self._n_rows] is None:
            self.data_info[self._n_rows] = self._calculate_n_rows()
        return self.data_info[self._n_rows]


    @property
    def dim(self):
        if self.data_info[self._dim] is None:
            self.data_info[self._dim] = self._calculate_dim()
        return self.data_info[self._dim]


    @property
    def dtypes(self):
        if self.data_info[self._dtypes] is None:
            self.data_info[self._dtypes] = self._calculate_dtypes()
        return self.data_info[self._dtypes]


    @property
    def feature_info(self):
        if self.data_info[self._feature_info] is None:
            self.data_info[self._feature_info] = self._calculate_feature_info()
        return self.data_info[self._feature_info]


    def _build_metastore(self, bin_count):

        medians = np.median(np.array(self.data), axis=0).reshape(1, self.dim)

        # how far each data point is from the global median
        dists = cosine_distances(np.array(self.data), Y=medians).reshape(-1)

        # the percentile distance of each datapoint to the global median
        # dist_percentiles = map(lambda i: int(stats.percentileofscore(dists, i)), dists)

        bins = np.linspace(0, 100, num=bin_count + 1)
        unique_dists = np.unique(dists)

        if len(unique_dists) > 1:
            ranks_rounded = pd.qcut(dists, bins / 100, labels=False)
            unique_ranks = np.unique(ranks_rounded)
        else:
            ranks_rounded = np.ones(self.n_rows)
            unique_ranks = np.ones(1)
        return {
            'median': medians,
            'dists': dists,
            'n_rows': self.n_rows,
            'unique_ranks': unique_ranks,
            'ranks_rounded': ranks_rounded
        }


    def __getitem__(self, key):
        if issubclass(self.data_type, pd.DataFrame):
            return self.__getitem_pandas__(key)
        elif issubclass(self.data_type, np.ndarray):
            return self.__getitem_ndarray__(key)
        else:
            raise ValueError("Can't get item for data of type {}".format(self.data_type))


    def __setitem__(self, key, newval):
        if issubclass(self.data_type, pd.DataFrame):
            self.__setcolumn_pandas__(key, newval)
        elif issubclass(self.data_type, np.ndarray):
            self.__setcolumn_ndarray__(key, newval)
        else:
            raise ValueError("Can't set item for data of type {}".format(self.data_type))
        self.sync_metadata()


    def __getrows__(self, idx):
        if self.data_type == pd.DataFrame:
            return self.__getrows_pandas__(idx)
        elif self.data_type == np.ndarray:
            return self.__getrows_ndarray__(idx)
        else:
            raise ValueError("Can't get rows for data of type {}".format(self.data_type))


    def __getitem_pandas__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        return self.data[i]


    def __getitem_ndarray__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if StaticTypes.data_types.is_string(i) or StaticTypes.data_types.is_numeric(i):
            idx = self.feature_ids.index(i)
        elif hasattr(i, '__iter__'):
            idx = [self.feature_ids.index(j) for j in i]
        else:
            raise(ValueError("Unrecongized index type: {}. This should not happen, "
                             "submit a issue here: "
                             "https://github.com/datascienceinc/Skater/issues"
                             .format(type(i))))
        return self.data[:, idx]


    def __setcolumn_pandas__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        self.data[i] = newval


    def __setcolumn_ndarray__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if i in self.feature_ids:
            idx = self.feature_ids.index(i)
            self.data[:, idx] = newval
        else:
            self.data = add_column_numpy_array(self.data, newval)
            self.feature_ids.append(i)


    def __getrows_pandas__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        return self.data.loc[idx]


    def __getrows_ndarray__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        idx = [self.index.index(i) for i in idx]
        return self.data[idx]


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
            'sample': sample,
            'strategy': strategy,
            'n_samples_from_dataset': n_samples_from_dataset,
            'replace': replace,
            'samples_per_bin': samples_per_bin,
            'bin_count': bin_count
        }
        self.logger.debug("Generating sample with args:\n {}".format(arg_dict))

        if not sample:
            return self.data

        if strategy == 'random-choice':
            idx = np.random.choice(self.index, size=n_samples_from_dataset, replace=replace)
            values = self.__getrows__(idx)
            return values

        elif strategy == 'uniform-from-percentile':
            raise(NotImplementedError("We havent coded this yet."))

        elif strategy == 'uniform-over-similarity-ranks':
            metastore = self._build_metastore(bin_count)
            data_distance_ranks = metastore['ranks_rounded']
            unique_ranks = metastore['unique_ranks']

            samples = []

            for rank in unique_ranks:
                idx = np.where(data_distance_ranks == rank)[0]
                if idx.any():
                    new_samples_idx = np.random.choice(idx, replace=True, size=samples_per_bin)
                    new_samples = self.__getrows__(new_samples_idx)
                    samples.extend(new_samples)
            if self.data_type == pd.DataFrame:
                return pd.DataFrame(samples, columns=self.feature_ids)
            elif self.data_type == np.ndarray:
                return np.array(samples)
            else:
                self.logger.warn("Type {} not in explictely supported. "
                                 "Returning sample as list".format(self.data_type))
                return samples

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
        elif method == 'stratified':
            return self._generate_column_sample_stratified(feature_id, n_samples=n_samples)
        else:
            raise(NotImplementedError("Currenly we only support random-choice, stratified for "
                                      "column level sampling, not {} ".format(method)))

    def _generate_column_sample_random_choice(self, feature_id, n_samples=None):
        return np.random.choice(self[feature_id], size=n_samples)

    def _generate_column_sample_stratified(self, feature_id, n_samples=None, n_bins=100):
        """
        Tries to capture all relevant regions of space, relative to how many samples are allowed.
        Parameters:
        ----------
        feature_id:
        n_samples:

        Returns:
        ---------
        samples
        """
        if not self.data_info['feature_info'][feature_id]['numeric']:
            raise exceptions.DataSetError("Stratified sampling is currently "
                                          "supported for numeric features only.")

        bin_count, samples_per_bin = allocate_samples_to_bins(n_samples, ideal_bin_count=n_bins)
        percentiles = [100 * (i / bin_count) for i in range(bin_count + 1)]

        bins = list(np.percentile(self[feature_id], percentiles))
        sample_windows = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        samples = []
        for window, n in zip(sample_windows, samples_per_bin):
            samples.append(np.random.uniform(window[0], window[1], size=int(n)).tolist())

        return np.array(flatten(samples))

    def _generate_column_sample_modeled(self, feature_id, n_samples=None):
        pass
