from itertools import product

import numpy as np
import pandas as pd

from .base import BaseGlobalInterpretation


class PartialDependence(BaseGlobalInterpretation):
    def partial_dependence(self, feature_ids, predict_fn, grid=None, grid_resolution=100,
                           grid_range=(0.03, 0.97), sample=False, sampling_strategy='uniform-over-similarity-ranks',
                           n_samples=5000, bin_count=50, samples_per_bin=10):

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

            As of now, we only support looking at 1 or 2 features at a time.

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

        n_samples(int):
            The number of samples to use from the original dataset. Note this is only active if sample = True
            and sampling strategy = 'uniform'. If using 'uniform-over-similarity-ranks', use samples per bin

        bin_count(int):
            The number of bins to use when using the similarity based sampler. Note this is only active if
            sample = True and sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.

        samples_per_bin(int):
            The number of samples to collect for each bin within the sampler. Note this is only active if
            sample = True and sampling_strategy = 'uniform-over-similarity-ranks'. If using sampling_strategy = 'uniform',
            use n_samples. total samples = bin_count * samples per bin.



        '''

        predict_fn = self.build_annotated_model(predict_fn)
        assert all(feature_id in self.data_set.feature_ids for feature_id in feature_ids), "Pass in a valid ID"
        assert len(feature_ids) < 3, "Pass in at most 2 features for pdp. If you have a use case where you'd " \
                                     "like to look at 3 simultaneously, please let us know."

        # if you dont pass a grid, build one.
        if not grid:
            grid = self.data_set.generate_grid(feature_ids,
                                               grid_resolution=grid_resolution,
                                               grid_range=grid_range)

        # make sure data_set module is giving us correct data structure
        self._check_grid(grid, feature_ids, grid_resolution)

        # generate data
        X = self.data_set.generate_sample(strategy=sampling_strategy,
                                          sample=sample,
                                          n_samples_from_dataset=n_samples,
                                          samples_per_bin=samples_per_bin,
                                          bin_count=bin_count)

        # make sure data_set module is giving us correct data structure
        self._check_X(X)

        n_features = len(feature_ids)

        # will store [featurename: {val1: {mean:<>, std:<>}, etc...}]

        grid_expanded = np.array(list(product(*grid)))

        id_grid = np.array([range(grid_resolution) for _ in range(n_features)])
        id_grid_expanded = np.array(list(product(*id_grid)))

        # pandas dataframe
        X_mutable = X.copy()

        means = np.zeros([grid_resolution for i in range(n_features)])
        sds = np.zeros([grid_resolution for i in range(n_features)])

        for i in range(grid_expanded.shape[0]):
            new_row = grid_expanded[i]
            row_id = id_grid_expanded[i]
            for feature_idx, feature_id in enumerate(feature_ids):
                X_mutable[feature_id] = new_row[feature_idx]

            predictions = predict_fn(X_mutable.values)
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)

            means[row_id] = mean_prediction
            sds[row_id] = std_prediction

        pdp = {
            'features': feature_ids,
            'means': means,
            'sds': sds,
            'vals': grid_expanded
        }
        return pdp

    def partial_dependency_sklearn(self):
        pass

    @staticmethod
    def _check_grid(grid, feature_ids, grid_resolution):
        assert isinstance(grid, np.ndarray), "Grid is not a numpy array"
        assert len(grid.shape) == 2, "Grid is not 2D"
        assert len(feature_ids) == grid.shape[0], "There should be as many rows in grid as there are features."
        assert grid_resolution == grid.shape[1], "There should be as many columns in grid as grid_resolution."

    @staticmethod
    def _check_X(X):
        assert isinstance(X, pd.DataFrame)
