"""Partial Dependence class"""
from itertools import product, cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.axes._subplots import Axes as mpl_axes

from .base import BaseGlobalInterpretation
from ...util.static_types import StaticTypes
from ...util import exceptions
from ...util.kernels import flatten

COLORS = ['#328BD5', '#404B5A', '#3EB642', '#E04341', '#8665D0']
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (16, 7)
plt.style.use('ggplot')

class FeatureImportance(BaseGlobalInterpretation):
    """Contains methods for feature importance. Subclass of BaseGlobalInterpretation"""

    @staticmethod
    def _build_fresh_metadata_dict():
        return {
            'pdp_cols': {},
            'sd_col':'',
            'val_cols':[]
        }


    def feature_importance(self, modelinstance):

        """
        Computes partial_dependence of a set of variables. Essentially approximates
        the partial partial_dependence of the predict_fn with respect to the variables
        passed.

        Parameters:
        -----------
        feature_ids(list):
            the names/ids of the features for which we compute partial dependence.
            Note that the algorithm's complexity scales exponentially with additional
            features, so generally one should only look at one or two features at a
            time. These feature ids must be available in the class's associated DataSet.

            As of now, we only support looking at 1 or 2 features at a time.

        modelinstance: pyinterpret.model.model.Model subtype
            the machine learning model "prediction" function to explain, such that
            predictions = predict_fn(data).

            For instance:
            from pyinterpret.model import InMemoryModel
            from pyinterpret.core.explanations import Interpretation
            from sklearn.ensemble import RandomForestClassier
            rf = RandomForestClassier()
            rf.fit(X,y)


            model = InMemoryModel(rf, examples = X)
            interpreter = Interpretation()
            interpreter.load_data(X)
            interpreter.feature_importance.feature_importance(model)

            are acceptable use cases. Output types need to be 1D or 2D numpy arrays.

            Supports classification, multi-class classification, and regression.

        grid(numpy.ndarray):
            2 dimensional array on which we fix values of features. Note this is
            determined automatically if not given based on the percentiles of the
            dataset.

        grid_resolution(int):
            how many unique values to include in the grid. If the percentile range
            is 5% to 95%, then that range will be cut into <grid_resolution>
            equally size bins.

        grid_range(tuple):
            the percentile extrama to consider. 2 element tuple, increasing, bounded
            between 0 and 1.

        sample(Bool):
            Whether to sample from the original dataset.

        sampling_strategy(string):
            If sampling, which approach to take. See DataSet.generate_sample for
            details.

        n_samples(int):
            The number of samples to use from the original dataset. Note this is
            only active if sample = True and sampling strategy = 'uniform'. If
            using 'uniform-over-similarity-ranks', use samples per bin

        bin_count(int):
            The number of bins to use when using the similarity based sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.

        samples_per_bin(int):
            The number of samples to collect for each bin within the sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'. If using
            sampling_strategy = 'uniform', use n_samples.
            total samples = bin_count * samples per bin.

        """

        importances = {}
        original_predictions = modelinstance.predict(self.data_set.data.values)

        n = original_predictions.shape[0]

        # instead of copying the whole dataset, should we copy a column, change column values,
        # revert column back to copy?
        for feature_id in self.data_set.feature_ids:
            X_mutable = self.data_set.data.copy()
            samples = self.data_set.generate_column_sample(feature_id, n_samples=n, method='random-choice')
            feature_perturbations = X_mutable[feature_id].values - samples
            X_mutable[feature_id] = samples
            new_predictions = modelinstance.predict(X_mutable.values)
            changes_in_predictions = new_predictions - original_predictions
            importance = np.mean(np.std(changes_in_predictions, axis=0))
            importances[feature_id] = importance

        importances =  pd.Series(importances).sort_values()
        importances = importances / importances.sum()
        return importances

    def plot_feature_importance(self, predict_fn, ax=None):
        importances = self.feature_importance(predict_fn)

        if ax is None:
            f, ax = plt.subplots(1)
        else:
            f = ax.figure

        colors = cycle(COLORS)
        color = colors.next()
        importances.sort_values().plot(kind = 'barh',ax=ax, color=color)

