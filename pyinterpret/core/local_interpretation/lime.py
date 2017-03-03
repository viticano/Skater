import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from .base import BaseLocalInterpretation


class Lime(BaseLocalInterpretation):
    def lime_ds(self, data_row, predict_fn, similarity_method = 'cosine-similarity', sample=False,
                n_samples=5000, sampling_strategy='uniform-over-similarity-ranks',
                distance_metric='euclidean', kernel_width=None,
                explainer_model=None):

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

        predict_fn = self.build_annotated_model(predict_fn)
        kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        # data that has been sampled
        neighborhood = self.interpreter.data_set.generate_sample(strategy=sampling_strategy, sample=sample,
                                                                 n_samples_from_dataset=n_samples)

        self._check_neighborhood(neighborhood)

        if similarity_method == 'cosine-similarity':
            weights = self.get_weights_from_cosine_similarity(neighborhood.values, data_row)
        elif similarity_method == 'distance-then-transform':
            weights = self.get_weights_kernel_tranformation_of_euclidean_distance(neighborhood, data_row, kernel_width, distance_metric)

        predictions = predict_fn(neighborhood)

        assert np.isfinite(weights).all(), "weights are nan or inf"
        assert np.isfinite(predictions).all(), "predictions are nan or inf"
        assert np.isfinite(neighborhood.values).all(), "neighborhood are nan or inf"


        explainer_model.fit(neighborhood.values, predictions, sample_weight=weights)
        self._check_explainer_model_post_train(explainer_model)
        assert (explainer_model.coef_ != 0.).any(), "All coefs are 0"

        return explainer_model.coef_

    def lime(self):
        pass

    def get_weights_from_cosine_similarity(self, neighborhood, point):
        similarities = sklearn.metrics.pairwise.cosine_similarity(neighborhood, point.reshape(1, -1)).ravel()
        return abs(similarities)

    def get_weights_kernel_tranformation_of_euclidean_distance(self, neighborhood, point, kernel_width, distance_metric):
        kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        distances = sklearn.metrics.pairwise_distances(
            neighborhood,
            point.reshape(1, -1),
            metric=distance_metric) \
            .ravel()
        weights = kernel_fn(distances)
        return weights

    @staticmethod
    def _check_explainer_model_pre_train(explainer_model):
        assert hasattr(explainer_model, 'fit'), "Model needs to have a fit method "

    @staticmethod
    def _check_explainer_model_post_train(explainer_model):
        assert hasattr(explainer_model, 'coef_'), "Model needs to have coefficients to explain "

    @staticmethod
    def _check_neighborhood(neighborhood):
        assert isinstance(neighborhood, (np.ndarray, pd.DataFrame))
