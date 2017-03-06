"""Local Interpreter class"""

import numpy as np
import pandas as pd
from  sklearn import metrics
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ...util.kernels import rbf_kernel


from .base import BaseLocalInterpretation


class LocalInterpreter(BaseLocalInterpretation):

    """Contains all methods for LIME style interpretations"""

    def lime_ds(self, data_row, predict_fn, similarity_method='local-affinity-scaling',
                sample=False, n_samples=5000,
                sampling_strategy='uniform-over-similarity-ranks',

                distance_metric='euclidean', kernel_width=None,
                explainer_model=LinearRegression):

        """For performing Local interpretable model explanation. Creates a sample of data
        around of point, passing these points through your prediction function. This
        gives a linear model of the full model centered around a point.

        Parameters
        ----------
            data_row(numpy.ndarray):
                The observation to explain.
            predict_fn(function):
                The model to explain
            similarity_method(string):
                The means by which similarities are computed. Currently supported
                options:
                    cosine-similarity:
                        generally gives more global coefficients
                    unscaled-kernel-substitution:
                        calculates euclidean distances, then passes these distances
                        through a gaussian kernel with hyperparameter kernel_width
                    scaled-kernel-substitution:
                        calculates euclidean distances on standard-scaled data,
                        then passes these distances through a gaussian kernel with
                        hyperparameter kernel_width. Not recommended
                    local-affinity-weighting:
                        calculates euclidean distances, then passes these through an
                        adaptive kernel function that weights according to local scales
                        of the given data_row and each point in the neighborhood.

            sample(Bool):
                Whether or not to sample the data.
            distance_metric(string):
                The measure by which we evaluate distane between a point in the sampled
                data and the data row to explain.
            kernel_width(numeric):
                We pass distances through a kernel function, this specifies the width.
                Smaller widths will give more weight to points neighboring the
                explained point. Defaults to proportional to dimensionality of dataset.
            explainer_model(scikit learn model):
                Model to use to explain the point. Defaults to Linear regression.

        """

        if kernel_width is None:
            kernel_width = np.sqrt(self.data_set.dim) * .75

        explainer_model = explainer_model()

        self._check_explainer_model_pre_train(explainer_model)
        predict_fn = self.build_annotated_model(predict_fn)

        # data that has been sampled
        neighborhood = self.interpreter.data_set.generate_sample(strategy=sampling_strategy,
                                                                 sample=sample,
                                                                 n_samples_from_dataset=n_samples)
        self._check_neighborhood(neighborhood)

        if similarity_method == 'cosine-similarity':
            weights = self.get_weights_from_cosine_similarity(neighborhood.values,
                                                              data_row)
        elif similarity_method == 'unscaled-kernel-substitution':
            weights = self.get_weights_via_kernel_subtitution(neighborhood,
                                                              data_row,
                                                              kernel_width,
                                                              distance_metric)
        elif similarity_method == 'scaled-kernel-substitution':
            weights = self.get_weights_kernel_tranformation_of_scaled_euclidean_distance(neighborhood,
                                                                                         data_row,
                                                                                         kernel_width,
                                                                                         distance_metric)
        elif similarity_method == 'local-affinity-scaling':
            weights = self.get_weights_via_local_scaling_weights(neighborhood,
                                                                 data_row,
                                                                 distance_metric)
        else:
            raise ValueError("{} is not a valid similarity method".format(similarity_method))


        weights = kernel_fn(distances)
        predictions = predict_fn(neighborhood)
        explainer_model.fit(neighborhood, predictions, sample_weight=weights)
        self._check_explainer_model_post_train(explainer_model)

        #results = pd.DataFrame(explainer_model.coef_, self.data_set.feature_ids, index = 'coef')
        return explainer_model.coef_


    @staticmethod
    def get_weights_kernel_tranformation_of_scaled_euclidean_distance(neighborhood,
                                                                      point,
                                                                      kernel_width,
                                                                      distance_metric):
        """This method scales data, then computes distance metric, then passes
        through kernel"""

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled_neighborhood = scaler.fit_transform(neighborhood)
        scaled_point = scaler.transform(point)

        distances = metrics.pairwise_distances(
            scaled_neighborhood,
            scaled_point.reshape(1, -1),
            metric=distance_metric) \
            .ravel()
        weights = rbf_kernel(distances, kernel_width=kernel_width)
        return weights

    @staticmethod
    def get_weights_from_cosine_similarity(neighborhood, point):
        """This method computes absolute values of cosine similarities"""
        similarities = metrics.pairwise.cosine_similarity(neighborhood,
                                                          point.reshape(1, -1)).ravel()
        return abs(similarities)

    @staticmethod
    def get_weights_via_kernel_subtitution(neighborhood, point, kernel_width,
                                           distance_metric):
        """Computes distances, passes through gaussian kernel"""
        kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        distances = metrics.pairwise_distances(
            neighborhood,
            point.reshape(1, -1),
            metric=distance_metric) \
            .ravel()
        weights = kernel_fn(distances)
        return weights

    @staticmethod
    def get_weights_via_local_scaling_weights(neighborhood, point, distance_metric):
        """Computes distances, passes through kernel that weights distance by
        pointwise local densities"""
        distances = metrics.pairwise_distances(
            neighborhood,
            point.reshape(1, -1),
            metric=distance_metric) \
            .ravel()

        combined = np.concatenate((neighborhood, point[:, np.newaxis].T), axis=0)
        nearest_neighbor = NearestNeighbors(7)
        nearest_neighbor.fit(combined)
        local_distances, indices = nearest_neighbor.kneighbors(combined)
        local_distances = local_distances[:, 6]
        point_sigma = local_distances[-1:]
        population_sigmas = local_distances[:-1]

        affinities = np.exp((-1 * (distances ** 2)) / (point_sigma[0] * population_sigmas))\
            .reshape(-1)
        return affinities


    @staticmethod
    def _check_explainer_model_pre_train(explainer_model):
        assert hasattr(explainer_model, 'fit'), "Model needs to have a fit method "

    @staticmethod
    def _check_explainer_model_post_train(explainer_model):
        assert hasattr(explainer_model, 'coef_'), "Model needs to have coefficients to explain "

    @staticmethod
    def _check_neighborhood(neighborhood):
        assert isinstance(neighborhood, (np.ndarray, pd.DataFrame))


    def local_explainer(self, training_data, feature_names=None, categorical_features=None,
                        categorical_names=None, kernel_width=3, verbose=False, class_names=None,
                        feature_selection='auto', discretize_continuous=True):
        """Uses the lime package for explanations

        import lime
        import lime.lime_tabular
        return lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names,
                                               class_names=class_names, discretize_continuous=True)
        """
        pass

    def _check_explainer_model_pre_train(self, explainer_model):
        assert hasattr(explainer_model, 'fit'), "Model needs to have a fit method "

    def _check_explainer_model_post_train(self, explainer_model):
        assert hasattr(explainer_model, 'coef_'), "Model needs to have coefficients to explain "

    def _check_neighborhood(self, neighborhood):
        assert isinstance(neighborhood, (np.ndarray, pd.DataFrame))
