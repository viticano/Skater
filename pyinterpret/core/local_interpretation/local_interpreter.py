import numpy as np
import pandas as pd
from  sklearn import metrics
from sklearn.linear_model import LinearRegression

from .base import BaseLocalInterpretation


class LocalInterpreter(BaseLocalInterpretation):
    def lime_ds(self, data_row, predict_fn, sample=False,
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

        if explainer_model is None:
            explainer_model = LinearRegression

        explainer_model = explainer_model()

        self._check_explainer_model_pre_train(explainer_model)
        predict_fn = self.build_annotated_model(predict_fn)

        kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        # data that has been sampled
        neighborhood = self.interpreter.data_set.generate_sample(strategy=sampling_strategy, sample=sample,
                                                                 n_samples_from_dataset=n_samples)
        self._check_neighborhood(neighborhood)

        distances = metrics.pairwise_distances(
            neighborhood,
            data_row.reshape(1, -1),
            metric=distance_metric).ravel()

        weights = kernel_fn(distances)
        predictions = predict_fn(neighborhood)
        explainer_model.fit(neighborhood, predictions, sample_weight=weights)
        self._check_explainer_model_post_train(explainer_model)

        return explainer_model.coef_

    def local_explainer(self, training_data, feature_names=None, categorical_features=None,
                        categorical_names=None, kernel_width=3, verbose=False, class_names=None,
                        feature_selection='auto', discretize_continuous=True):
        #import lime
        #import lime.lime_tabular
        #, training_data, feature_names=None, categorical_features=None, categorical_names=None,
        #kernel_width=3, verbose=False, class_names=None, feature_selection='auto', discretize_continuous=True

        #return lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names,
        #                                       class_names=class_names, discretize_continuous=True)
        pass

    def _check_explainer_model_pre_train(self, explainer_model):
        assert hasattr(explainer_model, 'fit'), "Model needs to have a fit method "

    def _check_explainer_model_post_train(self, explainer_model):
        assert hasattr(explainer_model, 'coef_'), "Model needs to have coefficients to explain "

    def _check_neighborhood(self, neighborhood):
        assert isinstance(neighborhood, (np.ndarray, pd.DataFrame))
