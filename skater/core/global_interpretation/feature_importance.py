"""Feature Importance class"""
from itertools import cycle
import numpy as np
import pandas as pd

from ...data import DataManager
from .base import BaseGlobalInterpretation
from ...util.plotting import COLORS
from ...util.exceptions import *
from ...model.base import ModelType
from ...util.dataops import divide_zerosafe


class FeatureImportance(BaseGlobalInterpretation):
    """Contains methods for feature importance. Subclass of BaseGlobalInterpretation.

    """

    def feature_importance(self, model_instance, ascending=True, filter_classes=None):

        """
        Computes feature importance of all features related to a model instance.
        Supports classification, multi-class classification, and regression.

        Wei, Pengfei, Zhenzhou Lu, and Jingwen Song.
        "Variable Importance Analysis: A Comprehensive Review".
        Reliability Engineering & System Safety 142 (2015): 399-432.

        Parameters
        ----------
        model_instance: skater.model.model.Model subtype
            the machine learning model "prediction" function to explain, such that
            predictions = predict_fn(data).
        ascending: boolean, default True
            Helps with ordering Ascending vs Descending
        filter_classes: array type
            The classes to run partial dependence on. Default None invokes all classes.
            Only used in classification models.

        Returns
        -------
        importances : Sorted Series


        Examples
        --------
            >>> from skater.model import InMemoryModel
            >>> from skater.core.explanations import Interpretation
            >>> from sklearn.ensemble import RandomForestClassier
            >>> rf = RandomForestClassier()
            >>> rf.fit(X,y)
            >>> model = InMemoryModel(rf, examples = X)
            >>> interpreter = Interpretation()
            >>> interpreter.load_data(X)
            >>> interpreter.feature_importance.feature_importance(model)
        """

        if filter_classes:
            err_msg = "members of filter classes must be" \
                      "members of model_instance.classes." \
                      "Expected members of: {0}\n" \
                      "got: {1}".format(model_instance.target_names,
                                        filter_classes)
            assert all([i in model_instance.target_names for i in filter_classes]), err_msg

        importances = {}
        original_predictions = model_instance.predict_subset_classes(self.data_set.data, filter_classes)

        n = original_predictions.shape[0]

        copy_of_data_set = DataManager(self.data_set.data.copy(),
                                       feature_names=self.data_set.feature_ids,
                                       index=self.data_set.index)

        for feature_id in self.data_set.feature_ids:
            # collect perturbations
            if self.data_set.feature_info[feature_id]['numeric']:
                samples = self.data_set.generate_column_sample(feature_id, n_samples=n, method='stratified')
            else:
                samples = self.data_set.generate_column_sample(feature_id, n_samples=n, method='random-choice')
            copy_of_data_set[feature_id] = samples

            # predict based on perturbed values
            new_predictions = model_instance.predict_subset_classes(copy_of_data_set.data, filter_classes)

            importance = self.compute_importance(new_predictions,
                                                 original_predictions,
                                                 self.data_set[feature_id],
                                                 samples)
            importances[feature_id] = importance

            # reset copy
            copy_of_data_set[feature_id] = self.data_set[feature_id]

        importances = pd.Series(importances).sort_values(ascending=ascending)

        if not importances.sum() > 0:
            self.interpreter.logger.debug("Importances that caused a bug: {}".format(importances))
            raise(FeatureImportanceError("Something went wrong. Importances do not sum to a positive value"
                                         "This could be due to:"
                                         "1) 0 or infinite divisions"
                                         "2) perturbed values == original values"
                                         "3) feature is a constant"
                                         ""
                                         "Please submit an issue here:"
                                         "https://github.com/datascienceinc/Skater/issues"))

        importances = divide_zerosafe(importances, (np.ones(importances.shape[0]) * importances.sum()))
        return importances


    def plot_feature_importance(self, predict_fn, filter_classes=None, ascending=True, ax=None):
        """Computes feature importance of all features related to a model instance,
        then plots the results. Supports classification, multi-class classification, and regression.

        Parameters
        ----------
        predict_fn: skater.model.model.Model subtype
            estimator "prediction" function to explain the predictive model. Could be probability scores
            or target values
        filter_classes: array type
            The classes to run partial dependence on. Default None invokes all classes.
            Only used in classification models.
        ascending: boolean, default True
            Helps with ordering Ascending vs Descending
        ax: matplotlib.axes._subplots.AxesSubplot
            existing subplot on which to plot feature importance. If none is provided,
            one will be created.

        Returns
        -------
        f: figure instance
        ax: matplotlib.axes._subplots.AxesSubplot
            could be used to for further modification to the plots

        Examples
        --------
            >>> from skater.model import InMemoryModel
            >>> from skater.core.explanations import Interpretation
            >>> from sklearn.ensemble import RandomForestClassier
            >>> rf = RandomForestClassier()
            >>> rf.fit(X,y)
            >>> model = InMemoryModel(rf, examples = X)
            >>> interpreter = Interpretation()
            >>> interpreter.load_data(X)
            >>> interpreter.feature_importance.plot_feature_importance(model, ascending=True, ax=ax)
            """
        try:
            global pyplot
            from matplotlib import pyplot
        except ImportError:
            raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
        except RuntimeError:
            raise (MatplotlibDisplayError("Matplotlib unable to open display"))

        importances = self.feature_importance(predict_fn, filter_classes=filter_classes)

        if ax is None:
            f, ax = pyplot.subplots(1)
        else:
            f = ax.figure

        colors = cycle(COLORS)
        color = next(colors)
        # Below is a weirdness because of how pandas plot is behaving. There might be a better way
        # to resolve the issuse of sorting based on axis
        if ascending is True:
            importances.sort_values(ascending=False).plot(kind='barh', ax=ax, color=color)
        else:
            importances.sort_values(ascending=True).plot(kind='barh', ax=ax, color=color)
        return f, ax


    def compute_importance(self, new_predictions, original_predictions, original_x, perturbed_x,
                           method='output-variance', scaled=False):
        if method == 'output-variance':
            importance = self._compute_importance_via_output_variance(np.array(new_predictions),
                                                                      np.array(original_predictions),
                                                                      np.array(original_x),
                                                                      np.array(perturbed_x),
                                                                      scaled)
        else:
            raise(KeyError("Unrecongized method for computing feature_importance: {}".format(method)))
        return importance


    def _compute_importance_via_output_variance(self, new_predictions, original_predictions,
                                                original_x, perturbed_x, scaled=True):
        """Mean absolute error of predictions given perturbations in a feature"""
        changes_in_predictions = abs(new_predictions - original_predictions)

        if scaled:
            changes_in_predictions = self._importance_scaler(changes_in_predictions, original_x, perturbed_x)

        importance = np.mean(changes_in_predictions)
        return importance


    def _importance_scaler(self, values, original_x, perturbed_x):
        raise(NotImplementedError("We currently don't support scaling, we are researching the best"
                                  "approaches to do so."))
