"""Model class."""

import abc
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.multiclass import type_of_target
import pandas as pd

from ..util.static_types import StaticTypes
from ..util.logger import build_logger
from ..util import exceptions
from ..data import DataManager


class ModelType(object):
    """What is a model? A model needs to make predictions, so a means of
    passing data into the model, and receiving results.

    Goals:
        We want to abstract away how we access the model.
        We want to make inferences about the format of the output.
        We want to able to map model outputs to some smaller, universal set of output types.
        We want to infer whether the model is real valued, or classification (n classes?)
    """
    __metaclass__ = abc.ABCMeta


    def __init__(self, log_level=30, target_names=None, examples=None, feature_names=None, unique_values=None,
                 input_formatter=None, output_formatter=None):
        """
        Base model class for wrapping prediction functions. Common methods
        involve output type inference in requiring predict methods

        Parameters
        ----------
            log_level: int
                0, 10, 20, 30, 40, or 50 for verbosity of logs.
            target_names: arraytype
                The names of the target variable/classes. There should be as many
                 names as there are outputs per prediction (n=1 for regression,
                 n=2 for binary classification, etc). Defaults to Predicted Value for
                 regression and Class 1...n for classification.



        Attributes
        ----------
            model_type: string

        """
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)
        self.examples = None
        self.model_type = StaticTypes.unknown
        self.output_var_type = StaticTypes.unknown
        self.output_shape = StaticTypes.unknown
        self.n_classes = StaticTypes.unknown
        self.input_shape = StaticTypes.unknown
        self.probability = StaticTypes.unknown
        self.transformer = self.identity_function
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()
        self.target_names = target_names
        self.feature_names = feature_names
        self.unique_values = unique_values
        self.input_formatter = input_formatter or self.identity_function
        self.output_formatter = output_formatter or self.identity_function
        self.has_metadata = False

        if examples is not None:
            self.input_type = type(examples)
            examples = DataManager(examples, feature_names=feature_names)
            self._build_model_metadata(examples)
        else:
            self.input_type = None
            self.logger.warn("No examples provided, cannot infer model type")


    def predict(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        return self.transformer(self.output_formatter(self._execute(self.input_formatter(*args, **kwargs))))


    @abc.abstractmethod
    def _execute(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        return


    @abc.abstractmethod
    def _predict(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        return

    @abc.abstractmethod
    def _get_static_predictor(self, *args, **kwargs):
        """Return a static prediction function to avoid shared state in multiprocessing"""
        return


    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    def check_examples(self, examples):
        """
        Ties examples to self. equivalent to self.examples = np.array(examples).
        Parameters
        ----------
        examples: array type


        """
        if isinstance(examples, (pd.DataFrame, np.ndarray)):
            return examples
        else:
            return np.array(examples)


    def _build_model_metadata(self, dataset):
        """
        Determines the model_type, output_type. Side effects
        of this method are to mutate object's attributes (model_type,
        n_classes, etc).

        Parameters
        ----------
        examples: pandas.DataFrame or numpy.ndarray
            The examples that will be passed through the predict function.
            The outputs from these examples will be used to make inferences
            about the types of outputs the function generally makes.

        """
        self.logger.debug("Beginning output checks")

        if self.input_type in (None, pd.DataFrame):
            outputs = self.predict(dataset.data)
        elif self.input_type == np.ndarray:
            outputs = self.predict(dataset.data)
        else:
            raise ValueError("Unrecognized input type: {}".format(self.input_type))

        self.input_shape = dataset.data.shape
        self.output_shape = outputs.shape

        ndim = len(outputs.shape)
        if ndim > 2:
            raise(ValueError("Unsupported model type, output dim = {}".format(ndim)))

        try:
            # continuous, binary, continuous multioutput, multiclass, multilabel-indicator
            self.output_type = type_of_target(outputs)
        except:
            self.output_type = False

        if self.output_type == 'continuous':
            self.model_type = StaticTypes.model_types.regressor
            self.n_classes = 1
            self.probability = StaticTypes.not_applicable

        elif self.output_type == 'multiclass':
            self.model_type = StaticTypes.model_types.classifier
            self.probability = False
            self.n_classes = len(np.unique(outputs))

        elif self.output_type == 'continuous-multioutput':
            self.model_type = StaticTypes.model_types.classifier
            self.probability = True
            self.n_classes = outputs.shape[1]

        elif self.output_type == 'binary':
            self.model_type = StaticTypes.model_types.classifier
            self.probability = False
            self.n_classes = 2

        elif self.output_type == 'multilabel-indicator':
            self.model_type = StaticTypes.model_types.classifier
            self.probability = False
            self.n_classes = outputs.shape[1]

        else:
            err_msg = "Could not infer model type"
            self.logger.debug("Inputs: {}".format(dataset.data))
            self.logger.debug("Outputs: {}".format(outputs))
            self.logger.debug("sklearn response: {}".format(self.output_type))
            exceptions.ModelError(err_msg)

        if self.target_names is None:
            self.target_names = range(self.n_classes)

        if self.unique_values is None and self.model_type == 'classifier' and self.probability is False:
            raise (exceptions.ModelError('If using classifier without probability scores, unique_values cannot '
                                         'be None'))

        self.transformer = self.transformer_func_factory(outputs)

        reports = self.model_report(dataset.data)
        for report in reports:
            self.logger.debug(report)

        if self.target_names is None:
            self.target_names = range(self.n_classes)

        self.has_metadata = True


    def predict_function_transformer(self, output):
        """
        Call this method when model returns a 1D array of
        predicted classes. The output is one hot encoded version.

        Parameters
        ----------
        output: array type
            The output of the pre-formatted predict function

        Returns
        ----------
        output: numpy.ndarray
            The one hot encoded outputs of predict_fn
        """

        _labels = self.label_encoder.transform(output)[:, np.newaxis]
        # target_names = label_encoder.classes_.tolist()

        self.logger.debug("Using transforming function. Found {} classes".format(len(self.label_encoder.classes_)))
        self.logger.debug("Label shape: {}".format(len(_labels.shape)))
        output = self.one_hot_encoder.transform(_labels).todense()
        output = np.squeeze(np.asarray(output))
        return DataManager(output, feature_names=self.label_encoder.classes_)[self.unique_values]


    def transformer_func_factory(self, outputs):
        """
        In the event that the predict func returns 1D array of predictions,
        then this returns a formatter to convert outputs to a 2D one hot encoded
        array.

        For instance, if:
            predict_fn(data) -> ['apple','banana']
        then
            transformer = Model.transformer_func_factory()
            transformer(predict_fn(data)) -> [[1, 0], [0, 1]]

        Returns
        ----------
        (callable):
            formatter function to wrap around predict_fn
        """

        # Note this expression below assumptions (not probability) evaluates to false if
        # and only if the model does not return probabilities. If unknown, should be true
        if self.model_type == StaticTypes.model_types.classifier and not self.probability:
            # fit label encoder
            artificial_samples = np.array(self.unique_values)
            self.logger.debug("Label encoder fit on examples of shape: {}".format(outputs.shape))
            self.label_encoder.fit(artificial_samples)
            labels = self.label_encoder.transform(artificial_samples)[:, np.newaxis]
            self.logger.debug("Onehot encoder fit on examples of shape: {}".format(labels.shape))
            self.one_hot_encoder.fit(labels)
            return self.predict_function_transformer
        else:
            return lambda x: x


    def model_report(self, examples):
        """
        Just returns a list of model attributes as a list

        Parameters
        ----------
        examples: array type:
            Examples to use for which we report behavior of predict_fn.


        Returns
        ----------
        reports: list of strings
            metadata about function.

        """
        examples = DataManager(examples, feature_names=self.feature_names)
        reports = []
        if isinstance(self.examples, np.ndarray):
            raw_predictions = self.predict(examples)
            reports.append("Example: {} \n".format(examples[0]))
            reports.append("Outputs: {} \n".format(raw_predictions[0]))
        reports.append("Model type: {} \n".format(self.model_type))
        reports.append("Output Var Type: {} \n".format(self.output_var_type))
        reports.append("Output Shape: {} \n".format(self.output_shape))
        reports.append("N Classes: {} \n".format(self.n_classes))
        reports.append("Input Shape: {} \n".format(self.input_shape))
        reports.append("Probability: {} \n".format(self.probability))
        return reports

    def predict_subset_classes(self, data, subset_of_classes):
        """Filters predictions to a subset of classes."""
        if subset_of_classes is None:
            return self.predict(data)
        else:
            return DataManager(self.predict(data), feature_names=self.target_names)[subset_of_classes]


    @staticmethod
    def identity_function(x):
        return x
