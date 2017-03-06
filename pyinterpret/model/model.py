"""Model class."""

import abc
import requests
import numpy as np
from ..util.static_types import StaticTypes, return_data_type

class Model(object):
    """What is a model? A model needs to make predictions, so a means of
    passing data into the model, and receiving results.

    Goals:
        We want to abstract away how we access the model.
        We want to make inferences about the format of the output.
        We want to able to map model outputs to some smaller, universal set of output types.
        We want to infer whether the model is real valued, or classification (n classes?)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, examples=None):
        self.examples = examples
        self.model_type = StaticTypes.unknown
        self.output_var_type = StaticTypes.unknown
        self.output_shape = StaticTypes.unknown
        self.n_classes = StaticTypes.unknown
        self.input_shape = StaticTypes.unknown
        self.probability = StaticTypes.unknown

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """The way in which the submodule predicts values given an input"""
        return

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def check_output_signature(self, examples):
        """Determines the model_type, output_type"""
        ids = range(len(examples))
        idx = np.random.choice(ids, size=5, replace=True)
        inputs = examples[idx]
        outputs = self(inputs)
        self.output_shape = outputs.shape
        if len(self.output_shape) == 1:
            self.output_shape = (None,)
            # the predict function is either a continuous prediction,
            # or a most-likely classification
            example_output = outputs[0]
            self.output_var_type = return_data_type(example_output)

            if self.output_var_type == StaticTypes.output_types.string:
                # the prediction is yield groups as strings, as in a classification model
                self.model_type = StaticTypes.model_types.classifier
                self.probability = False

            elif self.output_var_type == StaticTypes.output_types.int:
                # the prediction is yield groups as integers, as in a classification model
                self.model_type = StaticTypes.model_types.classifier
                self.probability = False

            elif self.output_var_type == StaticTypes.output_types.float:
                # the prediction is yield groups, as in a classification model
                self.model_type = StaticTypes.model_types.regressor
                self.n_classes = StaticTypes.not_applicable
                self.probability = StaticTypes.not_applicable
            else:
                pass  # default unknowns will take care of this
        elif len(self.output_shape) == 2:
            self.output_shape = (None, self.output_shape[1])
            self.model_type = StaticTypes.model_types.classifier
            self.n_classes = self.output_shape[1]
            example_output = outputs[0][0]
            self.output_var_type = return_data_type(example_output)
            if self.output_var_type == StaticTypes.output_types.float:
                self.probability = True
            else:
                self.probability = False
        else:
            raise ValueError("Unsupported model type, output dim = 3")


class InMemoryModel(Model):
    """This model can be called directly from memory"""
    def __init__(self, prediction_fn, examples=None):
        """This model can be called directly from memory
        Parameters
        ----------
        prediction_fn(callable)
            function that returns predictions

        examples(numpy.array or pandas.dataframe):
            examples to use to make inferences about the function.
            prediction_fn must be able to take examples as an
            argument.
        """
        super(InMemoryModel, self).__init__(self)
        self.prediction_fn = prediction_fn
        self.examples = np.array(examples)
        if self.examples.any():
            self.check_output_signature(self.examples)

    def predict(self, *args, **kwargs):
        """Just use the function itself for predictions"""
        return self.prediction_fn(*args, **kwargs)


class WebModel(Model):
    """Model that gets predictions from a deployed model"""
    def __init__(self, uri, parse_function=None):
        """This model can be called by making http requests
        to the passed in uri.

        Parameters
        ----------
        uri(string)
            Where to post requests

        parse_function(callable)
            This function will run on outputs before returning
            results to interpretation objects.
        """
        super(WebModel, self).__init__(self)
        self.uri = uri
        self.parse_function = parse_function or self.default_parser

    @staticmethod
    def default_parser(content):
        """Just returns raw results"""
        return content

    def predict(self, request_body, **kwargs):
        if self.__confirm_server_is_healthy(self.uri):
            output = self.parse_function(requests.get(self.uri, data=request_body, **kwargs).content)
            return output
        else:
            raise ValueError("Server is not running.")

    def __call__(self, *args, **kwargs):
        """Just use the function itself for predictions"""
        return self.predict(*args, **kwargs)

    @staticmethod
    def __confirm_server_is_healthy(uri):
        """Makes sure uri is up"""
        return bool(uri)
