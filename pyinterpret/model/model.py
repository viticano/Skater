"""Model class."""

import abc
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
        self.formatter = lambda x: x

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
            print "shape is 1"
            self.output_shape = (None,)
            # the predict function is either a continuous prediction,
            # or a most-likely classification
            example_output = outputs[0]
            self.output_var_type = return_data_type(example_output)
            print "output type: {}".format(self.output_var_type)
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
            self.probability = (self.output_var_type == StaticTypes.output_types.float)
        else:
            raise ValueError("Unsupported model type, output dim = 3")

        self.formatter = self.return_formatter_func()

    @staticmethod
    def classifier_prediction_to_encoded_output(output):
        """Call this method when model returns a 1D array of
        predicted classes. The output is one hot encoded version"""
        label_encoder = LabelEncoder()
        _labels = label_encoder.fit_transform(output)[:, np.newaxis]
        #class_names = label_encoder.classes_.tolist()

        onehot_encoder = OneHotEncoder()
        output = onehot_encoder.fit_transform(_labels).todense()
        output = np.squeeze(np.asarray(output))
        return output


    def return_formatter_func(self):
        """In the event that the predict func returns 1D array of predictions,
        then this returns a formatter to convert outputs to a 2D one hot encoded
        array."""
        #Note this expression below assumptions (not probability) evaluates to false if
        #and only if the model does not return probabilities. If unknown, should be true
        if self.model_type == StaticTypes.model_types.classifier and not self.probability:
            return self.classifier_prediction_to_encoded_output
        else:
            return lambda x: x
