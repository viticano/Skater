"Subclass of Model for deployed models"
import requests
import numpy as np
from .model import Model


class DeployedModel(Model):
    """Model that gets predictions from a deployed model"""
    def __init__(self, uri, input_formatter, output_formatter,
                 log_level=30, class_names=None, examples=None):
        """
        This model can be called by making http requests
        to the passed in uri.

        Parameters
        ----------
        uri(string)
            Where to post requests

        parse_function(callable)
            This function will run on outputs before returning
            results to interpretation objects.
        """
        super(DeployedModel, self).__init__()
        self.uri = uri
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
        examples = self.set_examples(examples)
        if examples.any():
            self.check_output_signature(examples)

    @staticmethod
    def default_input_wrapper(data):
        return {'input': data.tolist()}

    @staticmethod
    def default_output_wrapper(response, key='prediction'):
        return np.array(response.json()[key])

    @staticmethod
    def static_predict(data, uri, input_formatter, output_formatter, formatter):

        query = input_formatter(data)
        response = requests.post(uri, json=query)
        return formatter(output_formatter(response))

    def predict(self, data):
        query = self.input_formatter(data)
        response = requests.post(self.uri, json=query)
        return self.formatter(self.output_formatter(response))

    def __call__(self, *args, **kwargs):
        """Just use the function itself for predictions"""
        return self.predict(*args, **kwargs)

