"Subclass of Model for deployed models"
import requests
import numpy as np
from .model import ModelType


class DeployedModel(ModelType):
    """Model that gets predictions from a deployed model"""
    def __init__(self, uri, input_formatter, output_formatter,
                 log_level=30, class_names=None, examples=None):
        """This model can be called by making http requests
        to the passed in uri.

        Parameters
        ----------
        uri: string
            Where to post requests

        input_formatter: callable
            This function will run on input data before passing
            to requests library. This usually should take array types
            and convert them to JSON.

        output_formatter: callable
            This function will run on outputs before returning
            results to interpretation objects. This usually should take
            request objects and convert them to array types.

        log_level: int
            see pyinterpret.model.Model for details

        class_names: array type
            see pyinterpret.model.Model for details

        examples:
            see pyinterpret.model.Model for details
        """
        self.uri = uri
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
        super(DeployedModel, self).__init__(examples=examples,
                                            class_names=class_names,
                                            log_level=log_level)


    @staticmethod
    def default_input_wrapper(data, key='input'):
        return {key: data.tolist()}


    @staticmethod
    def default_output_wrapper(response, key='prediction'):
        return np.array(response.json()[key])


    @staticmethod
    def _predict(data, uri, input_formatter, output_formatter, formatter=None):
        """Static prediction function for multiprocessing usecases

        Parameters
        ----------
        data: arraytype

        uri: string
            Where to post requests

        input_formatter: callable
            This function will run on input data before passing
            to requests library. This usually should take array types
            and convert them to JSON.

        output_formatter: callable
            This function will run on outputs before returning
            results to interpretation objects. This usually should take
            request objects and convert them to array types.

        formatter: callable
            function responsible for formatting model outputs as necessary. For instance,
            one hot encoding multiclass outputs.

        predict_fn: callable

        Returns
        -----------
        predictions: arraytype
        """

        query = input_formatter(data)
        response = requests.post(uri, json=query)
        results = output_formatter(response)
        if formatter:
            results = formatter(results)
        return results


    def predict(self, data):
        """Predict method for deployed models. Takes an array,
        passes it through a formatter for the requests library,
        which makes a request. The response is then passed to the
        output formatter, which parses results into an array type.
        Finally, the interal formatter (one hot encoding, etc),
        formats the final results.

        Parameters
        ----------
        data: array type

        Returns
        ----------
        predictions: array type
        """
        return self._predict(data,
                             uri=self.uri,
                             input_formatter=self.input_formatter,
                             output_formatter=self.output_formatter,
                             formatter=self.formatter
                             )
