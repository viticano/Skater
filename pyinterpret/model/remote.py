"Subclass of Model for deployed models"
import requests
from .model import Model

class DeployModel(Model):
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
        super(DeployModel, self).__init__(self)
        self.uri = uri
        self.parse_function = parse_function or self.default_parser

    @staticmethod
    def default_parser(content):
        """Just returns raw results"""
        return content

    def predict(self, request_body, **kwargs):
        if self.__confirm_server_is_healthy(self.uri):
            output = self.parse_function(
                requests.get(self.uri, data=request_body, **kwargs).content
            )
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
