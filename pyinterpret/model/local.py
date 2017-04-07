"""Model subclass for in memory predict functions"""

from .model import Model
from ..util import exceptions


class InMemoryModel(Model):
    """
    This model can be called directly from memory
    """

    def __init__(self, prediction_fn, log_level=30, class_names=None, examples=None):
        """This model can be called directly from memory
        Parameters
        ----------
        prediction_fn: callable
            function that returns predictions

        log_level: int
            config setting to see model logs. 10 is a good value for seeing debug messages.
            30 is warnings only.

        class_names: array type
            optional names of classes that describe model outputs.

        examples(numpy.array or pandas.dataframe):
            examples to use to make inferences about the function.
            prediction_fn must be able to take examples as an
            argument.
        """
        super(InMemoryModel, self).__init__(log_level=log_level, class_names=class_names)
        self.prediction_fn = prediction_fn

        if not hasattr(prediction_fn, "__call__"):
            raise exceptions.ModelError("Predict function must be callable")
        examples = self.set_examples(examples)
        if examples.any():
            self.check_output_signature(examples)

    def predict(self, *args, **kwargs):
        """
        Just use the function itself for predictions
        """
        return self.formatter(self.prediction_fn(*args, **kwargs))

    @staticmethod
    def static_predict(data, formatter, predict_fn):
        """Static prediction function for multiprocessing usecases

        Parameters
        ----------
        data: arraytype

        formatter: callable
            function responsible for formatting model outputs as necessary. For instance,
            one hot encoding multiclass outputs.

        predict_fn: callable

        Returns
        -----------
        predictions: arraytype
        """
        return formatter(predict_fn(data))
