"""Model subclass for in memory predict functions"""
import numpy as np
from .model import Model

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
        # if self.examples.any():
        #     self.check_output_signature(self.examples)

    def predict(self, *args, **kwargs):
        """Just use the function itself for predictions"""
        return self.formatter(self.prediction_fn(*args, **kwargs))
