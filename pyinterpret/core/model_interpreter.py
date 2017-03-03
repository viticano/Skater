"""Base ModelInterpreter class. submodules like lime and partial dependence"
must have these methods"""

from ..model.model import InMemoryModel


class ModelInterpreter(object):
    '''
    Base Interpreter class. Common methods include loading a dataset and type setting.
    '''

    def __init__(self, interpreter):
        self.interpreter = interpreter

    @property
    def data_set(self):
        """data_set routes to the Interpreter's dataset"""
        return self.interpreter.data_set

    @staticmethod
    def _types():
        return ['partial_dependence', 'lime']

    def consider(self, training_data, index=None, feature_names=None):
        """.consider routes to Interpreter's .consider"""
        self.interpreter.consider(training_data, index=index, feature_names=feature_names)

    def build_annotated_model(self, prediction_function):
        """returns a callable model that has annotations.
        Uses examples from the Interpreter's dataset if available

        Parameters
        ----------
        prediction_function(callable):
            function to generate predictions

        Returns
        ----------
        pyinterpret.model.Model type.
        """
        if self.interpreter.data_set:
            examples = self.interpreter.data_set.generate_sample(sample=True,
                                                                 n_samples_from_dataset=5,
                                                                 strategy='random-choice')
        else:
            examples = None
        annotated_model = InMemoryModel(prediction_function, examples=examples)
        return annotated_model
