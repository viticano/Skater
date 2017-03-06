from .global_interpretation.partial_dependence import PartialDependence
from .local_interpretation.local_interpreter import LocalInterpreter
from ..data.dataset import DataSet
from ..model.model import InMemoryModel


# Create based on class name:
class Interpretation(object):
    '''
    Returns an interpretation class.

    Parameters:
    -----------
        interpretation_type(string): pdp, lime

    Returns:
    ----------
        interpretation subclass
    '''

    def __init__(self):
        self.local_interpreter = LocalInterpreter(self)
        self.partial_dependence = PartialDependence(self)
        self.data_set = None

    def consider(self, training_data, feature_names=None, index=None):
        """Creates a DataSet object from inputs, ties to interpretation object.
        This will be exposed to all submodules.

        Parameters
        ----------
        training_data(numpy.ndarray, pandas.DataFrame):
            the dataset. can be 1D or 2D

        feature_names(array-type):
            names to call features.

        index(array-type):
            names to call rows.

        """
        self.data_set = DataSet(training_data, feature_names=feature_names, index=index)

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
        if self.data_set:
            examples = self.data_set.generate_sample(sample=True,
                                                                 n_samples_from_dataset=5,
                                                                 strategy='random-choice')
        else:
            examples = None
        annotated_model = InMemoryModel(prediction_function, examples=examples)
        return annotated_model
