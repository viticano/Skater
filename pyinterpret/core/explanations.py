"""Interpretation Class"""

from .global_interpretation.partial_dependence import PartialDependence
from .local_interpretation.local_interpreter import LocalInterpreter
from ..data.dataset import DataSet
from ..model.local import InMemoryModel
from ..util.logger import build_logger


class Interpretation(object):
    """
    Interpretation class. Before calling interpretation subclasses like partial
    dependence, one must call Interpretation.load_data().

    Attributes:
    ----------
    data_set: pyinterpret.data.DataSet
        Pyinterpret's data abstraction layer. All interactions with data goes through
         this object.

    local_interpreter: pyinterpret.core.local_interpreter.LocalInterpreter
        Contains methods for interpreting single prediction. Currently exposes the lime
        library.

    global_interpreter: pyinterpret.core.global_interpreter.GlobalInterpreter
        Contains methods for evaluating a model over entire regions of the domain. Currently
        exposes partial dependency


    Examples:
    ----------

    from pyinterpret.core.explanations import Interpretation
    interpreter = Interpretation()
    interpreter.load_data(X, feature_ids = ['a','b'])
    interpreter.partial_dependence([feature_id1, feature_id2], regressor.predict)
    """

    def __init__(self, log_level='WARNING'):
        """
        Attaches local and global interpretations
        to Interpretation object.

        Parameters:
        -----------
        log_level: int
            Logger Verbosity, see https://docs.python.org/2/library/logging.html
            for details.

        """
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)
        self.local_interpreter = LocalInterpreter(self)
        self.partial_dependence = PartialDependence(self)
        self.data_set = None

    def load_data(self, training_data, feature_names=None, index=None):
        """
        Creates a DataSet object from inputs, ties to interpretation object.
        This will be exposed to all submodules.

        Parameters
        ----------
        training_data: numpy.ndarray, pandas.DataFrame
            the dataset. can be 1D or 2D

        feature_names: array-type
            names to call features.

        index: array-type
            names to call rows.


        Returns
        ---------
            None
        """

        self.logger.info("Loading Data")
        self.data_set = DataSet(training_data,
                                feature_names=feature_names,
                                index=index,
                                log_level=self._log_level)
        self.logger.info("Data loaded")
        self.logger.debug("Data shape: {}".format(self.data_set.data.shape))
        self.logger.debug("Dataset Feature_ids: {}".format(self.data_set.feature_ids))

    def build_annotated_model(self, prediction_function, class_names=None, examples=None):
        """
        Returns a callable model that has annotations.
        Uses examples from the Interpreter's dataset if available

        Parameters
        ----------
        prediction_function: callable
            function to generate predictions

        examples: np.ndarray or pd.DataFrame
            The examples to use when calling InMemoryModel.check_output_types(examples).


        Returns
        ----------
        annotated_model: pyinterpret.model.InMemoryModel
        """

        if examples is None:
            self.logger.warn('Model will not be annotated, no examples provided.')

        annotated_model = InMemoryModel(prediction_function,
                                        class_names=class_names,
                                        examples=examples,
                                        log_level=self._log_level)
        return annotated_model
