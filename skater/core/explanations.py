"""Interpretation Class"""

from .global_interpretation.partial_dependence import PartialDependence
from .global_interpretation.feature_importance import FeatureImportance
from ..data import DataManager
from ..util.logger import build_logger


class Interpretation(object):
    """
    Interpretation class. Before calling interpretation subclasses like partial
    dependence, one must call Interpretation.load_data().

    Attributes:
    ----------
    data_set: skater.data.DataSet
        skater's data abstraction layer. All interactions with data goes through
         this object.

    local_interpreter: skater.core.local_interpreter.LocalInterpreter
        Contains methods for interpreting single prediction. Currently exposes the lime
        library.

    global_interpreter: skater.core.global_interpreter.GlobalInterpreter
        Contains methods for evaluating a model over entire regions of the domain. Currently
        exposes partial dependency


    Examples:
    ----------

    from skater.core.explanations import Interpretation
    interpreter = Interpretation()
    interpreter.load_data(X, feature_ids = ['a','b'])
    interpreter.partial_dependence([feature_id1, feature_id2], regressor.predict)
    """

    def __init__(self, training_data=None, feature_names=None, index=None,
                 log_level=30):
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
        self.partial_dependence = PartialDependence(self)
        self.feature_importance = FeatureImportance(self)
        self.data_set = None

        if training_data is not None:
            self.load_data(training_data, feature_names=feature_names, index=index)


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
        self.data_set = DataManager(training_data,
                                    feature_names=feature_names,
                                    index=index,
                                    log_level=self._log_level)
        self.logger.info("Data loaded")
        self.logger.debug("Data shape: {}".format(self.data_set.data.shape))
        self.logger.debug("Dataset Feature_ids: {}".format(self.data_set.feature_ids))
