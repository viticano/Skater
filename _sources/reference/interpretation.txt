Interpretation Objects
=======================================

Overview
---------------------------------------
Interpretation are initialized with a DataManager object, and expose interpretation algorithms as methods. For instance:

.. code-block:: python
   :linenos:

   from pyinterpret import Interpretation()
   interpreter = Interpretation()
   interpreter.load_data(data)
   interpreter.feature_importance.feature_importance(model)

Loading Data
---------------------------------------
Before running interpretation algorithms on a model, the Interpretation object usually needs data, either to learn about
the distribution of the training set or to pass inputs into a prediction function.

When calling Interpretation.load_data, the object creates a DataManager object, which handles the data, keeping track of feature
and observation names, as well as providing various sampling algorithms.

Currently load_data requires a numpy ndarray or pandas DataFrame, though we may add support for additional data structures in the future.
For more details on what the DataManager does, please see the relevant documentation [PROVIDE LINK].

.. currentmodule:: pyinterpret


.. autosummary::

   core.explanations.Interpretation.load_data
