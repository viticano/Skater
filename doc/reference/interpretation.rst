Interpretation Objects
======================

.. _interpretation-overview:

Overview
--------
Interpretation are initialized with a DataManager object, and expose interpretation algorithms as methods. For instance:

.. code-block:: python
   :linenos:

   from lynxes import Interpretation()
   interpreter = Interpretation()
   interpreter.load_data(data)
   interpreter.feature_importance.feature_importance(model)

Loading Data
------------
Before running interpretation algorithms on a model, the Interpretation object usually needs data, either to learn about
the distribution of the training set or to pass inputs into a prediction function.

When calling Interpretation.load_data, the object creates a DataManager object, which handles the data, keeping track of feature
and observation names, as well as providing various sampling algorithms.

Currently load_data requires a numpy ndarray or pandas DataFrame, though we may add support for additional data structures in the future.
For more details on what the DataManager does, please see the relevant documentation [PROVIDE LINK].

.. currentmodule:: lynxes


.. automethod:: lynxes.core.explanations.Interpretation.load_data

Global Interpretations
----------------------
A predictive model is a mapping from an input space to an output space. Interpretation algorithms
are divided into those that offer statistics and metrics on regions of the domain, such as the
marginal distribution of a feature, or the joint distribution of the entire training set.
In an ideal world there would exist some representation that would allow a human
to interpret a decision function in any number of dimensions. Given that we generally can only
intuit visualizations of a few dimensions at time, global interpretation algorithms either aggregate
or subset the feature space.

Currently, model agnostic global interpretation algorithms supported by lynxes include
partial dependence and feature importance.


.. _interpretation-feature-importance:

Feature Importance
~~~~~~~~~~~~~~~~~~
Feature importance is generic term for the degree to which a predictive model relies on a particular
feature. Lynxes feature importance implementation is based on an information theoretic criteria,
measuring the entropy in the change of predictions, given a perturbation of a given feature.
The intuition is that the more a model's decision criteria depend on a feature, the
more we'll see predictions change as a function of perturbing a feature.

.. autoclass:: lynxes.core.global_interpretation.feature_importance.FeatureImportance
   :members:


.. _interpretation-partial-dependence:

Partial Dependence
~~~~~~~~~~~~~~~~~~
Partial Dependence describes the marginal impact of a feature on model prediction, holding
other features in the model constant. The derivative of partial dependence describes the impact of a feature (analogous to a feature coefficient
in a regression model).

.. autoclass:: lynxes.core.global_interpretation.partial_dependence.PartialDependence
   :members:

.. _interpretation-overview-local:

Local Interpretations
----------------------
Local interpretations are based on using interpretable surrogate models to illustrate
how features impact predictions constrained to a particular point or small region in
the input space. Linear surrogates around a point correspond the LIME algorithm; tree like
surrogates around a point correspond to anchorLIME.

LIME
~~~~~~~~~~~~~~~~~~

.. autoclass:: lynxes.core.local_interpretation.lime.lime_tabular.LimeTabularExplainer
   :members:
