Tutorial
===========


The skater workflow
------------------

The general workflow within the skater package is to create an interpretation, create a model, and run interpretation algorithms.

Creating an interpretation object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An Interpretation consumes a dataset, and optionally some meta data like feature names and row ids.
Internally, the Interpretation will generate a DataManager to handle data requests and sampling.

.. code-block:: python

   from sklearn.datasets import load_boston
   boston = load_boston()
   X, y, features = boston.data, boston.target, boston.feature_names

   from skater import Interpretation
   interpreter = Interpretation(X, feature_names=features)

To begin using the Interpretation to explain models, we need to create a skater Model.

Creating a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To ensure formatting consistency, and common functionality among various model types and interfaces,
we require that models passed to Interpretations to be wrapped in a skater Model object. Currently we support regression type
models and classifiers that return probability scores. If you have a need for explaining classifiers that dont provide probability scores, please
let us know: https://github.com/datascienceinc/model-interpretation/issues/82.

To create a skater model based on a local function or method, pass in the predict function to an InMemoryModel. A user can optionally pass data samples
to the examples keyword argument. This is only used to infer output types and formats. Out of the box, skater allows models return numpy arrays and
pandas dataframes. If youd like support for additional formats, please let us know: https://github.com/datascienceinc/model-interpretation/issues/117

.. code-block:: python

   from sklearn.ensemble import GradientBoostedRegressor
   gb = GradientBoostedRegressor()
   gb.fit(X, y)

   from skater.model import InMemoryModel
   model = InMemoryModel(gb.predict, examples = X[:10])

If your model requires or returns different data structures, you can instead create a function that converts outputs to an appropriate
data structure.

.. code-block:: python

   def predict_as_dataframe(x):
       return pd.DataFrame(gb.predict(x))

   from skater.model import InMemoryModel
   model = InMemoryModel(predict_as_dataframe, examples = X[:10])

If your model is accessible through an api, use a DeployedModel, which wraps the requests library. DeployedModels require two functions,
an input formatter and an output formatter, which speak to the requests library for posting and parsing.

The input formatter takes a pandas DataFrame or a numpy ndarray, and returns an object (such as a dict) that can be converted to json
to be posted. The output formatter takes a requests.response as an input and returns a numpy ndarray or pandas DataFrame:

.. code-block:: python

   from skater.model import DeployedModel
   import numpy as np

   def input_formatter(x): return {'data': list(x)}
   def output_formatter(response): return np.array(response.json()['output'])
   uri = "https://yourorg.com/model/endpoint"
   model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10])


If your api requires additional configuration like cookies, use request_kwargs:

.. code-block:: python

   from skater.model import DeployedModel
   import numpy as np

   req_kwargs = {'cookies': {'cookie-name':'cookie'}}
   model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10], request_kwargs=req_kwargs)


With an Interpretation and a Model, one can run all skater interpretation algorithms.

.. code-block:: python

   interpreter.feature_importance.feature_importance(model)

   interpreter.partial_dependence.plot_partial_dependence([features[0], features[1]], model)

For details on the interpretation algorithms currently available, please see the documentation for:

- :ref:`interpretation-feature-importance`
- :ref:`interpretation-partial-dependence`
- :ref:`interpretation-overview-local`
