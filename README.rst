.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>

Skater
===========
Skater is a python package for model agnostic interpretation of predictive models.
With Skater, you can unpack the internal mechanics of arbitrary models; as long
as you can obtain inputs, and use a function to obtain outputs, you can use
Skater to learn about the models internal decision policies.


The package was originally developed by Aaron Kramer, Pramit Choudhary and internal DataScience Team at DataScience.com
to help enable practitioners explain and interpret predictive "black boxes" in a human interpretable way.

.. image:: https://api.travis-ci.com/repositories/datascienceinc/Skater.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master
    :target: https://travis-ci.com/datascienceinc/Skater
    :alt: Build Status

📖 Documentation
================

=================== ===
`Overview`_         Introduction to the Skater library
`Installing`_       How to install the Skater library
`Tutorial`_         Steps to use Skater effectively.
`API Reference`_    The detailed reference for Skater's API.
`Contributing`_     Guide to contributing to the Skater project.
=================== ===

.. _Overview: https://datascienceinc.github.io/Skater/overview.html
.. _Installing: https://datascienceinc.github.io/Skater/install.html
.. _Tutorial: https://datascienceinc.github.io/Skater/tutorial.html
.. _API Reference: https://datascienceinc.github.io/Skater/api.html
.. _Contributing: https://github.com/datascienceinc/Skater/blob/readme/CONTRIBUTING.rst

💬 Feedback/Questions
==========================

=========================  ===
**Feature Requests/Bugs**  `GitHub issue tracker`_
**Usage questions**        `Gitter chat`_
**General discussion**     `Gitter chat`_
=========================  ===

.. _GitHub issue tracker: https://github.com/datascienceinc/Skater/issues
.. _Gitter chat: https://gitter.im/datascienceinc-skater

Install Skater
================

Dependencies
~~~~~~~~~~~~~~~~
Skater relies on numpy, pandas, scikit-learn, and the DataScience.com fork of
the LIME package. Plotting functionality requires matplotlib, though it is not
required to install the package. Currently we only distribute to pypi, though
adding a conda distribution is on the roadmap.

pip
~~~~~~~~~~~~~~~~

When using pip, to ensure your system is not modified by an installation, it
is recommended that you use a virtual environment (virtualenv, conda environment).

::

    pip install -U Skater


Usage
==============
The code below illustrates a typical workflow with the Skater package.

::

    import numpy as np
    from scipy.stats import norm

    #gen some data
    B = np.random.normal(0, 10, size = 3)
    X = np.random.normal(0,10, size=(1000, 3))
    feature_names = ["feature_{}".format(i) for i in xrange(3)]
    e = norm(0, 5)
    y = np.dot(X, B) + e.rvs(1000)
    example = X[0]

    #model it
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor()
    regressor.fit(X, y)


    #partial dependence
    from skater.core.explanations import Interpretation
    from skater.model import InMemoryModel
    i = Interpretation()
    i.load_data(X, feature_names = feature_names)
    model = InMemoryModel(regressor.predict, examples = X)
    i.partial_dependence.plot_partial_dependence([feature_names[0], feature_names[1]],
                                                model)

    #local interpretation
    from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(X, feature_names = feature_names)
    model = InMemoryModel(regressor.predict)
    explainer.explain_regressor_instance(example,  model).show_in_notebook()

Testing
~~~~~~~
1. If repo is cloned:
::

    python skate/tests/all_tests.py

2. If pip installed:
::

    python -c "from skater.tests.all_tests import run_tests; run_tests()"


.. |Build Status-master| image:: https://api.travis-ci.com/repositories/datascienceinc/Skater.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master
