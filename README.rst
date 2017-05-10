.. raw:: html

    <a href="https://www.datascience.com/"><img src="/datascienceinc/Skater/blob/master/Skater.png?raw=true" alt="Skater.png" style="max-width: 100%"></a>

.. raw:: html

    <a href="https://www.datascience.com/"><img src="https://cdn2.hubspot.net/hubfs/532045/DS_LogoHorizontal%20Colored.svg" width="300" height="50" align="right" /></a>


Skater
===========
Skater is a python package for model agnostic interpretation of predictive models.
With Skater, you can unpack the internal mechanics of arbitrary models; as long
as you can obtain inputs, and use a function to obtain outputs, you can use
Skater to learn about the models internal decision criteria.

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
=================== ===

.. _Overview: https://datascienceinc.github.io/Skater/overview.html
.. _Installing: https://datascienceinc.github.io/Skater/install.html
.. _Tutorial: https://datascienceinc.github.io/Skater/tutorial.html
.. _API Reference: https://datascienceinc.github.io/Skater/api.html


Master: |Build Status-master|
'''''''''''''''''''''''''''''

.. raw:: html

   <!--![layout](../master/skate.png?raw=true)
   =======
   -->

Dev Installation
~~~~~~~~~~~~~~~~

::

    git clone git@github.com:datascienceinc/skate.git
    cd skate
    sudo python setup.py install

Prod Installation (platform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires that pip.conf is configured with pypi.datascience.com
credentials.

::

    pip install skate

Use this kind of this stuff to do cool stuff.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    from skate.core.explanations import Interpretation
    from skate.model import InMemoryModel
    i = Interpretation()
    i.load_data(X, feature_names = feature_names)
    model = InMemoryModel(regressor.predict, examples = X)
    i.partial_dependence.plot_partial_dependence([feature_names[0], feature_names[1]],
                                                model)

    #local interpretation
    from skate.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(X, feature_names = feature_names)
    explainer.explain_instance(example,  regressor.predict).show_in_notebook()

Testing
~~~~~~~

::

    python skate/tests/all_tests.py --debug --n=1000 --dim=3 --seed=1

API documentation
~~~~~~~~~~~~~~~~~

::

    https://datascienceinc.github.io/model-interpretation/py-modindex.html

.. |Build Status-master| image:: https://api.travis-ci.com/repositories/datascienceinc/Skater.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master
