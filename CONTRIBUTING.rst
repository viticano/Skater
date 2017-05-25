
.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>


Contributing to Skater
===========
Skater is an open source project released under the MIT license. We invite
users help contribute to the project by reporting bugs, requesting features, working
on the documentation and codebase.

.. contents:: Types of Contributions

Reporting a bug
---------------
As with any GitHub project, one should always begin addressing a bug by searching through existing issues.
If an issue for the bug does not exist, please create one with the relevant tags:

=================== ===
Performance         Memory and speed related issues.
Installation        Issues experienced attempting to install Skater.
Plotting            Issues associated with Skater's plotting functionality.
Enhancement         Request for a new feature, or augmentation of an existing feature.
Bug                 Errors or unexpected behavior.
=================== ===

We may augment this tag set as needed.

Submitting a test
-----------------
Currently there are 4 test files:

============================ ===
Feature Importance           test_feature_importance.py
Partial Dependence           test_partial_dependence.py
Tests of LIME functionality  test_lime.py
DataManager                  test_data.py
============================ ===

New tests should be added to relevant test file. If no current file covers
the feature, please add a new file with the following structure:

::

    class MyNewTestClass(unittest.TestCase):
        def setUp(self):
            create objects and data

        def test_function_1(self):
            ...
        def test_function_2(self):
            ...
    if __name__ == '__main__':
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(unittest.makeSuite(TestData))


Contributing Code
-----------------
Skater is distributed under an MIT license. By submitting a pull request for this project,
you agree to license your contribution under the MIT license to this project as well.

Style
~~~~~~~~~~~~~~~~~~~~
Stylistically, contributions should follow PEP8, with the exception that methods
are separated by 2 lines instead of 1.


Pull Requests
~~~~~~~~~~~~~~~~~~~~
Before a PR is accepted, travis builds must pass on all environments, and flake8
tests must all pass.


Dependencies
~~~~~~~~~~~~~~~~~~~~
Every additional package dependency adds a potential installation complexity,
so only dependencies that are critical to the package should be added to the
code base. PRs that involve the addition of a new dependency will be evaluated
by the following criteria:

- Is the application of the dependency isolated, such that removing it is trivial, or
  will it be deeply integrated into the package.
- Does the dependency have known installation issues on common platforms?
- Does the application using the dependency need to be in the Skater package?



.. |Build Status-master| image:: https://api.travis-ci.com/repositories/datascienceinc/Skater.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master
.. |Skater Logo White| image:: https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg


Roadmap
---------------
We'd like to improve the package in a few key ways. The list below
represents aspects we definitely want to address.

======================= ===
Performance             We would like to improve performance where ever possible. Model agnostic algorithms can
                        only be implemented under a "perturb and observe" framework, whereby inputs are selectively
                        chosen, outputs are observed, and metrics, inferences, visualizations are created. Therefore,
                        the bottleneck is always the speed of the prediction function, which we will not have control over.
                        The best way to improve Skater performance is with parallelization (what is the quickest way
                        to execute N function calls) and intelligent sampling (how few function calls can we make/
                        how few observations can we pass to each function).
Algorithms              There are other interpretation algorithms we'd like to support. One family of algorithms would
                        fall under the category of "model surrogates", where models are approximated, either locally
                        (like LIME) or globally. These algorithms must be accurate/faithful to the original model,
                        and simple/interpretable to be useful. We are considering bayesian rule lists and regressions for
                        now, though this may change. The user would also need to know if and where the surrogate is
                        a poor representative of the original model. There are also extensions to partial dependence such
                        as ICE and accumulated local effects plots that may give better indication of interaction effects.
Plotting                We'd like to iterate on our visualizations to make them more intuitive, and ideally not rely
                        on matplotlib.
======================= ===

The following list represents aspects that are not definite, but are currently being considered for the roadmap.

=======================   ===
Validation                Currently Skater explains model behavior. It in no way evaluates the quality of that behavior
                          through validation. Extending the library to support conditional validation--when and why does a
                          model do well or poorly--may be within scope.
Model Comparison          Early users of the package used Skater to compare models to each other. Currently, model comparisons
                          must be done manually by the user; run an algorithm, store results in a dictionary, plot.
Dependence probabilities  Currently there is no built in way to assess whether a model may have learned an interaction
                          other than explicitly plotting partial dependence with respect to two features. We're interested
                          in providing a matrix of probabilities of dependence. More formally, for features x1, x2, and model f
                          f(x1, x2, x_compliment), the probability that the partial derivative of f with respect to x1
                          is unequal to the probability is unequal to that conditioned on values of x2.
========================  ===
