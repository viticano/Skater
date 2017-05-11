
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
