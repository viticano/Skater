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

TODO: Add a note on whether we distribute sources/binaries

When using pip, to ensure your system is not modified by an installation, it
is recommended that you use a virtual environment (virtualenv, conda environment).

::

    pip install -U Skater


testing
~~~~~~~~~~~~~~~~

::

    python -c "from skater.tests.all_tests import run_tests; run_tests()"
