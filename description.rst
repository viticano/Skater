Skater
===========
Skater is a python package for model agnostic interpretation of predictive models.
With Skater, you can unpack the internal mechanics of arbitrary models; as long
as you can obtain inputs, and use a function to obtain outputs, you can use
Skater to learn about the models internal decision policies.


The package was originally developed by Aaron Kramer, Pramit Choudhary and internal DataScience Team at DataScience.com
to help enable practitioners explain and interpret predictive "black boxes" in a human interpretable way.

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
.. _Gitter chat: https://gitter.im/datascienceinc/skater

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


