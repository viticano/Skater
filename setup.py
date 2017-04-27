# Reference: http://python-packaging.readthedocs.io/en/latest/dependencies.html
from setuptools import setup, find_packages

setup(name='lynxes',
      version='0.0.1',
      description='Model Agnostic Interpretation Library',
      author='Aaron Kramer',
      author_email='aaron@datascience.com',
      url='https://github.com/datascienceinc/lynxes/tree/master',
      packages=find_packages(),
      install_requires=['scikit-learn>=0.18', 'pandas>=0.19', 'lime>=0.1.1.20', 'requests',
                        'pathos==0.2.0', 'dill>=0.2.6'],
      extras_require ={'all':'matplotlib'},
      include_package_data=True,
      zip_safe=False,
      dependency_links=['https://github.com/datascienceinc/lime/tarball/v1-release#egg=lime-0.1.1.20'])
