# Reference: http://python-packaging.readthedocs.io/en/latest/dependencies.html
from setuptools import setup, find_packages

setup(name='pyinterpret',
      version='0.0.1',
      description='Model Agnostic Interpretation Library',
      author='Aaron Kramer',
      author_email='aaron@datascience.com',
      url='https://github.com/datascienceinc/PyInterpret/tree/master',
      packages=find_packages(),
      install_requires=['cython', 'numpy>=1.10', 'scipy>=0.15',' scikit-learn>=0.18', 'pandas>=0.17', 'lime>=0.1.1.20'],
      include_package_data=True,
      zip_safe=False,
      dependency_links=['https://github.com/datascienceinc/lime/tarball/v1-dev#egg=lime-0.1.1.20'])
