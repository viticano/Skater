from setuptools import setup

setup(name='PyInterpret',
      version='0.0.1',
      description='Model Agnostic Interpretation Library',
      author='Aaron Kramer',
      author_email='aaron@datascience.com',
      url='https://github.com/datascienceinc/PyInterpret/tree/master',
      packages=['PyInterpret'],
      install_requires = ['numpy','scipy','scikit-learn>=0.18'],
      include_package_data=True,
      zip_safe=False,
     )
