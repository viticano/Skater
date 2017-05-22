# Reference: http://python-packaging.readthedocs.io/en/latest/dependencies.html
from setuptools import setup, find_packages
import os, io, sys
import contextlib

@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)

def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    with chdir(root):
        with io.open(os.path.join(root, 'skater', 'about.py'), encoding='utf8') as f:
            about = {}
            exec(f.read(), about)

        with io.open(os.path.join(root, 'description.rst'), encoding='utf8') as f:
            readme = f.read()

    setup(
        name=about['__title__'],
        zip_safe=False,
        packages=find_packages(),
        description=about['__summary__'],
        long_description=readme,
        author=about['__author__'],
        author_email=about['__email__'],
        version=about['__version__'],
        url=about['__uri__'],
        license=about['__license__'],
        install_requires=[
            'scikit-learn>=0.18',
            'pandas>=0.19',
            'ds-lime>=0.1.1.21',
            'requests',
            'pathos==0.2.0',
            'dill>=0.2.6'],
        extras_require ={'all':'matplotlib'},
        )

if __name__ == '__main__':
    setup_package()
