from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("fastspike.pyx"),
    include_dirs=[numpy.get_include()]
)

# Command to run:
# python setup.py build_ext --inplace