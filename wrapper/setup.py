"""
Setup script for SloughGPT Wrapper

Build with:
    python setup.py build_ext --inplace
    
Or install Cython first:
    pip install cython
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "sloughgpt_wrapper",
        sources=["sloughgpt_wrapper.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=["-lstdc++"],
    )
]

setup(
    name="sloughgpt_wrapper",
    version="1.0.0",
    description="SloughGPT ML Wrapper - Protected Cython Extension",
    author="SloughGPT Team",
    packages=[],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        }
    ),
    install_requires=[
        "numpy>=1.20.0",
    ],
)
