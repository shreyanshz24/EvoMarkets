# setup.py
from setuptools import setup, Extension
import pybind11
import sys
cpp_args = []
if sys.platform == 'win32':
    cpp_args = ['/std:c++17', '/O2']
else:
    cpp_args = ['-std=c++17', '-O3']

ext_modules = [
    Extension(
        'fast_lob',
        [
            'LimitOrderBook.cpp',
            'wrapper.cpp'
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='fast_lob',
    version='0.0.1',
    description='High-performance C++ LOB for market simulation',
    ext_modules=ext_modules,
)