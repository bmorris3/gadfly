#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception:
    __version__ = '{version}'
""".lstrip()

ext_modules = [
    Extension(
        "gadfly.cython",
        ["gadfly/cython.pyx"],
        extra_compile_args=['-g0'],
        extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'],
        include_dirs=[np.get_include()]
    )
]

setup(
    use_scm_version={'write_to': os.path.join('gadfly', 'version.py'),
                     'write_to_template': VERSION_TEMPLATE},
    ext_modules=cythonize(ext_modules)
)
