#/usr/bin/env python

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup

DESCRIPTION = "GPU-based statistical functions"
LONG_DESCRIPTION = """
gpustats is a PyCUDA-based library implementing functionality similar to that
present in scipy.stats. It implements a simple framework for specifying new CUDA
kernels and extending existing ones. Here is a (partial) list of target
functionality:

* Probability density functions (pdfs). These are intended to speed up
  likelihood calculations in particular in Bayesian inference applications, such
  as in PyMC

* Random variable generation using CURAND

Notes
-----
Requires workign PyCUDA installation
"""

REQUIRES = ['numpy', 'pycuda >= 0.94rc']
DISTNAME = 'gpustats'
LICENSE = 'BSD'
AUTHOR = "Wes McKinney"
AUTHOR_EMAIL = "wesmckinn@gmail.com"
URL = "https://github.com/dukestats/gpustats"
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
]

MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.beta'

def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path,
                           version=FULLVERSION)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('gpustats')
    return config

if __name__ == '__main__':
    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          classifiers=CLASSIFIERS,
          platforms='any',
          configuration=configuration)
