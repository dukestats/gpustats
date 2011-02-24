#/usr/bin/env python

from distutils.extension import Extension
from numpy.distutils.core import setup
from Cython.Distutils import build_ext
import numpy

def get_cuda_include():
    return '/usr/local/cuda/include'

pyx_ext = Extension('testmod', ['cytest.pyx'],
                    include_dirs=[numpy.get_include(),
                                  get_cuda_include()],
                    library_dirs=['.'],
                    libraries=['gpustats'])

setup(name='testmod', description='',
      ext_modules=[pyx_ext],
      cmdclass = {
          'build_ext' : build_ext
      })
