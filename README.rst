========
GPUStats
========

gpustats is a PyCUDA-based library implementing functionality similar to that
present in scipy.stats. It implements a simple framework for specifying new CUDA
kernels and extending existing ones. Here is a (partial) list of target
functionality:

* Probability density functions (pdfs). These are intended to speed up
  likelihood calculations in particular in Bayesian inference applications, such
  as in PyMC

* Random variable generation using CURAND

Installation and testing
------------------------

To install, run:

::

    python setup.py install

If you have `nose` installed, you may run the test suite by running:

::

    nosetests gpustats

Notes
-----
Requires working PyCUDA installation
