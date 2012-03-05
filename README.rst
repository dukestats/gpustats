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

Requirements
------------

* NumPy
* SciPy
* Working PyCUDA (http://pypi.python.org/pypi/pycuda) installation
* (optional) PyMC, for test suite

Installation and testing
------------------------

To install, run:

::

    python setup.py install

If you have `nose` installed, you may run the test suite by running:

::

    import gpustats
	gpustats.test()

Use
---

::

    import gpustats

Some development guidelines
---------------------------

* Use spaces (4 per indent), not tabs
* Trim whitespace at the end of lines (most text editors will do this for you)
* PEP8-consistent Python style

People
------

Cliburn Chan cliburn.chan (at) duke.edu
Andrew Cron ajc40 (at) stat.duke.edu
Jacob Frelinger jacob.frelinger (at) duke.edu
Wes McKinney wesmckinn (at) gmail.com
Adam Richards adam.richards (at) duke.edu
Marc Suchard msuchard (at) ucla.edu
Quanli Wang quanli (at) stat.duke.edu
Mike West mw (at) stat.duke.edu

Notes
-----
Requires working PyCUDA installation
