import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
import os
from pycuda.compiler import SourceModule

_dev_attr = drv.device_attribute

class DeviceInfo(object):

    def __init__(self, dev=0):
        self.dev = dev
        self._dev = drv.Device(dev)
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]

class CUDAModule(object):
    """
    Interfaces with PyCUDA
    """
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict
        self.support_code = _get_support_code()
        self.pycuda_module = SourceModule(self._get_full_source())

    def _get_full_source(self):
        kernel_code = [v.get_code() for v in self.kernel_dict.values()]
        return '\n\n'.join([self.support_code] + kernel_code)

    def get_function(self, name):
        return self.pycuda_module.get_function('k_%s' % name)

def _get_support_code():
    path = os.path.join(_get_cuda_code_path(), 'support.cu')
    return open(path).read()

def _get_mvcaller_code():
    path = os.path.join(_get_cuda_code_path(), 'callers.cu')
    return open(path).read()

def _get_cuda_code_path():
    import os.path as pth
    return pth.abspath(pth.split(__file__)[0])

class DensityKernel(object):
    """

    """
    def __init__(self, name, logic_code):
        self.name = name
        self.logic_code = logic_code

    def get_code(self):
        caller_code = _get_mvcaller_code()
        formatted_caller = caller_code % {'name' : self.name}

        code = '\n\n'.join((self.logic_code, formatted_caller))

        return code

def get_source_module():
    """
    Generates the fully assembled PyCUDA source module
    """

    pass

if __name__ == '__main__':
    from numpy.random import randn
    from numpy.linalg import cholesky as chol
    import numpy as np
    import numpy.linalg as L
    import scipy.special as sp
    import time

    import gpustats.kernels as kernels
    import gpustats.util as util

    reload(kernels)

    k = DensityKernel('pdf_mvnormal', kernels.pdf_mvnormal)
    module = CUDAModule({'pdf_mvnormal' : k})

    f = module.get_function('pdf_mvnormal')

    n = 256
    k = 4

    data = randn(n, k).astype(np.float32)
    mean = randn(k)
    cov = np.array(util.random_cov(k), dtype=np.float32)

    j = 1

    padded_data = util.pad_data(data)

    chol_sigma = chol(cov)
    ichol_sigma = L.inv(chol_sigma)

    logdet = np.log(np.linalg.det(cov))

    means = (mean,) * j
    covs = (ichol_sigma,) * j
    logdets = (logdet,) * j

    packed_params = util.pack_params(means, covs, logdets)

    data_per, params_par =

    block_design = (data_per * params_per, 1, 1)

    design = np.array(((data_per, params_per) +
                       padded_data.shape +
                       (k,) +
                       packed_params.shape),
                      dtype=np.float32)

    dest = np.zeros((n, j), dtype=np.float32, order='F')

    grid_design =

    f(drv.Out(dest),
      drv.In(padded_data),
      drv.In(packed_params),
      drv.In(design),
      block=block_design, shared=14000)
