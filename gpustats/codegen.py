import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
import os
from pycuda.compiler import SourceModule

class CUDAModule(object):
    """
    Interfaces with PyCUDA

    Parameters
    ----------
    kernel_dict :
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
    # for multivariate pdfs
    path = os.path.join(_get_cuda_code_path(), 'mvcaller.cu')
    return open(path).read()

def _get_univcaller_code():
    # For univariate pdfs
    path = os.path.join(_get_cuda_code_path(), 'univcaller.cu')
    return open(path).read()

def _get_cuda_code_path():
    import os.path as pth
    return pth.abspath(pth.split(__file__)[0])

class DensityKernel(object):
    """
    Generate kernel for probability density function
    """

    _caller = _get_univcaller_code()
    def __init__(self, name, logic_code):
        self.name = name
        self.logic_code = logic_code

    def get_code(self):
        caller_code = self._caller
        formatted_caller = caller_code % {'name' : self.name}

        code = '\n\n'.join((self.logic_code, formatted_caller))

        return code

class MVDensityKernel(DensityKernel):
    """

    """
    _caller = _get_mvcaller_code()

# right place for registry?

kernel_registry = {
    'pdf_mvnormal' : kernels.pdf_mvnormal
}

def get_full_cuda_module():
    return CUDAModule(kernel_registry)

if __name__ == '__main__':
    pass
