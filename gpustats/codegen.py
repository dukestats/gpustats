import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
import os
from pycuda.compiler import SourceModule
from gpustats.util import get_cufiles_path

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

        self.all_code = self._get_full_source()
        try:
            self.pycuda_module = SourceModule(self.all_code)
        except Exception:
            f = open('foo.cu', 'w')
            print >> f, self.all_code
            f.close()
            raise

    def _get_full_source(self):
        formatted_kernels = [kern.get_code()
                             for kern in self.kernel_dict.values()]
        return '\n'.join([self.support_code] + formatted_kernels)

    def get_function(self, name):
        return self.pycuda_module.get_function('k_%s' % name)

def _get_support_code():
    path = os.path.join(get_cufiles_path(), 'support.cu')
    return open(path).read()

def _get_mvcaller_code():
    # for multivariate pdfs
    path = os.path.join(get_cufiles_path(), 'mvcaller.cu')
    return open(path).read()

def _get_univcaller_code():
    # For univariate pdfs
    path = os.path.join(get_cufiles_path(), 'univcaller.cu')
    return open(path).read()

class Kernel(object):

    def __init__(self, name):
        if name is None:
            raise ValueError('Kernel must have a default name')

        self.name = name

    def get_code(self):
        logic = self.get_logic()
        caller = self.get_caller()
        return '\n'.join((logic, caller))

    def get_logic(self, **kwds):
        raise NotImplementedError

    def get_caller(self, **kwds):
        raise NotImplementedError

    def get_name(self, name=None):
        # can override default name, for transforms. this a hack?
        if name is None:
            name = self.name

        return name

class CUFile(Kernel):
    """
    Expose kernel contained in .cu file in the cufiles directory to code
    generation framework. Kernel need only have a template to be able to change
    the name of the generated kernel
    """
    def __init__(self, name, filepath):
        self.full_path = os.path.join(get_cufiles_path(),
                                      filepath)

        Kernel.__init__(self, name)

    def get_code(self):
        code = open(self.full_path).read()
        return code % {'name' : self.name}

class SamplerKernel(Kernel):
    """
    Holds info for measure sample kernel.
    """
    def __init__(self, name, logic_code):
        self.logic_code = logic_code
        Kernel.__init__(self, name)

    def get_logic(self, name=None):
        return self.logic_code

    def get_caller(self, name=None):
        return self._caller % {'name' : self.get_name(name)}

class DensityKernel(Kernel):
    """
    Generate kernel for probability density function
    """

    _caller = _get_univcaller_code()
    def __init__(self, name, logic_code):

        self.logic_code = logic_code

        Kernel.__init__(self, name)

    def get_logic(self, name=None):
        return self.logic_code % {'name' : self.get_name(name)}

    def get_caller(self, name=None):
        return self._caller % {'name' : self.get_name(name)}

class MVDensityKernel(DensityKernel):
    """

    """
    _caller = _get_mvcaller_code()

class Transform(Kernel):
    """
    Enable simple transforms of kernels to compute modified kernel code stub
    """
    def __init__(self, name, kernel):
        self.kernel = kernel
        Kernel.__init__(self, name)

    # XXX: HACK, not general for non-density kernels
    def is_multivariate(self):
        return isinstance(self.kernel, MVDensityKernel)

# flop the right name?

class Flop(Transform):
    op = None

    def get_logic(self, name=None):
        name = self.get_name(name)

        actual_name = '%s_stub' % name
        kernel_logic = self.kernel.get_logic(name=actual_name)

        if self.is_multivariate():
            stub_caller = _mv_stub_caller
        else:
            stub_caller = _univ_stub_caller

        transform_logic = stub_caller % {'name' : name,
                                         'actual_kernel' : actual_name,
                                         'op' : self.op}

        return '\n'.join((kernel_logic, transform_logic))

    def get_caller(self):
        return self.kernel.get_caller(self.name)

_univ_stub_caller = """
__device__ float %(name)s(float* x, float* params) {
    return %(op)s(%(actual_kernel)s(x, params));
}
"""

_mv_stub_caller = """
__device__ float %(name)s(float* x, float* params, int dim) {
    return %(op)s(%(actual_kernel)s(x, params, dim));
}
"""

class Exp(Flop):
    op = 'expf'

class Log(Flop):
    op = 'logf'

class Sqrt(Flop):
    op = 'sqrtf'

_cu_module = None

def get_full_cuda_module():
    import gpustats.kernels as kernels
    global _cu_module

    if _cu_module is None:
        objects = kernels.__dict__

        all_kernels = dict((k, v)
                           for k, v in kernels.__dict__.iteritems()
                           if isinstance(v, Kernel))
        _cu_module = CUDAModule(all_kernels)

    return _cu_module

if __name__ == '__main__':
    pass
