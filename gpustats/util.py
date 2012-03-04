import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda
import pycuda.autoinit

from pycuda.compiler import SourceModule




_dev_attr = drv.device_attribute

class DeviceInfo(object):

    def __init__(self):
        self._dev = pycuda.autoinit.device
        #self._dev = drv.Device(dev)
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]
        self.max_registers = self._attr[_dev_attr.MAX_REGISTERS_PER_BLOCK]
        self.compute_cap = self._dev.compute_capability()

HALF_WARP = 16

def random_cov(dim):
    from pymc.distributions import rinverse_wishart
    return rinverse_wishart(dim, np.eye(dim))

def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result

def pad_data_mult16(data, fill=0):
    """
    Pad data to be a multiple of 16 for discrete sampler.
    """

    if type(data) == gpuarray:
        data = data.get()

    n, k = data.shape

    km = int(k/16) + 1

    newk = km*16
    if newk != k:
        padded_data = np.zeros((n, newk), dtype=np.float32)
        if fill!=0:
            padded_data = padded_data + fill

        padded_data[:,:k] = data

        return padded_data
    else:
        return prep_ndarray(data)

def pad_data(data):
    """
    Pad data to avoid bank conflicts on the GPU-- dimension should not be a
    multiple of the half-warp size (16)
    """
    if type(data) == gpuarray:
        data = data.get()

    n, k = data.shape

    if not k % HALF_WARP:
        pad_dim = k + 1
    else:
        pad_dim = k

    if k != pad_dim:
        padded_data = np.empty((n, pad_dim), dtype=np.float32)
        padded_data[:, :k] = data

        return padded_data
    else:
        return prep_ndarray(data)

def prep_ndarray(arr):
    # is float32 and contiguous?
    if not arr.dtype == np.float32 or not arr.flags.contiguous:
        arr = np.array(arr, dtype=np.float32, order='C')

    return arr




def tune_blocksize(data, params, func_regs):
    """
    For multivariate distributions-- what's the optimal block size given the
    gpu?

    Parameters
    ----------
    data : ndarray
    params : ndarray

    Returns
    -------
    (data_per, params_per) : (int, int)
    """
    # TODO: how to figure out active device in this thread for the multigpu
    # case?
    info = DeviceInfo()

    max_smem = info.shared_mem * 0.9
    max_threads = int(info.max_block_threads * 0.5)
    max_regs = info.max_registers

    params_per = max_threads
    if (len(params) < params_per):
        params_per = _next_pow2(len(params), info.max_block_threads)

    data_per = max_threads / params_per

    def _can_fit(data_per, params_per):
        ok = compute_shmem(data, params, data_per, params_per) <= max_smem
        return ok and func_regs*data_per*params_per <= max_regs

    while True:
        while not _can_fit(data_per, params_per):
            if data_per <= 1:
                break

            if params_per > 1:
                # reduce number of parameters first
                params_per /= 2
            else:
                # can't go any further, have to do less data
                data_per /= 2

        if data_per <=1:
            # we failed somehow. start over
            data_per = 2
            params_per /= 2
            continue
        else:
            break

    while _can_fit(2 * data_per, params_per):
        if 2 * data_per * params_per < max_threads:
            data_per *= 2
        else:
            # hit block size limit
            break

    return data_per, params_per

def get_boxes(n, box_size):
    # how many boxes of size box_size are needed to hold n things
    return int((n + box_size - 1) / box_size)

def compute_shmem(data, params, data_per, params_per):
    result_space = data_per * params_per

    data_dim = 1 if len(data.shape) == 1 else data.shape[1]
    params_dim = len(params) if len(params.shape) == 1 else params.shape[1]

    param_space = params_dim * params_per
    data_space = data_dim * data_per
    return 4 * (result_space + param_space + data_space)

def _next_pow2(k, pow2):
    while k <= pow2 / 2:
        pow2 /= 2
    return pow2

def next_multiple(k, mult):
    if k % mult:
        return k + (mult - k % mult)
    else:
        return k

def get_cufiles_path():
    import os.path as pth
    basepath = pth.abspath(pth.split(__file__)[0])
    return pth.join(basepath, 'cufiles')


from pycuda.tools import context_dependent_memoize

@context_dependent_memoize
def _get_transpose_kernel():

    info = DeviceInfo()
    if info.max_block_threads >= 1024:
        t_block_size = 32
    else:
        t_block_size = 16

    import os.path as pth
    mod = SourceModule( 
        open(pth.join(get_cufiles_path(), "transpose.cu")).read() % { "block_size" : t_block_size })

    func = mod.get_function("transpose")
    func.prepare("PPii", block=(t_block_size, t_block_size, 1))

    from pytools import Record
    class TransposeKernelInfo(Record): pass

    return TransposeKernelInfo(func=func, 
                               block_size=t_block_size,
                               granularity=t_block_size)

def _transpose(tgt, src):
    krnl = _get_transpose_kernel()
    
    w, h = src.shape
    assert tgt.shape == (h, w)
    #assert w % krnl.granularity == 0
    #assert h % krnl.granularity == 0
    
    gw = int(np.ceil(float(w) / krnl.granularity))
    gh = int(np.ceil(float(h) / krnl.granularity))

    krnl.func.prepared_call(
        (gw, gh),
        tgt.gpudata, src.gpudata, w, h)

def transpose(src):
    w, h = src.shape

    result = gpuarray.empty((h, w), dtype=src.dtype)
    _transpose(result, src)
    return result
