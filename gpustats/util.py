import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda
drv.init()
if drv.Context.get_current() is None:
    import pycuda.autoinit
from pycuda.compiler import SourceModule

def threadSafeInit(device = 0):
    """
    If gpustats (or any other pycuda work) is used inside a 
    multiprocessing.Process, this function must be used inside the
    thread to clean up invalid contexts and create a new one on the 
    given device. Assumes one GPU per thread.
    """

    import atexit
    drv.init() # just in case

    ## clean up all contexts. most will be invalid from
    ## multiprocessing fork
    import os; import sys
    clean = False
    while not clean:
        _old_ctx = drv.Context.get_current()
        if _old_ctx is None:
            clean = True
        else:
            ## detach: will give warnings to stderr if invalid
            _old_cerr = os.dup(sys.stderr.fileno())
            _nl = os.open(os.devnull, os.O_RDWR)
            os.dup2(_nl, sys.stderr.fileno())
            _old_ctx.detach() 
            sys.stderr = os.fdopen(_old_cerr, "w")
            os.close(_nl)
    from pycuda.tools import clear_context_caches
    clear_context_caches()
        
    ## init a new device
    dev = drv.Device(device)
    ctx = dev.make_context()

    ## pycuda.autoinit exitfunc is bad now .. delete it
    exit_funcs = atexit._exithandlers
    for fn in exit_funcs:
        if hasattr(fn[0], 'func_name'):
            if fn[0].func_name == '_finish_up':
                exit_funcs.remove(fn)
            if fn[0].func_name == 'clean_all_contexts': # avoid duplicates
                exit_funcs.remove(fn)

    ## make sure we clean again on exit
    atexit.register(clean_all_contexts)


def clean_all_contexts():

    ctx = True
    while ctx is not None:
        ctx = drv.Context.get_current()
        if ctx is not None:
            ctx.detach()

    from pycuda.tools import clear_context_caches
    clear_context_caches()
    

def GPUarray_reshape(garray, shape=None, order="C"):
    if shape is None:
        shape = garray.shape
    return gpuarray.GPUArray(
        shape=shape,
        dtype=garray.dtype,
        allocator=garray.allocator,
        base=garray,
        gpudata=int(garray.gpudata),
        order=order)

def GPUarray_order(garray, order="F"):
    """
    will set the order of garray in place
    """
    if order=="F":
        if garray.flags.f_contiguous:
            exit
        else:
            garray.strides = gpuarray._f_contiguous_strides(
                garray.dtype.itemsize, garray.shape)
            garray.flags.f_contiguous = True
            garray.flags.c_contiguous = False
    elif order=="C":
        if garray.flags.c_contiguous:
            exit
        else:
            garray.strides = gpuarray._c_contiguous_strides(
                garray.dtype.itemsize, garray.shape)
            garray.flags.c_contiguous = True
            garray.flags.f_contiguous = False
            


_dev_attr = drv.device_attribute
## TO DO: should be different for each device .. assumes they are the same
class DeviceInfo(object):

    def __init__(self):
        #self._dev = pycuda.autoinit.device
        #self._dev = drv.Device(dev)
        self._dev = drv.Context.get_device()
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]
        self.max_registers = self._attr[_dev_attr.MAX_REGISTERS_PER_BLOCK]
        self.compute_cap = self._dev.compute_capability()
        self.max_grid_dim = (self._attr[_dev_attr.MAX_GRID_DIM_X],
                             self._attr[_dev_attr.MAX_GRID_DIM_Y])

info = DeviceInfo()

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
    #info = DeviceInfo()

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

    #info = DeviceInfo()
    if info.max_block_threads >= 1024:
        t_block_size = 32
    else:
        t_block_size = 16

    import os.path as pth
    mod = SourceModule( 
        open(pth.join(get_cufiles_path(), "transpose.cu")).read() % { "block_size" : t_block_size })

    func = mod.get_function("transpose")
    func.prepare("PPii") #, block=(t_block_size, t_block_size, 1))
    return t_block_size, func
    

    #from pytools import Record
    #class TransposeKernelInfo(Record): pass
    #return TransposeKernelInfo(func=func, 
    #                           block_size=t_block_size,
    #                           granularity=t_block_size)
    

def _transpose(tgt, src):
    block_size, func = _get_transpose_kernel()
    

    h, w = src.shape
    assert tgt.shape == (w, h)
    #assert w % block_size == 0
    #assert h % block_size == 0
    
    gw = int(np.ceil(float(w) / block_size))
    gh = int(np.ceil(float(h) / block_size))
    gz = int(1)

    ### 3D grids are needed for larger data ... should be comming soon ...
    #while gw > info.max_grid_dim[0]:
    #    gz += 1
    #    gw = int(np.ceil(float(w) / (gz * block_size) ))

    func.prepared_call(
        (gw, gh),
        (block_size, block_size, 1),
        tgt.gpudata, src.gpudata, w, h)


def transpose(src):
    h, w = src.shape

    result = gpuarray.empty((w, h), dtype=src.dtype)
    _transpose(result, src)
    del src
    return result
