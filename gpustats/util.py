import numpy as np
import pymc.distributions as pymc_dist
import pycuda.driver as drv

from pandas.util.testing import set_trace as st

_dev_attr = drv.device_attribute

class DeviceInfo(object):

    def __init__(self, dev=0):
        self.dev = dev
        self._dev = drv.Device(dev)
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]

HALF_WARP = 16

def random_cov(dim):
    return pymc_dist.rinverse_wishart(dim, np.eye(dim))

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

def pad_data(data):
    """
    Pad data to avoid bank conflicts on the GPU-- dimension should not be a
    multiple of the half-warp size (16)
    """
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
        arr = np.array(arr, dtype=np.float32)

    return arr


def tune_blocksize(data, params, device=0):
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
    info = DeviceInfo(device)

    max_smem = info.shared_mem * 0.9
    max_threads = info.max_block_threads

    params_per = max_threads
    if (len(params) < params_per):
        params_per = _next_pow2(len(params), info.max_block_threads)

    data_per = max_threads / params_per

    def _can_fit(data_per, params_per):
        return compute_shmem(data, params, data_per, params_per) <= max_smem

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

        if data_per == 0:
            # we failed somehow. start over
            data_per = 1
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
    param_space = params.shape[1] * params_per
    data_space = data.shape[1] * data_per
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
