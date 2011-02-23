import numpy as np
import pymc.distributions as pymc_dist


_dev_attr = drv.device_attribute

class DeviceInfo(object):

    def __init__(self, dev=0):
        self.dev = dev
        self._dev = drv.Device(dev)
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]

PAD_MULTIPLE = 16
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

def next_multiple(k, p):
    if k % p:
        return k + (p - k % p)

    return k

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
