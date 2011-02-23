import numpy as np
import pymc.distributions as pymc_dist

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

def pack_params(means, chol_sigmas, logdets):
    to_pack = []
    for m, ch, ld in zip(means, chol_sigmas, logdets):
        to_pack.append(pack_pdf_params(m, ch, ld))

    return np.vstack(to_pack)

def pack_pdf_params(mean, chol_sigma, logdet):
    '''


    '''
    k = len(mean)
    mean_len = k
    chol_len = k * (k + 1) / 2
    mch_len = mean_len + chol_len

    packed_dim = next_multiple(mch_len + 2, PAD_MULTIPLE)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = chol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = 1, logdet

    return packed_params
