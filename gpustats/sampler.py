import numpy as np

import gpustats.kernels as kernels
import gpustats.codegen as codegen
reload(codegen)
reload(kernels)
import gpustats.util as util
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

from pycuda.curandom import rand as curand

cu_module = codegen.get_full_cuda_module()

def sample_discrete(in_densities, logged=False, pad=False,
                    return_gpuarray=False):
    """
    Takes a categorical sample from the unnormalized univariate
    densities defined in the rows of 'densities'

    Parameters
    ---------
    densities : ndarray or gpuarray (n, k)
    logged: boolean indicating whether densities is on the
    log scale ...

    Returns
    -------
    indices : ndarray or gpuarray (if return_gpuarray=True)
    of length n and dtype = int32
    """

    if pad:
        if logged:
            densities = util.pad_data_mult16(in_densities, fill=1)
        else:
            densities = util.pad_data_mult16(in_densities, fill=0)

    else:
        densities = in_densities

    n, k = densities.shape

    cu_func = cu_module.get_function('sample_measure')

    if type(densities)==gpuarray:
        gpu_densities = densities
    else:
        densities = util.prep_ndarray(densities)
        gpu_densities = gpuarray.to_gpu(densities)

    # setup GPU data
    gpu_random = curand(n)
    gpu_dest = gpuarray.to_gpu(np.zeros(n, dtype=np.int32))
    dims = np.array([n,k],dtype=np.int32)

    # optimize design ...
    grid_design, block_design = util.tune_sfm(n, k, cu_func.num_regs, logged)

    shared_mem = 4*(block_design[0]*block_design[1] + 2*block_design[1])

    cu_func(gpu_densities, gpu_random, gpu_dest, dims[0], dims[1],
            block=block_design, grid=grid_design, shared=shared_mem)

    if return_gpuarray:
        return gpu_dest
    else:
        return gpu_dest.get()


