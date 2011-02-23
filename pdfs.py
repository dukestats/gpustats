from numpy.linalg import inv, cholesky as chol
import numpy as np

import testmod
import util

def mvnpdf(data, means, covs):
    '''
    Compute multivariate normal log pdf

    Parameters
    ----------

    Returns
    -------

    '''
    logdets = [np.log(np.linalg.det(c)) for c in covs]
    ichol_sigmas = [inv(chol(c)) for c in covs]

    packed_params = util.pack_params(means, ichol_sigmas, logdets)
    packed_data = util.pad_data(data)
    return testmod.mvn_call(packed_data, packed_params,
                            data.shape[1])
