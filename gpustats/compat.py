"""
Python versions of functions for testing purposes etc.
"""
import numpy as np
import pymc.distributions as pymc

def python_mvnpdf(data, means, covs, k):
    import pymc.distributions as pymc_dist
    actual_data = data[:, :k]

    pdf_func = pymc_dist.mv_normal_cov_like

    results = []
    for i, datum in enumerate(actual_data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs)))
