"""
Python versions of functions for testing purposes etc.
"""
import numpy as np
import pymc.distributions as pymc_dist

def python_mvnpdf(data, means, covs):
    pdf_func = pymc_dist.mv_normal_cov_like

    results = []
    for i, datum in enumerate(data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs)))
