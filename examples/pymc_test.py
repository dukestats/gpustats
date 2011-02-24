# pylint: disable=E1101

import pymc as pm
import pymc.distributions as dist
import numpy as np
from numpy.linalg import inv, cholesky as chol
import numpy.linalg as L
import numpy.random as rand

from pandas.util.testing import set_trace as st
import matplotlib.pyplot as plt

import pdfs
import util

gen_mean = {
    0 : [0, 5],
    1 : [-10, 0],
    2 : [-10, 10]
}

gen_sd = {
    0 : [0.5, 0.5],
    1 : [.5, 1],
    2 : [1, .25]
}

gen_corr = {
    0 : 0.5,
    1 : -0.5,
    2 : 0
}

def generate_data(n=1e5, k=2, ncomps=3, seed=1):
    rand.seed(seed)
    data_concat = []
    labels_concat = []

    for j in range(ncomps):
        mean = gen_mean[j]
        sd = gen_sd[j]
        corr = gen_corr[j]

        cov = np.empty((k, k))
        cov.fill(corr)
        cov[np.diag_indices(k)] = 1
        cov *= np.outer(sd, sd)

        rvs = pm.rmv_normal_cov(mean, cov, size=n)

        data_concat.append(rvs)
        labels_concat.append(np.repeat(j, n))

    return (np.concatenate(labels_concat),
            np.concatenate(data_concat, axis=0))

N = int(2e4) # n data points per component
K = 2 # ndim
ncomps = 3 # n mixture components

true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)

def plot_2d_mixture(data, labels):
    plt.figure(figsize=(10, 10))
    colors = 'bgr'
    for j in np.unique(labels):
        x, y = data[labels == j].T
        plt.plot(x, y, '%s.' % colors[j], ms=1)

# priors, fairly vague


sigma0 = np.diag([1., 1.])
prior_mean = data.mean(0)
c0 = np.cov(data.T)

alpha0 = np.ones(3.) / 3

# shared hyperparameter
theta_tau = pm.Wishart('theta_tau', n=4, Tau=L.inv(sigma0))

thetas = []
covs = []
for j in range(ncomps):
    # need a hyperparameter for degrees of freedom
    cov = pm.InverseWishart('C_%d' % j, n=4, C=c0)
    theta = pm.MvNormal('theta_%d' % j, mu=prior_mean,
                        tau=theta_tau)

    thetas.append(theta)
    covs.append(cov)

weights = pm.Dirichlet('weights', theta=alpha0)
labels = pm.Categorical('labels', p=weights, size=len(data))

@pm.stochastic(observed=True)
def mixture(value=data, thetas=thetas, covs=covs, labels=labels,
            weights=weights):


    # need to derive from dirichlet draw
    mix_weights = np.r_[weights, 1 - weights.sum()]
    likes = pdfs.mvnpdf(data, thetas, covs)

    # n = len(data)
    # upwrapped = likes.ravel('F').take(labels * n + np.arange(n))
    # loglike2 = (upwrapped * mix_weights[labels]).sum()

    loglike = 0
    for j, (theta, cov) in enumerate(zip(thetas, covs)):
        this_like = likes[:, j][labels == j].sum()
        loglike += mix_weights[j] * this_like
        # data = value[labels == j]
        # loglike += mix_weights[j] * pm.mv_normal_cov_like(data, theta, cov)

    return loglike

def plot_thetas(sampler):
    plot_2d_mixture(data, true_labels)

    def plot_theta(i):
        x, y = sampler.trace('theta_%d' % i)[:].T
        plt.plot(x, y, 'k.')

    for i in range(3):
        plot_theta(i)

sampler = pm.MCMC({'mixture' : mixture,
                   'thetas' : thetas,
                   'theta_tau' : theta_tau,
                   'covs' : covs,
                   'weights' : weights,
                   'labels' : labels})

sampler.sample(iter=5000, burn=1000, thin=10)
