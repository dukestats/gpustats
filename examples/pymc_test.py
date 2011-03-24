# pylint: disable=E1101

import pymc as pm
import pymc.distributions as dist
import numpy as np
from numpy.linalg import inv, cholesky as chol
import numpy.linalg as L
import numpy.random as rand

import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Generate MV normal mixture

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

group_weights = [0.6, 0.3, 0.1]

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

        num = int(n * group_weights[j])
        rvs = pm.rmv_normal_cov(mean, cov, size=num)

        data_concat.append(rvs)
        labels_concat.append(np.repeat(j, num))

    return (np.concatenate(labels_concat),
            np.concatenate(data_concat, axis=0))

N = int(1e5) # n data points per component
K = 2 # ndim
ncomps = 3 # n mixture components

true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)

def plot_2d_mixture(data, labels):
    plt.figure(figsize=(10, 10))
    colors = 'bgr'
    for j in np.unique(labels):
        x, y = data[labels == j].T
        plt.plot(x, y, '%s.' % colors[j], ms=2)


def plot_thetas(sampler):
    plot_2d_mixture(data, true_labels)

    def plot_theta(i):
        x, y = sampler.trace('theta_%d' % i)[:].T
        plt.plot(x, y, 'k.')

    for i in range(3):
        plot_theta(i)

#-------------------------------------------------------------------------------
# set up PyMC model

# priors, fairly vague
prior_mean = data.mean(0)
sigma0 = np.diag([1., 1.])
prior_cov = np.cov(data.T)

# shared hyperparameter?
# theta_tau = pm.Wishart('theta_tau', n=4, Tau=L.inv(sigma0))

# df = pm.DiscreteUniform('df', 3, 50)

thetas = []
taus = []
for j in range(ncomps):
    # need a hyperparameter for degrees of freedom?
    tau = pm.Wishart('C_%d' % j, n=3, Tau=inv(prior_cov))
    theta = pm.MvNormal('theta_%d' % j, mu=prior_mean, tau=inv(2 * prior_cov))

    thetas.append(theta)
    taus.append(tau)

alpha0 = np.ones(3.) / 3
weights = pm.Dirichlet('weights', theta=alpha0)
# labels = pm.Categorical('labels', p=weights, size=len(data))

from pandas.util.testing import set_trace as st
import pdfs
import util

def mixture_loglike(data, thetas, covs, labels):

    n = len(data)
    likes = pdfs.mvnpdf(data, thetas, covs)
    loglike = likes.ravel('F').take(labels * n + np.arange(n)).sum()

    if np.isnan(loglike):
        return -1e300

    return loglike

    if np.isnan(likes).any():
        loglike = 0.
        for j, (theta, cov) in enumerate(zip(thetas, covs)):
            this_data = data[labels == j]
            ch = chol(cov)
            loglike += pm.mv_normal_chol_like(this_data, theta, ch)

        return loglike

def mixture_loglike2(data, thetas, taus, weights):

    n = len(data)

    covs = [inv(tau) for tau in taus]

    likes = pdfs.mvnpdf(data, thetas, covs)
    loglike = (likes * weights).sum()

    # loglike = likes.ravel('F').take(labels * n + np.arange(n)).sum()

    if np.isnan(loglike):
        st()
        return -1e300

    return loglike

    if np.isnan(likes).any():
        loglike = 0.
        for j, (theta, cov) in enumerate(zip(thetas, covs)):
            this_data = data[labels == j]
            loglike += pm.mv_normal_chol_like(this_data, theta, ch)

        return loglike

@pm.deterministic
def adj_weights(weights=weights):
    return np.sort(np.r_[weights, 1 - weights.sum()])

@pm.stochastic(observed=True)
def mixture(value=data, thetas=thetas, taus=taus, weights=adj_weights):
    return mixture_loglike2(value, thetas, taus, weights)

sampler = pm.MCMC(locals())

sampler.sample(iter=3000, burn=100, tune_interval=100, thin=10)

