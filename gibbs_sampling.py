#%%
import numpy as np
from scipy.stats import norm
import seaborn as sns

# x0|x1 ~ N(µ0 + (∑01 / ∑11)(x1 - µ1) , ∑00 - ∑01^2/∑11)
# x1|x0 ~ N(µ1 + (∑10 / ∑00)(x0 - µ0) , ∑11 - ∑10^2/∑00)
def cond_sampler(i, x, mean, cov):
    mu = mean[i] + (cov[i, 1-i] / cov[1-i, 1-i]) * (x[1-i] - mean[1-i])
    sigma = np.sqrt(cov[i, i] - (cov[i, 1-i] ** 2) / cov[1-i, 1-i])
    return norm.rvs(mu, sigma)

def step(point, mean, cov):
    x0, x1 = point
    x0 = cond_sampler(0, np.array([x0, x1]), mean, cov)
    x1 = cond_sampler(1, np.array([x0, x1]), mean, cov)
    return np.array([x0, x1])

def gibbs_sampler(start_point, mean, cov):
    sample = np.array([*start_point])
    while True:
        sample = step(sample, mean, cov)
        yield sample

#%% initialize constants of the distribution from which we'll sample
start_point = np.array([0, 0])
mean = np.array([1, 2])
sigma1, sigma2, rho = [1.6, 1.3, 0.434]

covariance_matrix = np.array([
    [sigma1**2, rho * sigma1 * sigma2],
    [rho * sigma1 * sigma2, sigma2**2]
])

generator = gibbs_sampler(start_point, mean, covariance_matrix)

#%% sample 1e6 samples from the distribution so that the Markov Chain stabilizes
[next(generator) for _ in range(int(1e6))];

#%% plot the distribution of 1e5 samples
a = np.array([next(generator) for i in range(int(1e5))])
sns.kdeplot(a[:,0], a[:, 1])
