#%%
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import approx_fprime
import seaborn as sns

def potential_energy(position, target_dist):
    return -1 * np.log(0.5 * target_dist(position))

def kinetic_energy(momentum):
    return momentum.T @ momentum / 2

def hamiltonian(position, momentum, target_dist):
    return potential_energy(position, target_dist) + kinetic_energy(momentum)

def leapfrog_step(x0, v0, target_dist, step, n_iter):
    gradient = approx_fprime(x0, lambda x: potential_energy(x, target_dist), epsilon=1e-5)
    momentum = v0 - 0.5 * step * gradient
    x = x0 + step * momentum

    for _ in range(n_iter):
        gradient = approx_fprime(x, lambda x: potential_energy(x, target_dist), epsilon=1e-5)
        momentum -= 0.5 * step * gradient
        x += step * momentum

    gradient = approx_fprime(x, lambda x: potential_energy(x, target_dist), epsilon=1e-5)
    momentum -= 0.5 * step * gradient
    return x, momentum

def hmc_step(initial_x, target_dist, step, n_iter):
    momentum = norm.rvs(0, 1, 1)
    position, v = leapfrog_step(initial_x, momentum, target_dist, step, n_iter)

    orig = hamiltonian(initial_x, momentum, target_dist)
    current = hamiltonian(position, v, target_dist)
    p_accept = min(1.0, np.exp(orig - current))

    return position[0] if p_accept > np.random.uniform() else initial_x

def hamiltonian_sample(x, target_dist, step, n_iter):
    while True:
        x = hmc_step(x, target_dist, step, n_iter)
        yield x

#%% generate first 1e3 samples

# if __name__ == "__main__":
#     generator = hamiltonian_sample(0.4, lambda x: norm.pdf(x, 1, 1) / 2 + norm.pdf(x, 5, 1) / 2 ,0.2, 40)
#     [next(generator) for _ in range(int(1e3))];

#%% visualize 1e3 samples
# if __name__ == "__main__":
#     arr = [next(generator) for _ in range(int(1e3))]
#     sns.kdeplot(arr)
