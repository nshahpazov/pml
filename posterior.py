#%%
import numpy as np
from scipy.special import beta
from scipy.stats import norm
import seaborn as sns

x = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0])
x = np.array([1])

def posterior(π, x):
    n = (x == 1).sum()
    m = (x == 0).sum()
    return (π**n * (1-π)**m) / beta(n + 1, m + 1)

π = np.linspace(0, 1, 100)
posteriors = posterior(π, x)

sum(posteriors)