"""
Aggregate and idiosyncratic shock processes:
- an AR(1), z' = rho z + nu eps with eps ~ N(0, 1), for continuous aggregate shocks;
- a log AR(1) discretized into a finite Markov chain (Tauchen 1986), for shocks on a grid.
"""

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def ar1(z, eps, rho, nu):
    """Tomorrow's shock for today's z and a standard normal innovation eps."""
    return rho * z + nu * eps


def ar1_moments(z, rho, nu):
    """Mean and variance of tomorrow's shock given today's z."""
    return rho * z, nu**2


def tauchen_log_ar1(rho, sigma, n: int):
    """
    log X' = rho log X + sqrt(1 - rho^2) sigma eps (sigma: the stationary standard deviation of log X)
    as an n-state Markov chain: log X on an evenly spaced grid over +-3 sigma, transition
    probabilities from the normal distribution over the bins around each grid point. This is the
    discretization of the SRL tutorial code (srl.utils.discretize.discrete_log_ar1).
    Returns (levels exp(log grid) (n,), transition matrix (n, n), rows summing to 1).
    """
    assert 0 < rho < 1 and sigma > 0 and n >= 3
    log_grid = np.linspace(-3 * sigma, 3 * sigma, n)
    step = log_grid[1] - log_grid[0]
    edges = np.append(log_grid - 0.5 * step, log_grid[-1] + 0.5 * step)
    P = np.zeros((n, n))
    for i, x in enumerate(log_grid):
        probs = np.diff(norm.cdf(edges, loc=rho * x, scale=sigma * np.sqrt(1 - rho**2)))
        probs = np.clip(probs, 1e-10, 1)
        P[i] = probs / probs.sum()
    return jnp.exp(log_grid), jnp.asarray(P)
