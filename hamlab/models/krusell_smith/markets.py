"""
Markets: households' wealth is the economy's capital and their income levels its labour supply, so
    K = average wealth,     L = average income level
under the distribution mu over individual states (stored flat, wealth-major: see params.py).
"""

import jax.numpy as jnp


def capital(mu, wealth_grid, num_states):
    """K under distributions mu (..., n_states) -> (...)."""
    mu = mu.reshape(*mu.shape[:-1], *num_states)
    return jnp.sum(mu.sum(-1) * wealth_grid, axis=-1)


def labour(mu, income_grid, num_states):
    """L under distributions mu (..., n_states) -> (...)."""
    mu = mu.reshape(*mu.shape[:-1], *num_states)
    return jnp.sum(mu.sum(-2) * income_grid, axis=-1)
