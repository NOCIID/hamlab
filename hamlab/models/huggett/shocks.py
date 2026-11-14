"""
Shocks, both discretized log AR(1) processes (common.shocks.tauchen_log_ar1):
- aggregate income z, a 50-state Markov chain; each economy starts in a uniformly drawn state;
- each household's idiosyncratic income y, a 3-state Markov chain (params.income_transition).
"""

import jax


def initial_aggregate_state(key, params):
    """The aggregate income state of a new economy, uniform over the chain's states."""
    return jax.random.choice(key, params.n_z)


def next_aggregate_state(key, z_idx, params):
    """Tomorrow's aggregate income state, drawn from row z_idx of the transition matrix."""
    return jax.random.choice(key, params.n_z, p=params.z_transition[z_idx])
