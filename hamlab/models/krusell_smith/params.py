"""Calibration of the Krusell-Smith-style economy, and the grids derived from it."""

from dataclasses import field

import jax
import jax.numpy as jnp
from flax import struct

from hamlab.models.common import grids


@struct.dataclass
class KrusellSmithParams:
    # --- preferences ---
    sigma: float = 2.0  # CRRA curvature of utility

    # --- technology ---
    cobb_douglas_alpha: float = 0.36  # capital share

    # --- aggregate productivity: z' = rho z + nu eps ---
    rho: float = 0.9  # persistence
    nu: float = 0.03  # standard deviation of the innovation

    # --- idiosyncratic income: one level down / same / up ---
    idio_noise: bool = True  # False: income levels never move
    idio_atoms: jax.Array = jnp.array([-1, 0, 1])  # moves of the income level
    idio_atoms_probs: jax.Array = jnp.array([0.1, 0.8, 0.1])  # their probabilities

    # --- horizon ---
    max_steps_in_episode: int = 128  # periods; in the last one households consume everything

    # --- grids: individual states (wealth, income level), geometric per dimension ---
    lower_bound: jax.Array = jnp.array([0.0, 0.1])  # (wealth, income)
    upper_bound: jax.Array = jnp.array([99, 2.0])
    num_states: tuple[int, int] = (200, 5)

    # --- actions: consumption shares; MFAX's algorithms use this many grid points ---
    discrete_n_actions: int = 20

    # --- MFAX's environment interface (the aggregate shock is always drawn) ---
    common_noise: bool = True
    states: jax.Array = field(default_factory=lambda: jnp.empty((0, 0)))  # filled below

    def __post_init__(self):
        """Derived grids: shares, wealth and income levels, and every individual state."""
        object.__setattr__(self, "discrete_actions", grids.midpoints(self.discrete_n_actions))
        pivots = grids.pivots(self.lower_bound)
        discrete_states = [
            grids.geometric(lb, ub, ns, p)
            for lb, ub, ns, p in zip(self.lower_bound, self.upper_bound, self.num_states, pivots)
        ]
        states = grids.product(*discrete_states)
        object.__setattr__(self, "pivots", pivots)
        object.__setattr__(self, "discrete_states", discrete_states)  # [wealth grid, income grid]
        object.__setattr__(self, "states", states)  # (n_states, 2): (wealth, income) of each state
        object.__setattr__(self, "n_states", len(states))
