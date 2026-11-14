"""
Grids of individual states and actions.

Individual states with two dimensions (e.g. wealth and income) are stored as one flat index,
state = first * n_second + second, so that a distribution is a vector over all grid points.
"""

import jax.numpy as jnp
import numpy as np

from mfax.envs.pushforward.utils import distribute


def midpoints(n: int):
    """The centres of n equal bins of [0, 1] (a grid of consumption shares)."""
    return 0.5 * (jnp.linspace(0, 1, n + 1)[:-1] + jnp.linspace(0, 1, n + 1)[1:])


def pivots(lower):
    """Shifts that make geometric spacing possible from a lower bound of 0: max(|lower|, 0.25)."""
    return jnp.maximum(jnp.abs(lower), 0.25)


def geometric(lower, upper, n: int, pivot):
    """n points from lower to upper, geometrically spaced after a shift by `pivot` (denser near lower)."""
    return jnp.clip(jnp.geomspace(lower + pivot, upper + pivot, n) - pivot, a_min=0)


def asset_grid(lower, upper, n: int):
    """n points from lower to upper, geometrically spaced after a shift (denser near lower), computed
    in float64 with the first point exactly at `lower`. This is the grid of the SRL tutorial code
    (srl.utils.discretize.discrete_assets)."""
    pivot = np.maximum(np.abs(lower), 0.25)
    grid = np.geomspace(lower + pivot, upper + pivot, n) - pivot
    grid[0] = lower
    return jnp.asarray(grid)


def product(*axes):
    """Every combination of points of the given 1-D grids, the last axis varying fastest: (prod n_i, n_axes)."""
    return jnp.stack([g.ravel() for g in jnp.meshgrid(*axes, indexing="ij")], axis=1)


def flat_index(first_idx, second_idx, num_states):
    """Flat index of a two-dimensional grid point (indices clipped to the grid)."""
    first_idx = jnp.clip(jnp.int32(first_idx), 0, num_states[0] - 1)
    second_idx = jnp.clip(jnp.int32(second_idx), 0, num_states[1] - 1)
    return first_idx * num_states[1] + second_idx


def split_index(state_idx, num_states, n_states: int):
    """The two indices of a flat grid index."""
    state_idx = jnp.clip(jnp.int32(state_idx), 0, n_states - 1)
    return state_idx // num_states[1], state_idx % num_states[1]


def lottery(grid, x):
    """x split between its two neighbouring points of a 1-D grid, with linear weights so that the mean is
    x: (indices (2,), probabilities (2,)). Differentiable in x almost everywhere."""
    return distribute(grid, x)


def interpolation_weights(x, grid):
    """Linear interpolation of x on a sorted 1-D grid: (lower index, upper index, lower weight, upper
    weight), with x ~ w_lo grid[lo] + w_hi grid[hi], saturating at the ends of the grid. Used as a
    lottery between two grid points, like `lottery`; this variant is the SRL tutorial code's
    (srl.utils.safe_linalg.interp_two_point_nonuniform), which the Huggett model reproduces."""
    n = grid.shape[0]
    i_hi = jnp.clip(jnp.searchsorted(grid, x, side="right"), 1, n - 1).astype(jnp.int32)
    i_lo = (i_hi - 1).astype(jnp.int32)
    x_lo, x_hi = grid[i_lo], grid[i_hi]
    w_hi = jnp.clip((x - x_lo) / jnp.maximum(x_hi - x_lo, 1e-12), 0.0, 1.0)
    return i_lo, i_hi, 1.0 - w_hi, w_hi


def unit_scale(x, lower, upper, pivot):
    """Each dimension of x mapped to [0, 1], geometrically as the grids above (linearly where a
    dimension's bounds make the geometric map ill-conditioned)."""
    ratio = (upper + pivot) / (lower + pivot)
    ratio_is_small = jnp.isclose(ratio, 1.0)
    frac = jnp.clip((x + pivot) / (lower + pivot), 1e-12, None)
    geom_u = jnp.log(frac) / jnp.where(ratio_is_small, 1.0, jnp.log(ratio))
    width = upper - lower
    linear_u = jnp.where(jnp.isclose(upper, lower), 0.0, (x - lower) / jnp.where(width == 0, 1.0, width))
    return jnp.clip(jnp.where(ratio_is_small, linear_u, geom_u), 0.0, 1.0)
