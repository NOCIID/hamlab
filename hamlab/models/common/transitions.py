"""
Next individual states for households whose state is (wealth, income level): wealth is chosen, income
moves on a chain. The chosen next wealth falls between two grid points and is split between them (a
lottery with the same mean); independently, the income level moves.
"""

import jax.numpy as jnp

from hamlab.models.common.grids import flat_index, interpolation_weights, lottery


def wealth_and_income(next_wealth, income_idx, wealth_grid, num_states, moves, move_probs, idio_noise):
    """
    Income moves by one of `moves` (probabilities `move_probs`, clipped at the ends of the grid) with
    probability idio_noise, and stays otherwise (the Krusell-Smith model).
    Flat indices of the possible next individual states and their probabilities (they sum to 1).
    next_wealth: the chosen wealth (scalar); income_idx: today's income level; num_states:
    (n_wealth, n_income). Returns (indices (2 * len(moves) + 2,), probabilities (same)).
    """
    next_wealth_idxs, wealth_probs = lottery(wealth_grid, next_wealth)

    # --- income moves, with probability idio_noise ---
    moved_wealth_idxs = jnp.repeat(next_wealth_idxs, len(moves))
    moved_income_idxs = jnp.tile(jnp.clip(income_idx + moves, 0, num_states[1] - 1), 2)
    moved_idxs = flat_index(moved_wealth_idxs, moved_income_idxs, num_states)
    moved_probs = jnp.outer(wealth_probs, move_probs * idio_noise).ravel()

    # --- income stays, with probability 1 - idio_noise ---
    stay_idxs = flat_index(next_wealth_idxs, jnp.repeat(income_idx, 2), num_states)
    stay_probs = (1.0 - idio_noise) * wealth_probs

    idxs = jnp.concatenate([moved_idxs, stay_idxs], axis=0).reshape(-1)
    probs = jnp.concatenate([moved_probs, stay_probs], axis=0).reshape(-1)
    return idxs, probs / jnp.where(probs.sum() > 0, probs.sum(), 1.0)


def wealth_and_markov_income(next_wealth, income_idx, wealth_grid, n_income: int, income_transition):
    """
    Income follows a Markov chain (row income_idx of income_transition) (the Huggett model). Flat
    indices (wealth * n_income + income) of the possible next states and their probabilities, both
    (2 n_income,): the lower wealth point with every income level, then the upper one.
    """
    lo, hi, w_lo, w_hi = interpolation_weights(next_wealth, wealth_grid)
    income = jnp.arange(n_income, dtype=jnp.int32)
    idxs = jnp.concatenate([lo * n_income + income, hi * n_income + income])
    probs = jnp.concatenate([w_lo * income_transition[income_idx], w_hi * income_transition[income_idx]])
    return idxs, probs
