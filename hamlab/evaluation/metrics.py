"""
Consumption-equivalent (CE) measures, shared by every solver's evaluation.

Raw value ratios explode when values are close to zero. Instead, each value is turned into the
constant consumption c_bar that gives the same lifetime utility: with CRRA utility (no additive shift),
V = const * u(c_bar), so  c_bar_a / c_bar_b = (V_a / V_b)^(1 / (1 - sigma))  (sigma = 2: V_b / V_a).
The functions keep their inputs' precision (float32 arrays stay float32), so the evaluation scripts'
numbers are unchanged by using them.
"""

import numpy as np


def ce_ratio(v_a, v_b, sigma):
    """c_bar_a / c_bar_b: the ratio of the constant consumptions that give values v_a and v_b."""
    return (v_a / v_b) ** (1 / (1 - sigma))


def ce_gain_pct(v_hi, v_lo, sigma):
    """Consumption-equivalent gain of value v_hi over value v_lo, in %."""
    return 100 * (ce_ratio(v_hi, v_lo, sigma) - 1)


def mu0_weighted(v, mu0):
    """Values (E, n_states) weighted by the initial distribution mu0 (E, n_states), averaged over
    economies: the weighting of MFAX's exploitability."""
    return (np.asarray(v) * mu0).sum(-1).mean()


def ce_relerr_per_period(V_pol, V_br, mu, sigma):
    """|c_bar_policy / c_bar_BR - 1| averaged over individual states with weights mu_t (where households
    are): (T, E) for values and distributions (T, E, n_states)."""
    mu = np.asarray(mu)
    return (mu / mu.sum(-1, keepdims=True) * np.abs(ce_ratio(V_pol, V_br, sigma) - 1)).sum(-1)
