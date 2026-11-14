"""
The bond market: bonds are in zero net supply, so the bond price q must make households' total
savings equal the supply B:
    S(q) = sum over individual states of mu(b, y) b'(b, y; q) = B.
Households' choices are given at every price of a grid of candidate prices, so S is known on that
grid; the clearing price is found between the two grid prices whose savings straddle B, by linear
interpolation (the method of the SRL tutorial code, srl.utils.safe_linalg.find_clearing_point).
"""

import jax
import jax.numpy as jnp

from hamlab.models.huggett import households


def bond_demand(cash, shares_by_price, mu, prices, borrowing_limit, floor):
    """
    Total savings in bonds at every candidate price: cash on hand (J,), consumption shares at each price
    (n_prices, J), distribution mu (J,), prices (n_prices,) -> S (n_prices,).
    """
    c = households.consumption(cash[None, :], shares_by_price, prices[:, None], borrowing_limit, floor)
    return jnp.sum((cash[None, :] - c) * mu[None, :], axis=-1) / prices


def clearing_point(supply, demand):
    """
    The two adjacent candidate prices whose savings straddle the supply, and the interpolation
    weights there: (lo, hi, w_lo, w_hi) with supply ~ w_lo demand[lo] + w_hi demand[hi].
    """
    distances = jnp.abs(demand - supply)
    closest_idx = jnp.argmin(distances)
    closest_value = demand[closest_idx]

    def left_case():
        i_lo = jnp.argmin(jnp.where(demand < closest_value, jnp.abs(demand - supply), jnp.inf))
        return i_lo, closest_idx

    def right_case():
        i_hi = jnp.argmin(jnp.where(demand > closest_value, jnp.abs(demand - supply), jnp.inf))
        return closest_idx, i_hi

    i_lo, i_hi = jax.lax.cond(supply < closest_value, left_case, right_case)
    i_lo, i_hi = jnp.minimum(i_lo, i_hi), jnp.maximum(i_lo, i_hi)
    denom = demand[i_hi] - demand[i_lo]
    w_hi = jnp.clip((supply - demand[i_lo]) / jnp.where(denom == 0, 1.0, denom), 0.0, 1.0)
    return i_lo, i_hi, 1.0 - w_hi, w_hi
