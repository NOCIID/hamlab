"""
Households: a household with bonds b and idiosyncratic income y receives y z (z: aggregate income), so
its cash on hand is b + y z. It consumes a share of its cash on hand above the borrowing limit and buys
bonds with the rest at price q:
    c + q b' = b + y z,     b' >= borrowing limit,
with consumption kept between a floor and the amount that leaves b' just above the borrowing limit.
Utility is CRRA, (c^(1 - sigma) - 1) / (1 - sigma) (log c when sigma = 1).
"""

import jax
import jax.numpy as jnp


def cash_on_hand(bonds, income, z):
    return bonds + z * income


def consumption(cash, share, price, borrowing_limit, floor):
    """The share of cash on hand above the borrowing limit that is consumed, kept between the floor and
    the consumption that leaves next-period bonds at the borrowing limit plus the floor."""
    return jnp.clip((cash - borrowing_limit) * share, floor, cash - (borrowing_limit + floor) * price)


def next_bonds(cash, c, price):
    """Bonds bought with what is not consumed."""
    return (cash - c) / price


@jax.jit
def utility(c, sigma):
    """CRRA utility, consumption floored at 1e-5 (sigma is traced, as in the SRL tutorial code)."""
    c = jnp.maximum(c, 1e-5)
    return jax.lax.cond(sigma == 1, lambda x: jnp.log(x), lambda x: (x ** (1 - sigma) - 1) / (1 - sigma), c)
