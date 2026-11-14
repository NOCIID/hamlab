"""
Firms: a representative firm produces e^z K^alpha L^(1 - alpha) and pays factors their marginal
products, so the interest rate and the wage are
    r = e^z alpha (L / K)^(1 - alpha),     w = e^z (1 - alpha) (K / L)^alpha
(a small constant in the denominators guards against an empty economy).
"""

import jax.numpy as jnp


def prices(z, K, L, alpha):
    """(interest rate, wage) from productivity z, capital K and labour L (elementwise on batches)."""
    interest_rate = jnp.exp(z) * alpha * ((L / (K + 1e-6)) ** (1 - alpha))
    wage = jnp.exp(z) * (1 - alpha) * ((K / (L + 1e-6)) ** alpha)
    return interest_rate, wage
