"""
Shocks:
- aggregate productivity z follows an AR(1), z' = rho z + nu eps with eps ~ N(0, 1), from z_0 = 0;
- each household's income level moves one level down or up with probabilities (0.1, 0.1), and stays
  with probability 0.8 (params.idio_atoms, idio_atoms_probs), clipped at the ends of the grid.
"""

from hamlab.models.common.shocks import ar1, ar1_moments


def next_productivity(z, eps, params):
    return ar1(z, eps, params.rho, params.nu)


def productivity_moments(z, params):
    """Mean and variance of tomorrow's productivity given today's z."""
    return ar1_moments(z, params.rho, params.nu)
