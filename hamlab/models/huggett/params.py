"""Calibration of the Huggett economy with aggregate risk, and the grids derived from it."""

import math

import jax.numpy as jnp
from flax import struct

from hamlab.models.common import grids
from hamlab.models.common.shocks import tauchen_log_ar1


@struct.dataclass
class HuggettParams:
    # --- preferences ---
    beta: float = 0.96  # discount factor
    sigma: float = 2.0  # CRRA curvature of utility
    consumption_floor: float = 0.001  # minimum consumption (keeps utility finite)

    # --- bonds ---
    borrowing_limit: float = -1.0  # bonds held cannot fall below this
    bond_supply: float = 0.0  # B: bonds are in zero net supply
    bond_max: float = 50.0  # top of the bond grid
    n_bonds: int = 200

    # --- idiosyncratic income y: log y is an AR(1) ---
    rho_y: float = 0.6  # persistence
    sigma_y: float = 0.2  # stationary standard deviation of log y
    n_income: int = 3

    # --- aggregate income z (scales every household's income): log z is an AR(1) ---
    rho_z: float = 0.9
    sigma_z: float = 0.02  # stationary standard deviation of log z
    n_z: int = 50

    # --- candidate bond prices for market clearing ---
    price_min: float = 0.95
    price_max: float = 0.99
    n_prices: int = 20

    # --- horizon: infinite, truncated once beta^T falls below this ---
    truncation_tol: float = 1e-3

    def __post_init__(self):
        """Derived grids: bonds, income and aggregate income chains, prices, every individual state."""
        bond_grid = grids.asset_grid(0, self.bond_max - self.borrowing_limit, self.n_bonds) + self.borrowing_limit
        income_grid, income_transition = tauchen_log_ar1(self.rho_y, self.sigma_y, self.n_income)
        z_grid, z_transition = tauchen_log_ar1(self.rho_z, self.sigma_z, self.n_z)
        object.__setattr__(self, "bond_grid", bond_grid)  # (n_bonds,), dense near the borrowing limit
        object.__setattr__(self, "income_grid", income_grid)  # (n_income,)
        object.__setattr__(self, "income_transition", income_transition)  # (n_income, n_income)
        object.__setattr__(self, "z_grid", z_grid)  # (n_z,)
        object.__setattr__(self, "z_transition", z_transition)  # (n_z, n_z)
        object.__setattr__(self, "price_grid", jnp.linspace(self.price_min, self.price_max, self.n_prices))
        # every individual state (bonds, income), bonds-major: state = bond_idx * n_income + income_idx
        states = jnp.stack([jnp.repeat(bond_grid, self.n_income), jnp.tile(income_grid, self.n_bonds)], axis=1)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "max_steps_in_episode", math.ceil(math.log(self.truncation_tol) / math.log(self.beta)))
