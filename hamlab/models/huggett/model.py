"""
The Huggett economy assembled from its parts (params, households, markets, shocks), as a
MarketClearingModel.

Timing of one period, given the households' consumption shares at every candidate bond price:
1. the bond market clears: the price q at which total savings equal the bond supply (markets.py);
2. each household consumes its share at q (interpolated between the two neighbouring candidate
   prices), receives utility, and buys bonds b' (households.py);
3. its bonds move to the two grid points around b', and its income along its chain;
4. aggregate income draws its next state (shocks.py).
"""

import jax
import jax.numpy as jnp
from flax import struct

from hamlab.models.base import MarketClearingModel
from hamlab.models.common import transitions
from hamlab.models.huggett import households, markets, shocks
from hamlab.models.huggett.params import HuggettParams


@struct.dataclass
class HuggettState:
    mu: jax.Array  # distribution over individual states (n_states,)
    z_idx: jax.Array  # aggregate income state (index into params.z_grid)
    time: jax.Array


@struct.dataclass
class Clearing:
    price: jax.Array  # the bond price q at which the market clears
    lo: jax.Array  # the two neighbouring candidate prices (indices into params.price_grid) ...
    hi: jax.Array
    w_lo: jax.Array  # ... and the interpolation weights there
    w_hi: jax.Array
    demand: jax.Array  # total savings at every candidate price (n_prices,)


class Huggett(MarketClearingModel):
    name = "huggett"

    def __init__(self, params: HuggettParams = None):
        super().__init__(params if params is not None else HuggettParams())

    @property
    def n_prices(self) -> int:
        return self.params.n_prices

    def aggregate_income(self, aggregate_s):
        return self.params.z_grid[aggregate_s.z_idx]

    def is_terminal(self, time):
        """The infinite horizon is truncated after params.max_steps_in_episode periods."""
        return jnp.array(time >= self.params.max_steps_in_episode)

    def starting_policy(self):
        """
        A closed-form consumption rule, as a starting point for solvers: the consumption share, at every
        individual state, candidate price and aggregate income state, of a household that expects a
        constant gross interest rate R = 1 / q forever, R - (beta R)^(1 / sigma) times its wealth above
        the borrowing limit plus the present value of a unit income, clipped to [0.001, 1] (the initial
        guess of the SRL tutorial code). Returns (n_states, n_prices, n_z).
        """
        p = self.params
        b = p.bond_grid[:, None, None, None]
        y = p.income_grid[None, :, None, None]
        q = p.price_grid[None, None, :, None]
        z = p.z_grid[None, None, None, :]
        wealth = b + y * z - p.borrowing_limit
        R = 1.0 / q
        c = jnp.maximum((R - (p.beta * R) ** (1.0 / p.sigma)) * (wealth + 1.0 / (R - 1.0)), 0.0)
        return jnp.clip(c / wealth, 0.001, 1.0).reshape(self.n_states, p.n_prices, p.n_z)

    def reset(self, key, mu=None):
        """A uniform distribution over the grid (or mu), a uniformly drawn aggregate income state, period 0."""
        mu = jnp.ones(self.n_states) / self.n_states if mu is None else mu
        return HuggettState(mu=mu, z_idx=shocks.initial_aggregate_state(key, self.params), time=0)

    def clear(self, aggregate_s, choices):
        """
        choices: consumption shares of every individual state at every candidate price,
        (n_states, n_prices), for this period's aggregate income. Households take the price as given:
        no gradient flows through the clearing price.
        """
        p = self.params
        cash = households.cash_on_hand(p.states[:, 0], p.states[:, 1], self.aggregate_income(aggregate_s))
        demand = markets.bond_demand(
            cash, jax.lax.stop_gradient(choices).T, aggregate_s.mu, p.price_grid, p.borrowing_limit,
            p.consumption_floor,
        )
        lo, hi, w_lo, w_hi = markets.clearing_point(p.bond_supply, demand)
        price = p.price_grid[lo] * w_lo + p.price_grid[hi] * w_hi
        return Clearing(price=price, lo=lo, hi=hi, w_lo=w_lo, w_hi=w_hi, demand=demand)

    def period(self, aggregate_s, choices, clearing):
        """Every household at the clearing price: next-state indices and probabilities (n_states,
        2 n_income) each, and utilities (n_states,)."""
        p = self.params
        z = self.aggregate_income(aggregate_s)

        def one_state(state_idx):
            bond_idx, income_idx = state_idx // p.n_income, state_idx % p.n_income
            cash = households.cash_on_hand(p.bond_grid[bond_idx], p.income_grid[income_idx], z)
            share = choices[state_idx, clearing.lo] * clearing.w_lo + choices[state_idx, clearing.hi] * clearing.w_hi
            c = households.consumption(cash, share, clearing.price, p.borrowing_limit, p.consumption_floor)
            b_next = households.next_bonds(cash, c, clearing.price)
            idxs, probs = transitions.wealth_and_markov_income(
                b_next, income_idx, p.bond_grid, p.n_income, p.income_transition
            )
            return idxs, probs, households.utility(c, p.sigma)

        return jax.vmap(one_state)(jnp.arange(self.n_states, dtype=jnp.int32))

    def push_forward(self, mu, idxs, probs):
        return jnp.zeros_like(mu).at[idxs].add(probs * mu[:, None])

    def step(self, key, aggregate_s, choices):
        """One period (see the module docstring). Returns (next aggregate state, clearing, utilities)."""
        clearing = self.clear(aggregate_s, choices)
        idxs, probs, utilities = self.period(aggregate_s, choices, clearing)
        next_s = HuggettState(
            mu=self.push_forward(aggregate_s.mu, idxs, probs),
            z_idx=shocks.next_aggregate_state(key, aggregate_s.z_idx, self.params),
            time=aggregate_s.time + 1,
        )
        return next_s, clearing, utilities
