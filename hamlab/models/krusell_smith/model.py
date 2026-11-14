"""
The Krusell-Smith-style economy assembled from its parts (params, households, firms, markets, shocks),
as a hamlab Model and an MFAX push-forward environment.

Timing of one period: households observe the prices (r, w) set by today's distribution and
productivity, choose consumption shares (receiving utility), their wealth and income move to next
period's grid points, productivity draws its shock, and the new distribution sets tomorrow's prices.
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from hamlab.models.base import Model
from hamlab.models.common import grids, transitions
from hamlab.models.krusell_smith import firms, households, markets, shocks
from hamlab.models.krusell_smith.params import KrusellSmithParams
from mfax.envs.pushforward.base import PushforwardAggregateState


@struct.dataclass
class KrusellSmithState(PushforwardAggregateState):
    """Distribution mu over the grid, productivity z, date, and the prices they imply."""

    interest_rate: jax.Array
    wage: jax.Array


class KrusellSmith(Model):
    name = "krusell_smith"

    def __init__(self, params: KrusellSmithParams = None):
        super().__init__(params if params is not None else KrusellSmithParams())

    # --- spaces ---
    @property
    def is_partially_observable(self) -> bool:
        return True

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=jnp.float32)

    @property
    def n_actions(self) -> int:
        return self.params.discrete_n_actions

    @property
    def discrete_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.n_actions)

    @property
    def individual_s_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    # --- one household ---
    def _share(self, action):
        """A grid action (index) or a continuous share (array of shape (1,)) as a share in [0, 1]."""
        if action.ndim == 0:
            return self.params.discrete_actions[action]
        return jnp.clip(action.squeeze(), 0.0, 1.0)

    def _single_step(self, state, action, aggregate_s):
        """Next individual state before the income move: (next wealth, same income)."""
        cash = households.cash_on_hand(state[0], state[1], aggregate_s.interest_rate, aggregate_s.wage)
        return jnp.array([households.next_wealth(cash, action), state[1]])

    def _single_reward(self, state, action, aggregate_s, next_aggregate_s):
        cash = households.cash_on_hand(state[0], state[1], aggregate_s.interest_rate, aggregate_s.wage)
        return households.rewards(cash, action, self.params.sigma)

    def _single_pushforward_step(self, state_idx, action, aggregate_s):
        """Next states of one individual state and their probabilities."""
        assert state_idx.ndim == 0, "state_idx must be a scalar"
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"
        p = self.params
        next_wealth = self._single_step(p.states[state_idx], self._share(action), aggregate_s)[0]
        _, income_idx = grids.split_index(state_idx, p.num_states, p.n_states)
        return transitions.wealth_and_income(
            next_wealth, income_idx, p.discrete_states[0], p.num_states, p.idio_atoms, p.idio_atoms_probs, p.idio_noise
        )

    def _single_pushforward_reward(self, state_idx, action, aggregate_s, next_aggregate_s):
        assert state_idx.ndim == 0, "state_idx must be a scalar"
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"
        return self._single_reward(self.params.states[state_idx], self._share(action), aggregate_s, next_aggregate_s)

    # --- prices, shocks, summary (the Model interface) ---
    def prices(self, mu, z):
        p = self.params
        K = markets.capital(mu, p.discrete_states[0], p.num_states)
        L = markets.labour(mu, p.discrete_states[1], p.num_states)
        return firms.prices(z, K, L, p.cobb_douglas_alpha)

    def prices_from_aggregates(self, z, K, L):
        return firms.prices(z, K, L, self.params.cobb_douglas_alpha)

    def average_income(self, mu):
        return markets.labour(mu, self.params.discrete_states[1], self.params.num_states)

    def next_shock(self, z, eps):
        return shocks.next_productivity(z, eps, self.params)

    def next_shock_moments(self, z):
        return shocks.productivity_moments(z, self.params)

    def summary(self, aggregate_s):
        K = markets.capital(aggregate_s.mu, self.params.discrete_states[0], self.params.num_states)
        return jnp.stack([jnp.asarray(aggregate_s.z, dtype=K.dtype), K], axis=-1)

    def state_from_observation(self, mu, time, obs):
        """z is recovered from the interest rate, given K and L of the known distribution."""
        r, w = obs[:, 0], obs[:, 1]
        K, L = mu @ self.params.states[:, 0], mu @ self.params.states[:, 1]
        a = self.params.cobb_douglas_alpha
        z = jnp.log(r / (a * (L / (K + 1e-6)) ** (1 - a)))
        return KrusellSmithState(mu=mu, z=z, time=time, interest_rate=r, wage=w)

    def get_shared_obs(self, aggregate_s):
        """What households observe: the prices (r, w)."""
        return jnp.array([aggregate_s.interest_rate, aggregate_s.wage])

    # --- the economy over time (MFAX environment) ---
    def mf_reset_env(self, key):
        """Uniform distribution over the grid, z = 0, period 0."""
        mu_0 = jnp.ones(self.params.n_states) / self.params.n_states
        z = 0.0
        interest_rate, wage = self.prices(mu_0, z)
        aggregate_s = KrusellSmithState(mu=mu_0, z=z, time=0, interest_rate=interest_rate, wage=wage)
        return self.get_shared_obs(aggregate_s), aggregate_s

    def mf_step_env(self, key, aggregate_s, prob_a):
        """One period (see the module docstring); prob_a: (n_states, n_actions) grid-action probabilities."""
        next_mu = self.mf_transition(aggregate_s.mu, prob_a, aggregate_s)
        next_z = self.next_shock(aggregate_s.z, jax.random.normal(key))
        next_interest_rate, next_wage = self.prices(next_mu, next_z)
        next_time = aggregate_s.time + 1
        next_aggregate_s = KrusellSmithState(
            mu=next_mu, z=next_z, time=next_time, interest_rate=next_interest_rate, wage=next_wage
        )
        terminated = self.is_terminal(next_time)
        truncated = self.is_truncated(next_time)
        mat_r_step, mat_r_term = self.mf_reward(aggregate_s, next_aggregate_s)
        mat_r = jax.lax.select(terminated, mat_r_term, mat_r_step)
        return (
            jax.lax.stop_gradient(self.get_shared_obs(next_aggregate_s)),
            jax.lax.stop_gradient(next_aggregate_s),
            jax.lax.stop_gradient(mat_r),
            jax.lax.stop_gradient(terminated),
            jax.lax.stop_gradient(truncated),
        )

    def is_truncated(self, time):
        return jnp.array(0)

    def is_terminal(self, time):
        return jnp.array(time >= self.params.max_steps_in_episode)

    # --- inputs of MFAX's policy networks ---
    def normalize_obs(self, shared_obs, normalize_obs=False):
        return jax.lax.select(normalize_obs, shared_obs, shared_obs.astype(jnp.float32))

    def normalize_individual_s(self, individual_states, normalize_states=False):
        """Each dimension of the individual state mapped to [0, 1], geometrically as the grid."""
        if not normalize_states:
            return individual_states
        D = self.params.pivots.shape[0]
        assert individual_states.shape[-1] == D, f"expected last dimension to be {D}, got {individual_states.shape[-1]}"
        p = self.params
        return grids.unit_scale(individual_states, p.lower_bound, p.upper_bound, p.pivots)
