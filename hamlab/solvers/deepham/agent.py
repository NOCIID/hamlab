"""
A DeepHAM policy behind MFAX's agent interface, so that MFAX's evaluation code (sequence.py,
exploitability.py) and experiments/gpdp/step4_evaluate.py run it unchanged (as gpdp.policy.GPDPAgent).

MFAX's agents are called with observed prices only, but DeepHAM's policy takes moments of the whole
distribution. The agent therefore keeps the distribution in its hidden state, as GPDPAgent keeps a
clock: hidden = (t, mu_t). It starts from the reset distribution and pushes it forward with the grid
probabilities it returns, exactly as env.mf_step does, which is exact because in MFAX the population
always follows the evaluated policy (checked in experiments/deepham/deepham_checks.py).
The aggregate state is rebuilt from the tracked distribution and the observed prices by the model
(state_from_observation).
"""

import jax
import jax.numpy as jnp

from hamlab.solvers.deepham import nets as dn
from hamlab.solvers.gpdp.policy import shares_to_grid_probs


class DeepHAMAgent:
    def __init__(self, env, mf_params: dn.DeepHAMPolicy, grid_mode: str = "split"):
        self.env = env
        self.mf_params = mf_params
        self.grid_mode = grid_mode  # "split": each share between its two neighbouring grid actions; "nearest"

    def init_hidden(self, batch_size: int):
        _, reset_s = self.env.mf_reset(jax.random.PRNGKey(0))  # the reset distribution
        mu0 = jnp.broadcast_to(reset_s.mu, (batch_size, self.env.n_states))
        return jnp.concatenate([jnp.zeros((batch_size, 1)), mu0], axis=-1)

    def aggregate_state(self, aggregate_obs, hidden):
        """The aggregate state (batched) from the observed prices and the hidden state (t, mu)."""
        t, mu = hidden[:, 0], hidden[:, 1:]
        return self.env.state_from_observation(mu, t.astype(jnp.int32), aggregate_obs)

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None, mf_params=None):
        params = self.mf_params if mf_params is None else mf_params
        if done is not None:
            hidden_state = jnp.where(done[:, None], self.init_hidden(hidden_state.shape[0]), hidden_state)
        aggregate_s = self.aggregate_state(aggregate_obs, hidden_state)
        shares, grid = dn.policy_shares(self.env, params, aggregate_s), self.env.params.discrete_actions
        if self.grid_mode == "nearest":
            prob_a = jax.nn.one_hot(jnp.argmin(jnp.abs(shares[..., None] - grid), axis=-1), grid.shape[0])
        else:
            prob_a = shares_to_grid_probs(shares, grid)
        next_mu = jax.vmap(self.env.mf_transition)(aggregate_s.mu, prob_a, aggregate_s)
        return prob_a, jnp.concatenate([hidden_state[:, :1] + 1, next_mu], axis=-1)
