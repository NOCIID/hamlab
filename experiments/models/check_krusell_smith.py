"""
Check that hamlab's Krusell-Smith model reproduces MFAX's endogenous environment exactly (every
comparison is exact equality, not a tolerance):

1. Parameters and grids: individual states, wealth and income grids, grid actions.
2. Episodes: the same reset and the same steps (same keys, same random policies over the grid
   actions), for 8 economies over 130 periods (through an automatic reset): observations, aggregate
   states, rewards and termination flags.
3. Continuous shares (the per-state functions hamlab's solvers call): next states, probabilities and
   rewards of every individual state for random shares; prices from a distribution and from
   aggregates.
4. MFAX's evaluation: the exploitability of saved agents (RSPG, GPDP, DeepHAM) computed on either
   environment.
5. MFAX's own SPG: a few training steps from the same initialization on either environment.

    python experiments/models/check_krusell_smith.py
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hamlab.models import make as make_model  # noqa: E402
from mfax.algos.hsm.exploitability import make_exploitability  # noqa: E402
from mfax.envs import make_env  # noqa: E402

GAMMA = 0.95


def same(a, b):
    """Exact equality of two pytrees (values, shapes and dtypes)."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
    )


def report(name, ok):
    print(f"{'OK  ' if ok else 'FAIL'}  {name}")
    return ok


def main():
    mf = make_env("pushforward/endogenous", common_noise=True)
    ks = make_model("krusell_smith")
    p, q = mf.params, ks.params
    results = []

    # --- 1. parameters and grids ---
    results.append(report("grids and grid actions", same(
        (p.states, p.discrete_states, p.discrete_actions, p.pivots),
        (q.states, q.discrete_states, q.discrete_actions, q.pivots),
    ) and (p.sigma, p.rho, p.nu, p.cobb_douglas_alpha, p.max_steps_in_episode, mf.n_states, mf.n_actions)
        == (q.sigma, q.rho, q.nu, q.cobb_douglas_alpha, q.max_steps_in_episode, ks.n_states, ks.n_actions)))

    # --- 2. episodes with random grid policies ---
    E, T = 8, 130
    key = jax.random.PRNGKey(0)
    k_reset, k_steps, k_pol = jax.random.split(key, 3)

    def episode(env):
        obs, s = jax.vmap(env.mf_reset)(jax.random.split(k_reset, E))

        def _step(carry, inputs):
            s, obs = carry
            k, kp = inputs
            prob_a = jax.nn.softmax(jax.random.normal(kp, (E, env.n_states, env.n_actions)), axis=-1)
            out = jax.vmap(env.mf_step)(jax.random.split(k, E), s, prob_a)
            next_obs, next_s = out[0], out[2]
            return (next_s, next_obs), (obs, s.mu, s.z, s.time, s.interest_rate, s.wage, out[4], out[5], out[6])

        _, traj = jax.lax.scan(_step, (s, obs), (jax.random.split(k_steps, T), jax.random.split(k_pol, T)))
        return traj

    results.append(report(f"episodes: {E} economies x {T} periods (observations, mu, z, time, prices, rewards, flags)",
                          same(episode(mf), episode(ks))))

    # --- 3. continuous shares, per state; prices ---
    _, s = mf.mf_reset(jax.random.PRNGKey(1))
    s_mf = s.replace(mu=jax.random.dirichlet(jax.random.PRNGKey(2), jnp.ones(mf.n_states)), z=jnp.float32(0.03))
    _, s_ks = ks.mf_reset(jax.random.PRNGKey(1))
    s_ks = s_ks.replace(mu=s_mf.mu, z=s_mf.z)
    r, w = mf._prices(s_mf.mu, s_mf.z)
    s_mf, s_ks = s_mf.replace(interest_rate=r, wage=w), s_ks.replace(interest_rate=r, wage=w)
    shares = jax.random.uniform(jax.random.PRNGKey(3), (mf.n_states, 1))
    per_state = lambda env, a: (  # noqa: E731
        jax.vmap(env._single_pushforward_step, in_axes=(0, 0, None))(env.state_indices, shares, a),
        jax.vmap(env._single_pushforward_reward, in_axes=(0, 0, None, None))(env.state_indices, shares, a, a),
    )
    results.append(report("continuous shares: next states, probabilities and rewards of every state",
                          same(per_state(mf, s_mf), per_state(ks, s_ks))))
    K, L = ks.summary(s_ks)[1], ks.average_income(s_ks.mu)
    results.append(report("prices from a distribution, and from (z, K, L)",
                          same(mf._prices(s_mf.mu, s_mf.z), ks.prices(s_ks.mu, s_ks.z))
                          and same(mf._prices(s_mf.mu, s_mf.z), ks.prices_from_aggregates(s_ks.z, K, L))))

    # --- 4. MFAX's exploitability of saved agents, on either environment ---
    runs = {
        "RSPG": "runs/endogenous/hsm_rspg_gp/base_s0/mf_agent_wrapper.pkl",
        "GPDP": "runs/endogenous/gpdp/report_singlevgp_a1_s0/gpdp_agent.pkl",
        "DeepHAM": "runs/endogenous/deepham/deepham_gm1_s4/deepham_agent.pkl",
    }
    for name, path in runs.items():
        with open(ROOT / path, "rb") as f:
            agent = pickle.load(f)
        res = [
            make_exploitability(env=env, agent=agent, state_type="states", gamma=GAMMA, num_envs=16,
                                max_steps_in_episode=env.params.max_steps_in_episode)(jax.random.PRNGKey(777), agent.mf_params)
            for env in (mf, ks)
        ]
        vals = [(float(r.exploitability.exploitability), float(r.exploitability.mean_policy_return)) for r in res]
        results.append(report(f"MFAX exploitability, {name} agent: {vals[0][0]:.6f} vs {vals[1][0]:.6f}",
                              same(res[0].exploitability, res[1].exploitability)))

    # --- 5. MFAX's SPG: training steps from the same initialization ---
    import optax  # noqa: E402
    from flax.training.train_state import TrainState  # noqa: E402
    from mfax.algos.hsm.algos import spg  # noqa: E402
    from mfax.algos.hsm.utils.mf_policy_wrappers import MeanFieldContinuousPolicy  # noqa: E402

    def spg_params(env, n_steps=3):
        a = spg.args(task="endogenous", state_type="states", num_envs=4, discount_factor=GAMMA)
        net = MeanFieldContinuousPolicy(
            state_type="states", num_states=env.n_states, actions=env.params.discrete_actions,
            policy_kwargs=dict(action_dim=1, action_range=(0.0, 1.0), hidden_layer_sizes=(128, 128, 128),
                               activation="relu", state_type="states", num_states=env.n_states),
        )
        states = env.params.states
        rng, k_init, k_act = jax.random.split(jax.random.PRNGKey(0), 3)
        params = net.init(k_init, env.normalize_individual_s(states, True),
                          env.normalize_obs(jnp.ones((a.num_envs, env.obs_dim)), True), k_act)
        ts = TrainState.create(apply_fn=(), params=params,
                               tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-8)))
        step = jax.jit(spg.make_train_step(a, env, net, states))
        (ts, _), _ = jax.lax.scan(step, (ts, rng), None, n_steps)
        return ts.params

    results.append(report("MFAX SPG: 3 training steps give identical parameters", same(spg_params(mf), spg_params(ks))))

    print("ALL IDENTICAL" if all(results) else "SOME DIFFER")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
