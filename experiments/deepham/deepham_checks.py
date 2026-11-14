"""
Checks of the DeepHAM implementation (hamlab/solvers/deepham/).

1. Window objective = exact value. Along an economy simulated by deepham.train.forward, a window of
   W periods (with the same aggregate shocks) plus gamma^W times the exact value at its end must give
   the exact value at its start (core.continuous.path_evaluate), mu-weighted. Checked for a window
   inside the episode and for one that reaches the terminal period (no continuation).
2. Fictitious play. The gradient of the policy loss equals the gradient with the population's path
   computed beforehand and held fixed: no gradient flows through the others' behaviour.
3. Evaluation wrapper. Along MFAX's own simulation (sequence.py, grid actions), DeepHAMAgent's tracked
   distribution and recovered z equal the environment's, at every period.
4. Timings of one policy pass and of a short value fit, and a pickle round trip.

    python experiments/deepham/deepham_checks.py
"""

import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.sequence import mf_sequence  # noqa: E402
from hamlab.solvers.deepham import nets as dn  # noqa: E402
from hamlab.solvers.deepham import train as dt  # noqa: E402
from hamlab.core import continuous as gc  # noqa: E402
from hamlab.solvers.deepham.agent import DeepHAMAgent  # noqa: E402
from hamlab.models import make as make_model  # noqa: E402

GAMMA = 0.95


def main():
    env = make_model("krusell_smith")
    T, p = env.params.max_steps_in_episode, env.params
    cfg = dn.Config(n_fm=0, n_gm=1, use_time=True, T=T)
    key = jax.random.PRNGKey(0)
    k_v, k_net, k_vnet, k_sim = jax.random.split(key, 4)

    # --- a policy to test with: random network, normalizer from constant-share data ---
    seq0, _, V0 = dt.value_data(env, GAMMA, 0.15, k_v, 4)
    norm = dt.make_normalizer(env, seq0, V0.reshape(T, 4, -1))
    net = dn.init_net(k_net, cfg)
    policy = dn.DeepHAMPolicy(net, norm, cfg)
    vnets = jax.vmap(lambda k: dn.init_net(k, cfg))(jax.random.split(k_vnet, 3))

    # --- 1. window objective vs exact values ---
    E = 4
    seq, shares = dt.forward(env, policy, k_sim, E)
    V = gc.path_evaluate(env, seq, shares, GAMMA)  # (T, E, n)
    W = 29
    for t0 in (10, 110):
        a0 = jax.tree.map(lambda x: x[t0], seq)
        n_in = min(W, T - 1 - t0)  # innovations available inside the simulated path
        eps = (seq.z[t0 + 1 : t0 + 1 + n_in] - p.rho * seq.z[t0 : t0 + n_in]) / p.nu
        eps = jnp.concatenate([eps, jnp.zeros((W - n_in, E))], axis=0)  # after the end: unused
        end = min(t0 + W, T - 1)
        value_fn = lambda a, end=end: V[end]  # noqa: E731  (ignored when the window reaches the end)
        got = dt.window_value(env, GAMMA, policy, value_fn, a0, eps)
        want = jnp.sum(seq.mu[t0] * V[t0], axis=-1)
        print(f"1. window t0={t0:3d} W={W}: max rel. diff to the exact value {float(jnp.max(jnp.abs(got / want - 1))):.2e}")

    # --- 2. fictitious play: gradient with the population's path held fixed ---
    a0 = jax.tree.map(lambda x: x[10], seq)
    eps = jax.random.normal(jax.random.PRNGKey(3), (W, E))
    value_fn = lambda a: dn.values(env, cfg, norm, vnets, a)  # noqa: E731

    def loss_ours(net):
        return -dt.window_value(env, GAMMA, dn.DeepHAMPolicy(net, norm, cfg), value_fn, a0, eps).mean()

    def fixed_path(net):  # the population's path under `net`, computed without gradient
        def _fwd(a, e):
            s = dn.policy_shares(env, dn.DeepHAMPolicy(net, norm, cfg), a)
            return jax.vmap(lambda aa, ss, ee: dt.population_step(env, aa, ss, ee))(a, s, e), a

        last, path = jax.lax.scan(_fwd, a0, eps)
        return jax.lax.stop_gradient((last, path))

    last, path = fixed_path(net)

    def loss_fixed(net):
        pol = dn.DeepHAMPolicy(net, norm, cfg)
        v = value_fn(last)

        def _bwd(v_next, a):
            s = dn.policy_shares(env, pol, a)
            return jax.vmap(lambda aa, ss, vn: gc.q_values(env, ss, aa, vn, GAMMA))(a, s, v_next), None

        v0, _ = jax.lax.scan(_bwd, v, path, reverse=True)
        return -jnp.sum(a0.mu * v0, axis=-1).mean()

    g1, g2 = jax.grad(loss_ours)(net), jax.grad(loss_fixed)(net)
    diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)))
    scale = max(float(jnp.max(jnp.abs(a))) for a in jax.tree.leaves(g1))
    print(f"2. gradient vs population held fixed: max abs diff {diff:.2e} (largest gradient {scale:.2e}); "
          f"all finite: {all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(g1))}")

    # --- 3. the evaluation wrapper tracks MFAX's distribution ---
    agent = DeepHAMAgent(env, policy)
    traj = mf_sequence(jax.random.PRNGKey(7), env, agent, "states", 4, T)
    hid = traj.aggregate_hidden  # (T, E, 1 + n): the hidden state fed to the agent at each period
    mu_err = float(jnp.max(jnp.abs(hid[1:, :, 1:] - traj.aggregate_s.mu[1:])))
    t_err = float(jnp.max(jnp.abs(hid[1:, :, 0] - traj.aggregate_s.time[1:])))
    z_rec = jax.vmap(lambda o, h: agent.aggregate_state(o, h).z)(traj.aggregate_obs[1:], hid[1:])
    z_err = float(jnp.max(jnp.abs(z_rec - traj.aggregate_s.z[1:])))
    print(f"3. agent vs MFAX: max |mu diff| {mu_err:.2e}, max |t diff| {t_err:.0f}, max |z diff| {z_err:.2e}")

    # --- 4. timings and pickle ---
    pdata = dt.policy_data(seq0, jax.random.PRNGKey(5), 32)
    B = 32
    lossf = jax.jit(jax.value_and_grad(
        lambda net, a, e: -dt.window_value(env, GAMMA, dn.DeepHAMPolicy(net, norm, cfg), value_fn, a, e).mean()
    ))
    a = jax.tree.map(lambda x: x[:B], pdata)
    e = jax.random.normal(jax.random.PRNGKey(6), (W, B))
    jax.block_until_ready(lossf(net, a, e))
    t0 = time.perf_counter()
    for _ in range(5):
        jax.block_until_ready(lossf(net, a, e))
    print(f"4. one policy step (B={B}, W={W}): {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")
    _, data_v, Vv = dt.value_data(env, GAMMA, 0.15, k_v, 36)
    args_fit = dict(n_epochs=2, batch=128, lr=1e-4, n_valid=512)
    jax.block_until_ready(dt.fit_values(env, cfg, norm, vnets, data_v, Vv, jax.random.PRNGKey(8), **args_fit))
    t0 = time.perf_counter()
    jax.block_until_ready(dt.fit_values(env, cfg, norm, vnets, data_v, Vv, jax.random.PRNGKey(9), **args_fit))
    per_epoch = (time.perf_counter() - t0) / 2
    print(f"   value fit: {per_epoch:.2f} s per epoch for the ensemble ({per_epoch * 200:.0f} s for 200 epochs)")
    agent2 = pickle.loads(pickle.dumps(DeepHAMAgent(env, jax.device_get(policy))))
    s1 = dn.policy_shares(env, agent2.mf_params, a)
    print(f"   pickle round trip: max share diff {float(jnp.max(jnp.abs(s1 - dn.policy_shares(env, policy, a)))):.1e}")


if __name__ == "__main__":
    main()
