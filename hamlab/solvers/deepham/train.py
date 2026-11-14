"""
DeepHAM's algorithm (Han, Yang & E, Quantitative Economics 2026, Algorithm 1; reference code
github.com/frankhan91/DeepHAM: train_KS.py, policy.py, value.py, dataset.py) for the endogenous
mean-field game. The networks are in nets.py.

    start:  a constant share generates the first data; the value ensemble is fitted on it
    then num_steps Adam steps on the policy:
      - policy data: cross-sections (mu_t0, z_t0, t0) of economies simulated under the current policy,
        re-simulated after each pass over them (the reference's epoch_resample = 0)
      - every freq_update_v steps: value data re-simulated under the current policy and the value
        ensemble re-fitted, warm started (the reference's freq_update_v)
      - policy loss (eq. (10)): from a batch of cross-sections, simulate `window` periods with fresh
        aggregate shocks and maximize
            E[ sum_{k < W} gamma^k u(c_k) + gamma^W V(s_W, z_W, Q^V_W, T - t_W) ]
        averaged over the individual states with weights mu_t0 (the paper's agent i = 1 is a random
        household of the cross-section). Windows that reach the end of the episode end with the
        terminal reward instead of V.
Fictitious play: the population's path is simulated with the current policy and wrapped in
stop_gradient, so the optimized household takes the others' current behaviour as given: the
reference code's opt_type "game" (stop_gradient on agents 1..N-1's consumption), and what MFAX's
env.mf_step does for SPG. The household's expected utility is exact over its idiosyncratic income
(the push-forward of core/continuous.py); only the aggregate shocks are sampled.
Value targets (eq. (8)-(9)): lifetime utility of every individual state under the current policy
along simulated economies, exact over idiosyncratic income and to the end of the episode
(core.continuous.path_evaluate; the reference implementation uses one sampled path truncated after 800 periods). The
mu-weighted squared error is the expectation of the reference's per-agent loss over the cross-section.
The settings and how they map to the reference configuration are in config.py.
"""

import json
import os
import pickle
import time
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hamlab.core import continuous as gc
from hamlab.evaluation.metrics import ce_gain_pct, ce_ratio, ce_relerr_per_period
from hamlab.solvers.deepham import nets as dn
from hamlab.solvers.deepham.agent import DeepHAMAgent
from hamlab.solvers.gp_critic.critic import aggregate_features
from mfax.algos.hsm.exploitability import make_exploitability
from hamlab.models import make as make_model
from hamlab.models.base import Model


# --- simulation ---
def _shares(env, policy, aggregate_s):
    """Shares of every state (E, n_states): a DeepHAM policy, or a constant share (the start)."""
    if isinstance(policy, dn.DeepHAMPolicy):
        return dn.policy_shares(env, policy, aggregate_s)
    return jnp.full((aggregate_s.time.shape[0], env.n_states), policy)


@partial(jax.jit, static_argnames=("env", "n_econ"))
def forward(env, policy, key, n_econ):
    """
    Simulate n_econ economies from the reset under the population's policy, with continuous shares;
    the same keys as gpdp.train.forward, so the same key gives the same aggregate shocks.
    Returns (aggregate states (T, E), shares (T, E, n_states)); the last share is unused (terminal).
    """
    T = env.params.max_steps_in_episode
    k_reset, k_steps = jax.random.split(key)
    _, aggregate_s = jax.vmap(env.mf_reset)(jax.random.split(k_reset, n_econ))

    def _step(aggregate_s, k):
        shares = _shares(env, policy, aggregate_s)
        next_aggregate_s, _, _ = jax.vmap(lambda kk, a, s: gc.step(env, kk, a, s))(
            jax.random.split(k, n_econ), aggregate_s, shares
        )
        return next_aggregate_s, (aggregate_s, shares)

    last, (seq, shares) = jax.lax.scan(_step, aggregate_s, jax.random.split(k_steps, T - 1))
    seq = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None]], axis=0), seq, last)
    shares = jnp.concatenate([shares, _shares(env, policy, last)[None]], axis=0)
    return seq, shares


def population_step(env, aggregate_s, shares, eps):
    """core.continuous.step for one economy, with the shock's standard normal innovation eps given."""
    next_mu = gc.transition(env, aggregate_s.mu, shares, aggregate_s)
    next_z = env.next_shock(aggregate_s.z, eps)
    next_interest_rate, next_wage = env.prices(next_mu, next_z)
    return aggregate_s.replace(
        mu=next_mu, z=next_z, time=aggregate_s.time + 1, interest_rate=next_interest_rate, wage=next_wage
    )


def window_value(env, gamma, policy, value_fn, aggregate_s0, eps):
    """
    The policy loss's objective for a batch of B cross-sections (leading axis B): the mu_t0-weighted
    expected utility over W = eps.shape[0] periods plus gamma^W value_fn at the window's end
    (value_fn(aggregate states (B,)) -> (B, n_states)). Periods after the episode's end count zero,
    and a window that reaches it ends with the terminal reward (core.continuous.q_values).
    eps: (W, B) innovations of the aggregate shock. Returns (B,).
    """
    T = env.params.max_steps_in_episode

    def _fwd(aggregate_s, e):
        shares = dn.policy_shares(env, policy, aggregate_s)  # the optimized household: gradient flows
        # next individual states and their probabilities, and rewards, under the household's shares
        # (computed once and used twice: as core.continuous.q_values for the household, and, without
        # gradient, as core.continuous.transition for the population)
        idxs, probs = jax.vmap(lambda s, a: gc._per_state(env, s, a))(shares, aggregate_s)  # (B, n, n_next)
        terminal = env.is_terminal(aggregate_s.time + 1)
        r = jax.vmap(lambda s, a, term: gc.reward(env, s, a, term))(shares, aggregate_s, terminal)
        next_mu = jax.vmap(  # the population: taken as given (fictitious play)
            lambda mu, i, pr: jnp.zeros((env.n_states,)).at[i.reshape(-1)].add((mu[:, None] * pr).reshape(-1))
        )(aggregate_s.mu, idxs, jax.lax.stop_gradient(probs))
        next_z = env.next_shock(aggregate_s.z, e)
        next_r, next_w = jax.vmap(env.prices)(next_mu, next_z)
        nxt = aggregate_s.replace(mu=next_mu, z=next_z, time=aggregate_s.time + 1, interest_rate=next_r, wage=next_w)
        return jax.lax.stop_gradient(nxt), (aggregate_s.time, terminal, idxs, probs, r)

    last, path = jax.lax.scan(_fwd, aggregate_s0, eps)
    v = jnp.where((last.time <= T - 1)[:, None], value_fn(last), 0.0)

    def _bwd(v_next, inputs):
        t, terminal, idxs, probs, r = inputs
        cont = jnp.sum(jax.vmap(lambda vn, i: vn[i])(v_next, idxs) * probs, axis=-1)  # E[v_next(x') | x]
        v = r + gamma * jnp.where(terminal[:, None], 0.0, cont)
        return jnp.where((t <= T - 1)[:, None], v, 0.0), None

    v0, _ = jax.lax.scan(_bwd, v, path, reverse=True)
    return jnp.sum(aggregate_s0.mu * v0, axis=-1)


# --- data ---
def flatten(seq):
    """(T, E, ...) aggregate states -> (T * E, ...) cross-sections."""
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), seq)


def policy_data(seq, key, per_econ):
    """per_econ cross-sections per economy at distinct random periods 0 .. T-2 (decision periods)."""
    T, E = seq.time.shape
    t_idx = jnp.argsort(jax.random.uniform(key, (E, T - 1)), axis=1)[:, :per_econ]  # (E, per_econ)
    e_idx = jnp.broadcast_to(jnp.arange(E)[:, None], t_idx.shape)
    return jax.tree.map(lambda x: x[t_idx.reshape(-1), e_idx.reshape(-1)], seq)


def make_normalizer(env, seq, V):
    """Input and value-target statistics of the first data set (mu-weighted over individual states)."""
    states = env.params.states
    mu = seq.mu / seq.mu.sum(-1, keepdims=True)  # (T, E, n)
    n_cs = mu.shape[0] * mu.shape[1]

    def wstats(x):  # x broadcastable to mu: mu-weighted mean and std over all cross-sections
        m = jnp.sum(mu * x) / n_cs
        return m, jnp.sqrt(jnp.sum(mu * (x - m) ** 2) / n_cs)

    def stats(x):
        return jnp.mean(x), jnp.std(x)

    T = env.params.max_steps_in_episode
    cols = [
        wstats(states[:, 0]),
        wstats(states[:, 1]),
        stats(seq.z),
        stats(mu @ states[:, 0]),
        stats((T - seq.time).astype(jnp.float32)),
    ]
    mean = jnp.array([c[0] for c in cols])
    std = jnp.array([c[1] for c in cols])
    v_mean, v_std = wstats(V)
    return dn.Normalizer(
        x_mean=mean, x_std=jnp.where(std > 1e-8, std, 1.0), v_mean=v_mean, v_std=jnp.where(v_std > 1e-8, v_std, 1.0)
    )


# --- value function: supervised learning (eq. (9)) ---
@partial(jax.jit, static_argnames=("env", "cfg", "n_epochs", "batch", "lr", "n_valid"))
def fit_values(env, cfg, norm, vnets, data, V, key, n_epochs, batch, lr, n_valid):
    """
    Fit each network of the ensemble (leading axis) on targets V (N, n_states) at cross-sections
    `data` (N,): Adam as the reference code (beta_1 = beta_2 = 0.99), mini-batches of cross-sections,
    a mu-weighted squared error on standardized targets. The first n_valid cross-sections are held
    out. Returns (vnets, last epoch's training loss (n_vnet,), validation loss (n_vnet,)).
    """
    N = V.shape[0]
    n_train = N - n_valid
    n_batches = n_train // batch
    target = (V - norm.v_mean) / norm.v_std
    opt = optax.adam(lr, b1=0.99, b2=0.99, eps=1e-8)

    def loss_fn(net, idx):
        a = jax.tree.map(lambda x: x[idx], data)
        pred = dn.value_normalized(env, cfg, norm, net, a)
        return jnp.mean(jnp.sum(a.mu * (pred - target[idx]) ** 2, axis=-1))

    def train_one(net, key):
        def _epoch(carry, k):
            idxs = (n_valid + jax.random.permutation(k, n_train))[: n_batches * batch].reshape(n_batches, batch)

            def _step(carry, idx):
                net, opt_state = carry
                loss, grads = jax.value_and_grad(loss_fn)(net, idx)
                updates, opt_state = opt.update(grads, opt_state, net)
                return (optax.apply_updates(net, updates), opt_state), loss

            carry, losses = jax.lax.scan(_step, carry, idxs)
            return carry, losses.mean()

        (net, _), losses = jax.lax.scan(_epoch, (net, opt.init(net)), jax.random.split(key, n_epochs))
        return net, losses[-1], loss_fn(net, jnp.arange(n_valid))

    return jax.vmap(train_one)(vnets, jax.random.split(key, jax.tree.leaves(vnets)[0].shape[0]))


def value_data(env, gamma, policy, key, n_econ):
    """Cross-sections of economies simulated under `policy` and their exact values: (data (N,), V (N, n))."""
    seq, shares = forward(env, policy, key, n_econ)
    V = gc.path_evaluate(env, seq, shares, gamma)
    return seq, flatten(seq), V.reshape(-1, V.shape[-1])


def value_ce_error(env, cfg, norm, vnets, data, V, sigma):
    """mu-weighted mean |c_bar_NN / c_bar_target - 1| (consumption equivalents) on cross-sections, in %."""
    pred = dn.values(env, cfg, norm, vnets, data)
    err = jnp.abs(ce_ratio(pred, V, sigma) - 1)
    return 100 * float(jnp.mean(jnp.sum(data.mu / data.mu.sum(-1, keepdims=True) * err, axis=-1)))


# --- the whole algorithm ---
def train(args):
    env = make_model(args.model)
    if not isinstance(env, Model):
        raise ValueError(f"model {args.model!r}: this solver needs a model with explicit prices (hamlab.models.base.Model)")
    gamma, sigma = args.discount_factor, env.params.sigma
    T = env.params.max_steps_in_episode
    cfg = dn.Config(n_fm=args.n_fm, n_gm=args.n_gm, use_time=args.use_time, gm_income=args.gm_income, T=T)
    assert cfg.n_fm + cfg.n_gm > 0 or args.allow_no_moments, "no moments: the policy sees only (s, z, time)"
    out_dir = f"runs/{args.task}/{args.algo}" + (f"/{args.run_name}" if args.run_name else "")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            json.dump(asdict(args), f, indent=2)

    width, gm_width = tuple(args.net_width), tuple(args.gm_width)
    per_econ = args.cross_sections_per_econ
    steps_per_pass = args.policy_paths * per_econ // args.policy_batch
    assert args.freq_update_v % steps_per_pass == 0 and args.eval_every % steps_per_pass == 0, (
        f"freq_update_v and eval_every must be multiples of the steps per pass over the policy data ({steps_per_pass})"
    )

    # --- evaluation (fixed economies and key across the run, as gpdp/train.py) ---
    eval_key = jax.random.PRNGKey(args.eval_seed)
    path_br = jax.jit(lambda a: gc.path_best_response(env, a, gamma))
    path_values = jax.jit(lambda a, s: gc.path_evaluate(env, a, s, gamma))
    agent = DeepHAMAgent(env, None)
    mfax_exploitability = make_exploitability(
        env=env, agent=agent, state_type="states", gamma=gamma, num_envs=args.eval_envs, max_steps_in_episode=T
    )

    def evaluate(policy):
        seq, shares = forward(env, policy, eval_key, args.eval_envs)
        mu0 = np.asarray(seq.mu[0])
        w = lambda v: float((np.asarray(v) * mu0).sum(-1).mean())  # noqa: E731
        V_pol = np.asarray(path_values(seq, shares))
        v0_br, br_shares = path_br(seq)
        V_br = np.asarray(path_values(seq, br_shares))
        mu = np.asarray(seq.mu)
        wt = mu / mu.sum(-1, keepdims=True)
        per_period = ce_relerr_per_period(V_pol, V_br, mu, sigma)
        K = np.asarray(jax.vmap(lambda a: aggregate_features(env, a))(seq))[..., 1]  # (T, E)
        row = dict(
            policy_value=w(V_pol[0]),
            gain_clairvoyant_pct=ce_gain_pct(w(v0_br), w(V_pol[0]), sigma),
            ce_relerr_all_periods_pct=100 * float(per_period.mean()),
            ce_relerr_periods_120_126_pct=100 * float(per_period[120:127].mean()),
            mfax_exploitability=float(mfax_exploitability(eval_key, policy).exploitability.exploitability),
            K_mean_at={str(t): float(K[t].mean()) for t in (25, 50, 100)},
            mean_share_at={str(t): float((wt[t] * np.asarray(shares[t])).sum(-1).mean()) for t in (0, 64, 120)},
        )
        if cfg.n_gm:  # what the generalized moments encode: R^2 of linear fits of K and of the date on Q^C
            Q = np.asarray(dn.moments(env, cfg, policy.norm, policy.net.basis, flatten(seq).mu))[:, cfg.n_fm :]
            X = np.concatenate([np.ones((len(Q), 1)), Q], axis=1)
            for name, y in (("K", K.reshape(-1)), ("time", np.asarray(seq.time, np.float64).reshape(-1))):
                resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                row[f"gm_{name}_r2"] = float(1 - resid.var() / y.var())
        return row

    # --- start: a constant share generates the first data (the reference's init_const_share) ---
    key = jax.random.PRNGKey(args.seed)
    key, k_val, k_pol, k_sel, k_net, k_vnet, k_fit = jax.random.split(key, 7)
    t_start = time.perf_counter()
    seq_v, data_v, V = value_data(env, gamma, args.start_share, k_val, args.value_paths)
    norm = make_normalizer(env, seq_v, V.reshape(T, args.value_paths, -1))
    net = dn.init_net(k_net, cfg, width, gm_width)
    vnets = jax.vmap(lambda k: dn.init_net(k, cfg, width, gm_width))(jax.random.split(k_vnet, args.num_vnet))

    def refit_values(vnets, data_v, V, key):
        k_perm, k_fit = jax.random.split(key)
        perm = jax.random.permutation(k_perm, V.shape[0])  # the validation rows: random cross-sections
        data_v, V = jax.tree.map(lambda x: x[perm], data_v), V[perm]
        vnets, train_loss, valid_loss = fit_values(
            env, cfg, norm, vnets, data_v, V, k_fit, args.value_epochs, args.value_batch, args.value_lr, args.value_valid
        )
        vd = jax.tree.map(lambda x: x[: args.value_valid], data_v)
        info = dict(
            train_loss=np.asarray(train_loss).tolist(),
            valid_loss=np.asarray(valid_loss).tolist(),
            valid_ce_relerr_pct=value_ce_error(env, cfg, norm, vnets, vd, V[: args.value_valid], sigma),
        )
        return vnets, info

    vnets, vinfo = refit_values(vnets, data_v, V, k_fit)
    value_fits = [dict(step=0, **vinfo)]
    pdata = policy_data(forward(env, args.start_share, k_pol, args.policy_paths)[0], k_sel, per_econ)
    opt = optax.adam(args.policy_lr, b1=0.99, b2=0.99, eps=1e-8)
    opt_state = opt.init(net)
    train_time = time.perf_counter() - t_start
    print(f"start: value fit valid CE error {vinfo['valid_ce_relerr_pct']:.3f}%  ({train_time:.1f}s)", flush=True)

    def loss_fn(net, norm, vnets, aggregate_s0, eps):
        policy = dn.DeepHAMPolicy(net, norm, cfg)
        value_fn = lambda a: dn.values(env, cfg, norm, vnets, a)  # noqa: E731
        return -jnp.mean(window_value(env, gamma, policy, value_fn, aggregate_s0, eps))

    @jax.jit
    def policy_pass(net, opt_state, norm, vnets, pdata, key):
        """One pass over the policy data: steps_per_pass Adam steps, fresh aggregate shocks for each."""
        k_perm, k_eps = jax.random.split(key)
        n = steps_per_pass * args.policy_batch
        idxs = jax.random.permutation(k_perm, pdata.time.shape[0])[:n].reshape(steps_per_pass, args.policy_batch)

        def _step(carry, inputs):
            net, opt_state = carry
            idx, k = inputs
            aggregate_s0 = jax.tree.map(lambda x: x[idx], pdata)
            eps = jax.random.normal(k, (args.window, args.policy_batch))
            loss, grads = jax.value_and_grad(loss_fn)(net, norm, vnets, aggregate_s0, eps)
            updates, opt_state = opt.update(grads, opt_state, net)
            return (optax.apply_updates(net, updates), opt_state), loss

        (net, opt_state), losses = jax.lax.scan(_step, (net, opt_state), (idxs, jax.random.split(k_eps, steps_per_pass)))
        return net, opt_state, losses

    log = dict(iterations=[], value_fits=value_fits, steps_per_pass=steps_per_pass)

    def log_eval(step, losses, non_finite):
        policy = dn.DeepHAMPolicy(net, norm, cfg)
        row = dict(step=step, train_time=train_time, policy_loss=float(np.mean(losses)) if losses else float("nan"),
                   non_finite_passes=non_finite, **evaluate(policy))
        log["iterations"].append(row)
        print(
            f"step {step:6d}  train {train_time:7.1f}s  loss {row['policy_loss']:.4f}  "
            f"gain clairvoyant {row['gain_clairvoyant_pct']:.3f}%  MFAX expl {row['mfax_exploitability']:.4f}  "
            f"CE all {row['ce_relerr_all_periods_pct']:.3f}% end {row['ce_relerr_periods_120_126_pct']:.2f}%  "
            f"K100 {row['K_mean_at']['100']:.2f}"
            + (f"  R2(K|Q) {row['gm_K_r2']:.3f} R2(t|Q) {row['gm_time_r2']:.3f}" if cfg.n_gm else ""),
            flush=True,
        )

    log_eval(0, [], 0)
    step, losses, non_finite = 0, [], 0
    while step < args.num_steps:
        t0 = time.perf_counter()
        key, k_pass, k_data, k_sel, k_val, k_fit = jax.random.split(key, 6)
        new_net, new_opt_state, pass_losses = policy_pass(net, opt_state, norm, vnets, pdata, k_pass)
        pass_losses = np.asarray(pass_losses)
        if np.all(np.isfinite(pass_losses)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(new_net)):
            net, opt_state = new_net, new_opt_state
            losses.extend(pass_losses.tolist())
        else:  # safeguard (not in the reference code): a non-finite pass is dropped and logged
            non_finite += 1
            print(f"step {step}: non-finite policy pass dropped", flush=True)
        step += steps_per_pass
        policy = dn.DeepHAMPolicy(net, norm, cfg)
        if step % args.freq_update_v == 0 and step < args.num_steps:
            _, data_v, V = value_data(env, gamma, policy, k_val, args.value_paths)
            vnets, vinfo = refit_values(vnets, data_v, V, k_fit)
            value_fits.append(dict(step=step, **vinfo))
            print(f"step {step}: value refit, valid CE error {vinfo['valid_ce_relerr_pct']:.3f}%", flush=True)
        pdata = policy_data(forward(env, policy, k_data, args.policy_paths)[0], k_sel, per_econ)
        jax.block_until_ready(pdata)
        train_time += time.perf_counter() - t0
        if step % args.eval_every == 0 or step >= args.num_steps:
            log_eval(step, losses, non_finite)
            losses = []

    log["train_time"] = train_time
    policy = dn.DeepHAMPolicy(net, norm, cfg)
    if args.save:
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)
        with open(os.path.join(out_dir, "deepham_agent.pkl"), "wb") as f:
            pickle.dump(DeepHAMAgent(env, jax.device_get(policy)), f)
        with open(os.path.join(out_dir, "deepham_value.pkl"), "wb") as f:
            pickle.dump(dict(vnets=jax.device_get(vnets), norm=jax.device_get(norm), cfg=cfg), f)
        print(f"Saved to {out_dir}")
    return policy, vnets, log
