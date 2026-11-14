"""
Checks for the policy GP with a learned memory (hamlab/solvers/gpdp/memory.py), against a
fixed population: the final rule of a GPDP run in the reference configuration of run_report_config.sh
(policy GP over (z, K, T - t)).

The backward pass gives the best shares on 8 training stations and on 8 held-out stations. The
policy GP is fitted on the training stations in several ways:
    zkt          (z, K, T - t)                       GPDP's default policy GP (has the date)
    zk           (z, K)                              no date, no memory
    zkh_fixed    (z, K, h), chrono GRU not trained   memory that is not learned
    zkh          (z, K, h), GRU learned (chrono init)
    zkh_random   (z, K, h), GRU learned (flax's default init)
    h            h only, GRU learned (chrono init)   RSPG's information set
and judged on the held-out stations:
    share error  population-weighted mean |predicted share - best share|, overall and by block
    CE loss      consumption lost by following the predicted shares instead of the best shares
                 along the held-out paths (% of consumption; start of episode and all periods)
    date in h    R^2 of T - t regressed linearly on h (fitted on the training stations)
plus the same share error on the training stations (overfitting) and the fit time.
A consistency test checks that the public prediction path reproduces the fit's own predictions.

    python experiments/gpdp/memory_checks.py --run runs/endogenous/gpdp/report_singlevgp_a1_s0
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hamlab.solvers.gpdp import backward as gb  # noqa: E402
from hamlab.core import continuous as gc  # noqa: E402
from hamlab.solvers.gpdp import memory as gm  # noqa: E402
from hamlab.solvers.gpdp import policy as gpol  # noqa: E402
from hamlab.solvers.gpdp import train as gt  # noqa: E402
from hamlab.models import make as make_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
BLOCKS = [(0, 32), (32, 64), (64, 96), (96, 112), (112, 120), (120, 127)]


def shares_of(env, policy, seq, n_periods):
    """Predicted shares (n_periods, N, n_states) along station paths, for a GP or a memory policy GP."""
    if isinstance(policy, gpol.RecurrentPolicyGP):
        H = gm.hidden_sequence(policy.gru, gm.path_obs(seq))
        return jnp.stack([
            gpol.predict_shares_memory(env, policy, jax.tree.map(lambda x: x[t], seq), H[t]) for t in range(n_periods)
        ])
    return gpol.path_shares(env, policy, jax.tree.map(lambda x: x[:n_periods], seq))


def share_error(seq, pred, best):
    """Population-weighted mean |pred - best|: overall, and by block of periods."""
    mu = np.asarray(seq.mu[: best.shape[0]])
    w = mu / mu.sum(-1, keepdims=True)
    e = (w * np.abs(np.asarray(pred) - np.asarray(best))).sum(-1)  # (T-1, N)
    return float(e.mean()), {f"{a}-{b - 1}": float(e[a:b].mean()) for a, b in BLOCKS}


def ce_loss(env, seq, pred, best, gamma):
    """% of consumption lost by following pred instead of best along the paths: (t = 0, all periods)."""
    sigma = env.params.sigma
    pad = lambda s: jnp.concatenate([s, jnp.full_like(s[:1], 0.5)], axis=0)  # noqa: E731  (terminal share unused)
    V_p = np.asarray(gc.path_evaluate(env, seq, pad(pred), gamma))
    V_b = np.asarray(gc.path_evaluate(env, seq, pad(best), gamma))
    mu = np.asarray(seq.mu)
    w = mu / mu.sum(-1, keepdims=True)
    m_p, m_b = (w * V_p).sum(-1)[:-1], (w * V_b).sum(-1)[:-1]  # (T-1, N) population-weighted values
    loss = 100 * (1 - (m_p / m_b) ** (1 / (1 - sigma)))
    return float(loss[0].mean()), float(loss.mean())


def date_r2(policy, seq_tr, seq_te, n_periods):
    """R^2 on the held-out stations of a linear regression of T - t on h, fitted on the training stations."""
    T = 128

    def design(seq):
        H = np.asarray(gm.hidden_sequence(policy.gru, gm.path_obs(seq)))[:n_periods]
        tau = np.broadcast_to((T - np.arange(n_periods))[:, None], H.shape[:2])
        return np.column_stack([np.ones(H[..., 0].size), H.reshape(-1, H.shape[-1])]), tau.reshape(-1)

    A, y = design(seq_tr)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    A, y = design(seq_te)
    return float(1 - np.sum((A @ coef - y) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/endogenous/gpdp/report_singlevgp_a1_s0")
    p.add_argument("--stations", type=int, default=8)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--input-weight-decay", type=float, default=0.1)
    p.add_argument("--noise-floor", type=float, default=0.03)
    p.add_argument("--min-lengthscale", type=float, default=0.05)
    p.add_argument("--out", default="memory_checks.json")
    cli = p.parse_args()

    env = make_model("krusell_smith")
    gamma, T = 0.95, env.params.max_steps_in_episode
    with open(Path(cli.run) / "gpdp_agent.pkl", "rb") as f:
        population = pickle.load(f).mf_params
    seq_tr, _ = gt.forward(env, population, jax.random.PRNGKey(4242), cli.stations)
    seq_te, _ = gt.forward(env, population, jax.random.PRNGKey(777), cli.stations)
    _, best_tr, _, _ = gb.backward_pass_single(env, seq_tr, gamma)
    _, best_te, _, _ = gb.backward_pass_single(env, seq_te, gamma)
    n_periods = best_tr.shape[0]
    out = dict(run=cli.run, stations=cli.stations, hidden=cli.hidden, steps=cli.steps,
               weight_decay=cli.weight_decay, n_val=cli.n_val, input_weight_decay=cli.input_weight_decay,
               noise_floor=cli.noise_floor, min_lengthscale=cli.min_lengthscale, variants={})

    def fit_plain(use_time):
        X, Y = gpol.training_data(env, seq_tr, best_tr, use_time=use_time)
        return gpol.fit_policy_gp(X, Y, cli.steps)[0], {}

    def fit_mem(chrono=True, train=True, with_zk=True, seed=0):
        gru = gm.init_learned_gru(jax.random.PRNGKey(seed), cli.hidden, chrono=chrono)
        return gm.fit_memory_policy(
            env, seq_tr, best_tr, gru, with_zk=with_zk, n_steps=cli.steps, weight_decay=cli.weight_decay,
            n_val=cli.n_val, train_gru=train, set_obs_norm=True, input_weight_decay=cli.input_weight_decay,
            noise_floor=cli.noise_floor, min_lengthscale=cli.min_lengthscale,
        )

    variants = dict(
        zkt=lambda: fit_plain(True),
        zk=lambda: fit_plain(False),
        zkh_fixed=lambda: fit_mem(train=False),
        zkh=lambda: fit_mem(),
        zkh_random=lambda: fit_mem(chrono=False),
        h=lambda: fit_mem(with_zk=False),
    )
    for name, fit in variants.items():
        t0 = time.perf_counter()
        policy, info = fit()
        jax.block_until_ready(policy)
        fit_time = time.perf_counter() - t0
        pred_tr = shares_of(env, policy, seq_tr, n_periods)
        pred_te = shares_of(env, policy, seq_te, n_periods)
        err_tr, _ = share_error(seq_tr, pred_tr, best_tr)
        err_te, blocks = share_error(seq_te, pred_te, best_te)
        ce0, ce_all = ce_loss(env, seq_te, pred_te, best_te, gamma)
        row = dict(
            fit_time_s=fit_time,
            share_error_train=err_tr,
            share_error_heldout=err_te,
            share_error_heldout_by_block=blocks,
            ce_loss_heldout_t0_pct=ce0,
            ce_loss_heldout_all_periods_pct=ce_all,
            heldout_logit_range=[float(np.log(np.min(pred_te)) - np.log1p(-np.min(pred_te))),
                                 float(np.log(np.max(pred_te)) - np.log1p(-np.max(pred_te)))],
            **{f"fit_{k}": v for k, v in info.items()},
        )
        if isinstance(policy, gpol.RecurrentPolicyGP):
            row["date_r2_heldout"] = date_r2(policy, seq_tr, seq_te, n_periods)
            row["length_scales"] = np.exp(np.asarray(policy.gp.hyp.log_ell)).round(3).tolist()
        out["variants"][name] = row
        print(f"{name:11s} fit {fit_time:6.1f}s | share err train {err_tr:.4f} held-out {err_te:.4f} | "
              f"CE loss held-out t0 {ce0:.3f}% all {ce_all:.3f}%"
              + (f" | date R2 {row['date_r2_heldout']:.3f}" if "date_r2_heldout" in row else ""), flush=True)

    # --- consistency: the public prediction path (as the forward pass and the MFAX agent use it)
    #     reproduces the fitted GP's own predictions on the training stations ---
    policy, _ = fit_mem()
    H = gm.hidden_sequence(policy.gru, gm.path_obs(seq_tr))[:n_periods]
    X_raw = jax.vmap(lambda a, h: gpol.memory_inputs(env, a, h, True))(
        jax.tree.map(lambda x: x[:n_periods], seq_tr), H).reshape(-1, policy.gp.X.shape[-1])
    X_std = (X_raw - policy.gp.x_mean) / policy.gp.x_std
    out["consistency_max_abs_input_diff"] = float(jnp.max(jnp.abs(X_std - policy.gp.X)))
    print("consistency: max |public-path inputs - fitted inputs| =", out["consistency_max_abs_input_diff"])

    OUT.mkdir(exist_ok=True)
    (OUT / cli.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
