"""
Checks for the JEPA world model of the economy (hamlab/world_models/jepa/wm.py), against a fixed population: the final rule of
a GPDP run in the configuration of reports/gpdp_report.tex.

A1. What the latent s = E(mu) learns, on held-out economies:
    - linear probes from s to capital K, the date (T - t) and the shock z (fitted on the training
      economies; R^2 on the held-out ones; also with quadratic features);
    - effective rank of the latents (how many dimensions are used);
    - forecasts of K, 1, 5 and 20 periods ahead and open loop over the whole episode (latent rollout
      along the realized shocks, then a GP probe from s to K), against the K-based law-of-motion GP
      (gpdp/lom.py, fitted on 8 economies, as in GPDP).
A2. Does s do harm as a policy-GP input? The policy GP fitted on the best shares of 8 economies
    with inputs (z, s, T - t), (z, K, T - t) (GPDP's default) and (z, T - t), judged on 8 new
    economies (share error, consumption lost over the whole episode). On a fixed population K is
    almost a function of the date, so (z, T - t) already does as well as (z, K, T - t): this checks
    that s costs little, not that it helps. Whether it helps needs a changing population (GPDP).

    python experiments/gpdp/jepa_wm_checks.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_checks import ce_loss, share_error  # noqa: E402

from hamlab.solvers.gpdp import backward as gb  # noqa: E402
from hamlab.solvers.gpdp import lom as glom  # noqa: E402
from hamlab.solvers.gpdp import policy as gpol  # noqa: E402
from hamlab.solvers.gpdp import train as gt  # noqa: E402
from hamlab.world_models.jepa import wm as jw  # noqa: E402
from hamlab.solvers.gp_critic.critic import aggregate_features, gp_predict, init_hyper  # noqa: E402
from hamlab.models import make as make_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def paths(env, seq):
    """K (T, E), z (T, E), time left (T, E) along simulated paths."""
    f = np.asarray(jax.vmap(lambda a: aggregate_features(env, a))(seq))
    return f[..., 1], f[..., 0], (env.params.max_steps_in_episode - np.asarray(seq.time)).astype(np.float64)


def ridge(X, y, lam=1e-6):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def apply_ridge(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def quad(X):
    i, j = np.triu_indices(X.shape[1])
    return np.column_stack([X, X[:, i] * X[:, j]])


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def effective_rank(S):
    ev = np.clip(np.linalg.eigvalsh(np.cov(S, rowvar=False)), 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def lom_rollout(lom, K0, z, time, k):
    """K after k steps of the law-of-motion GP's mean, along the realized shocks: from K0 (M,) at dates time[:, 0]."""
    K = K0.copy()
    for j in range(k):
        K, _ = glom.predict_next_K(lom, np.column_stack([z[:, j], K, time[:, j]]))
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/endogenous/gpdp/report_singlevgp_a1_s0")
    p.add_argument("--configs", nargs="+", default=["4:64", "4:8", "2:64", "8:64"], help="latent dim : training economies")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--out", default="jepa_wm_checks.json")
    cli = p.parse_args()

    env = make_model("krusell_smith")
    gamma, T = 0.95, env.params.max_steps_in_episode
    with open(Path(cli.run) / "gpdp_agent.pkl", "rb") as f:
        population = pickle.load(f).mf_params
    train_all, _ = gt.forward(env, population, jax.random.PRNGKey(1), 64)
    test, _ = gt.forward(env, population, jax.random.PRNGKey(777), 32)
    K_te, z_te, tau_te = paths(env, test)
    # --- the K law of motion, as in GPDP: fitted on 8 economies ---
    first8 = jax.tree.map(lambda x: x[:, :8], train_all)
    lom, _ = glom.fit_law_of_motion(*glom.transitions(env, first8))
    # --- A2 data: best shares at 8 training and 8 new stations ---
    st_tr, _ = gt.forward(env, population, jax.random.PRNGKey(4242), 8)
    st_te, _ = gt.forward(env, population, jax.random.PRNGKey(999), 8)
    _, best_tr, _, _ = gb.backward_pass_single(env, st_tr, gamma)
    _, best_te, _, _ = gb.backward_pass_single(env, st_te, gamma)
    n_periods = best_tr.shape[0]

    out = dict(run=cli.run, steps=cli.steps, rollout=cli.rollout, lam=cli.lam, configs={})
    horizons = (1, 5, 20)
    starts = np.arange(0, T - max(horizons) - 1, 3)  # start periods for the k-step forecasts

    # --- reference: the law of motion's forecast errors ---
    ref = {}
    for k in horizons:
        e = []
        for t0 in starts:
            Kh = lom_rollout(lom, K_te[t0], z_te[t0:t0 + k].T, tau_te[t0:t0 + k].T, k)
            e.append(np.abs(Kh / K_te[t0 + k] - 1))
        ref[f"{k}_step_mean_rel_err_pct"] = float(100 * np.mean(e))
    Kh = lom_rollout(lom, K_te[0], z_te[: T - 1].T, tau_te[: T - 1].T, T - 1)
    ref["open_loop_t127_rel_err_pct"] = float(100 * np.mean(np.abs(Kh / K_te[T - 1] - 1)))
    out["law_of_motion_gp_8_economies"] = ref
    print("law-of-motion GP:", json.dumps(ref), flush=True)

    for cfg in cli.configs:
        d, n_econ = (int(x) for x in cfg.split(":"))
        train = jax.tree.map(lambda x: x[:, :n_econ], train_all)
        t0_ = time.perf_counter()
        wm, log = jw.train_world_model(train, d=d, k=cli.rollout, steps=cli.steps, lam=cli.lam)
        fit_time = time.perf_counter() - t0_
        S_tr = np.asarray(jw.encode(wm, jnp.asarray(train.mu)), np.float64)  # (T, E, d)
        S_te = np.asarray(jw.encode(wm, jnp.asarray(test.mu)), np.float64)
        K_tr, z_tr, tau_tr = paths(env, train)
        row = dict(fit_time_s=fit_time, final_loss=log[-1][1], final_pred_mse=log[-1][2], final_sigreg=log[-1][3],
                   effective_rank=effective_rank(S_te.reshape(-1, d)), probes={}, forecasts={})
        # --- A1: probes ---
        for name, y_tr, y_te in (("K", K_tr, K_te), ("time_left", tau_tr, tau_te), ("z", z_tr, z_te)):
            for feat_name, feat in (("linear", lambda S: S), ("quadratic", quad)):
                coef = ridge(feat(S_tr.reshape(-1, d)), y_tr.reshape(-1))
                row["probes"][f"{name}_{feat_name}_r2"] = r2(y_te.reshape(-1), apply_ridge(coef, feat(S_te.reshape(-1, d))))
        # --- A1: forecasts of K through the latent, decoded by a GP probe s -> K (fitted on 1000
        #     training latents; a linear probe's own error, about 1%, would hide the forecast's) ---
        sub = np.random.default_rng(0).choice(S_tr.reshape(-1, d).shape[0], 1000, replace=False)
        probe_gp, _ = gb.fit_gp_bounded(jnp.asarray(S_tr.reshape(-1, d)[sub], jnp.float32),
                                        jnp.asarray(K_tr.reshape(-1)[sub, None], jnp.float32), init_hyper(d, d), 300)
        decode = lambda S: np.asarray(gp_predict(probe_gp, jnp.asarray(S.reshape(-1, d), jnp.float32)))[:, 0].reshape(S.shape[:-1])  # noqa: E731
        row["forecasts"]["0_step_decode_rel_err_pct"] = float(100 * np.mean(np.abs(decode(S_te) / K_te - 1)))
        for k in horizons:
            e = []
            for t0 in starts:
                Sh = jw.rollout(wm, jnp.asarray(S_te[t0], jnp.float32), jnp.asarray(z_te[t0:t0 + k], jnp.float32),
                                jnp.asarray(T - tau_te[t0:t0 + k], jnp.float32))[-1]
                e.append(np.abs(decode(np.asarray(Sh, np.float64)) / K_te[t0 + k] - 1))
            row["forecasts"][f"{k}_step_mean_rel_err_pct"] = float(100 * np.mean(e))
        Sh = jw.rollout(wm, jnp.asarray(S_te[0], jnp.float32), jnp.asarray(z_te[: T - 1], jnp.float32),
                        jnp.asarray(T - tau_te[: T - 1], jnp.float32))[-1]
        row["forecasts"]["open_loop_t127_rel_err_pct"] = float(100 * np.mean(np.abs(decode(np.asarray(Sh, np.float64)) / K_te[T - 1] - 1)))
        # --- A2: the policy GP with the latent in place of K ---
        a2 = {}
        S_st_tr = np.asarray(jw.encode(wm, jnp.asarray(st_tr.mu)), np.float64)[:n_periods]
        S_st_te = np.asarray(jw.encode(wm, jnp.asarray(st_te.mu)), np.float64)[:n_periods]
        K_a, z_a, tau_a = (x[:n_periods] for x in paths(env, st_tr))
        K_b, z_b, tau_b = (x[:n_periods] for x in paths(env, st_te))
        inputs = dict(
            z_s_tau=(np.concatenate([z_a[..., None], S_st_tr, tau_a[..., None]], -1), np.concatenate([z_b[..., None], S_st_te, tau_b[..., None]], -1)),
            z_K_tau=(np.stack([z_a, K_a, tau_a], -1), np.stack([z_b, K_b, tau_b], -1)),
            z_tau=(np.stack([z_a, tau_a], -1), np.stack([z_b, tau_b], -1)),
        )
        for name, (X_tr, X_te) in inputs.items():
            gp, _ = gb.fit_gp_bounded(jnp.asarray(X_tr.reshape(-1, X_tr.shape[-1]), jnp.float32),
                                      gpol.logit(best_tr).reshape(-1, best_tr.shape[-1]), init_hyper(X_tr.shape[-1], X_tr.shape[-1]), 300)
            pred = jax.nn.sigmoid(gp_predict(gp, jnp.asarray(X_te.reshape(-1, X_te.shape[-1]), jnp.float32))).reshape(best_te.shape)
            err, blocks = share_error(st_te, pred, best_te)
            _, ce_all = ce_loss(env, st_te, pred, best_te, gamma)
            a2[name] = dict(share_error=err, share_error_120_126=blocks["120-126"], ce_loss_all_pct=ce_all)
        row["policy_gp"] = a2
        out["configs"][cfg] = row
        print(f"d={d} economies={n_econ}: fit {fit_time:.0f}s, rank {row['effective_rank']:.2f} | probes "
              + " ".join(f"{k} {v:.4f}" for k, v in row["probes"].items())
              + " | forecasts " + " ".join(f"{k} {v:.3f}" for k, v in row["forecasts"].items())
              + " | policy GP CE loss " + " ".join(f"{k} {v['ce_loss_all_pct']:.4f}%" for k, v in a2.items()), flush=True)
        OUT.mkdir(exist_ok=True)
        (OUT / cli.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
