"""
Forecasting with the JEPA world model of the economy: two ways to improve the JEPA's forecasts,
side by side, against the K law-of-motion GP.

Encoders (latent size 8, trained on 64 economies of a fixed population, the final rule of a GPDP run in the configuration of reports/gpdp_report.tex):
    base   4-step training rollouts, 4000 steps (as in jepa_wm_checks.py)
    long   16-step training rollouts, 8000 steps (trained to forecast further)
Predictors, for each encoder:
    mlp    the JEPA's own network predictor
    gp     a GP law of motion on the frozen latent (the hybrid: JEPA encoder, GP predictor; jepa.wm.fit_latent_gp),
           fitted on the transitions of 8 economies, as the K law of motion
Judged on 32 held-out economies, 1, 5 and 20 steps ahead and open loop over the whole episode:
    latent error   ||predicted s - encoded s|| / sqrt(d)  (the latent has about unit spread)
    K error        capital read from the predicted latent (GP probe s -> K), relative to the true K;
                   with the decode floor: the same probe applied to the TRUE future latent, i.e.
                   the error a perfect forecast would still have.

    python experiments/gpdp/jepa_wm_forecast.py
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

from jepa_wm_checks import lom_rollout, paths  # noqa: E402

from hamlab.solvers.gpdp import backward as gb  # noqa: E402
from hamlab.solvers.gpdp import lom as glom  # noqa: E402
from hamlab.solvers.gpdp import train as gt  # noqa: E402
from hamlab.world_models.jepa import wm as jw  # noqa: E402
from hamlab.solvers.gp_critic.critic import gp_predict, init_hyper  # noqa: E402
from hamlab.models import make as make_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
HORIZONS = (1, 5, 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/endogenous/gpdp/report_singlevgp_a1_s0")
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--encoders", nargs="+", default=["base:4:4000", "long:16:8000"],
                   help="name:rollout:steps, or name:d:rollout:steps for a latent size other than --d")
    p.add_argument("--out", default="jepa_wm_forecast.json")
    cli = p.parse_args()

    env = make_model("krusell_smith")
    T = env.params.max_steps_in_episode
    with open(Path(cli.run) / "gpdp_agent.pkl", "rb") as f:
        population = pickle.load(f).mf_params
    train, _ = gt.forward(env, population, jax.random.PRNGKey(1), 64)
    test, _ = gt.forward(env, population, jax.random.PRNGKey(777), 32)
    first8 = jax.tree.map(lambda x: x[:, :8], train)
    K_tr, _, _ = paths(env, train)
    K_te, z_te, tau_te = paths(env, test)
    t_te = T - tau_te  # period index
    starts = np.arange(0, T - max(HORIZONS) - 1, 3)

    # --- reference: the K law-of-motion GP (8 economies) ---
    lom, _ = glom.fit_law_of_motion(*glom.transitions(env, first8))
    ref = {}
    for k in HORIZONS:
        e = [np.abs(lom_rollout(lom, K_te[t0], z_te[t0:t0 + k].T, tau_te[t0:t0 + k].T, k) / K_te[t0 + k] - 1) for t0 in starts]
        ref[f"K_err_{k}_pct"] = float(100 * np.mean(e))
    ref["K_err_open_loop_t127_pct"] = float(100 * np.mean(np.abs(
        lom_rollout(lom, K_te[0], z_te[: T - 1].T, tau_te[: T - 1].T, T - 1) / K_te[T - 1] - 1)))
    out = dict(run=cli.run, d=cli.d, law_of_motion_gp=ref, encoders={})
    print("K law-of-motion GP:", json.dumps({k: round(v, 4) for k, v in ref.items()}), flush=True)

    for spec in cli.encoders:
        parts = spec.split(":")
        name, d, k_roll, steps = (parts[0], cli.d, *parts[1:]) if len(parts) == 3 else (parts[0], int(parts[1]), *parts[2:])
        t0_ = time.perf_counter()
        wm, log = jw.train_world_model(train, d=d, k=int(k_roll), steps=int(steps))
        enc_time = time.perf_counter() - t0_
        S_tr = np.asarray(jw.encode(wm, jnp.asarray(train.mu)), np.float64)
        S_te = np.asarray(jw.encode(wm, jnp.asarray(test.mu)), np.float64)  # (T, E, d)
        # --- the probe s -> K (GP on 1000 training latents) and its decode ---
        sub = np.random.default_rng(0).choice(S_tr.reshape(-1, d).shape[0], 1000, replace=False)
        probe, _ = gb.fit_gp_bounded(jnp.asarray(S_tr.reshape(-1, d)[sub], jnp.float32),
                                     jnp.asarray(K_tr.reshape(-1)[sub, None], jnp.float32), init_hyper(d, d), 300)
        decode = lambda S: np.asarray(gp_predict(probe, jnp.asarray(np.reshape(S, (-1, d)), jnp.float32)))[:, 0]  # noqa: E731
        # --- the hybrid's predictor ---
        t1 = time.perf_counter()
        lgp = jw.fit_latent_gp(wm, first8)
        gp_time = time.perf_counter() - t1

        def step_mlp(S, z, t):
            return np.asarray(jw.predict(wm, jnp.asarray(S, jnp.float32), jnp.asarray(z, jnp.float32), jnp.asarray(t, jnp.float32)), np.float64)

        def step_gp(S, z, t):
            return jw.latent_gp_step(lgp, S, z, t, T)

        enc = dict(d=d, rollout=int(k_roll), steps=int(steps), encoder_train_time_s=enc_time, final_pred_mse=log[-1][2], final_sigreg=log[-1][3],
                   latent_gp_fit_time_s=gp_time, latent_gp_noise=float(np.exp(lgp.log_sn)), predictors={})
        for pname, step in (("mlp", step_mlp), ("gp", step_gp)):
            row = {}
            for k in HORIZONS:
                lat, kerr, floor = [], [], []
                for t0 in starts:
                    S = S_te[t0]
                    for j in range(k):
                        S = step(S, z_te[t0 + j], t_te[t0 + j])
                    lat.append(np.linalg.norm(S - S_te[t0 + k], axis=-1) / np.sqrt(d))
                    kerr.append(np.abs(decode(S) / K_te[t0 + k] - 1))
                    floor.append(np.abs(decode(S_te[t0 + k]) / K_te[t0 + k] - 1))
                row[f"latent_err_{k}"] = float(np.mean(lat))
                row[f"K_err_{k}_pct"] = float(100 * np.mean(kerr))
                row[f"K_decode_floor_{k}_pct"] = float(100 * np.mean(floor))
            S = S_te[0]
            for j in range(T - 1):
                S = step(S, z_te[j], t_te[j])
            row["latent_err_open_loop_t127"] = float(np.mean(np.linalg.norm(S - S_te[T - 1], axis=-1) / np.sqrt(d)))
            row["K_err_open_loop_t127_pct"] = float(100 * np.mean(np.abs(decode(S) / K_te[T - 1] - 1)))
            row["K_decode_floor_t127_pct"] = float(100 * np.mean(np.abs(decode(S_te[T - 1]) / K_te[T - 1] - 1)))
            enc["predictors"][pname] = row
            print(f"{name:5s} encoder ({enc_time:.0f}s) + {pname:3s}: " + " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
        out["encoders"][name] = enc
        OUT.mkdir(exist_ok=True)
        (OUT / cli.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
