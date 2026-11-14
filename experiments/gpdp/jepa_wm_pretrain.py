"""
Pre-train the frozen parts of the JEPA law of motion (hamlab/world_models/jepa/lom.py): the world model's encoder and the probe s -> K.

Training data: the stations of every iteration of one GPDP run in the configuration of reports/gpdp_report.tex
(single value GP, alpha = 1, with the date), with a seed not used by the comparison runs: 20 iterations x 8
economies, covering every population GPDP goes through, from the 15% start to the equilibrium.

Diagnostics, on economies the encoder never saw:
    - the converged population (that run's final rule) and the start population (15% everywhere);
    - probe R^2 for K and the decode error, for the float32 policy-GP probe and the float64 probe;
    - the JEPA law of motion's one-step K forecast (latent GP fitted on 8 stations of the converged
      population, then the probe), against the K law-of-motion GP on the same stations.
Saves runs/endogenous/jepa_wm/<name>/model.pkl (JepaModel with the more precise probe) and diagnostics.json.

    python experiments/gpdp/jepa_wm_pretrain.py --d 8 --steps 4000 --name d8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_wm_checks import paths, r2  # noqa: E402

from hamlab.solvers.gpdp.config import args as GPDPArgs  # noqa: E402
from hamlab.solvers.gpdp import backward as gb  # noqa: E402
from hamlab.solvers.gpdp import lom as glom  # noqa: E402
from hamlab.solvers.gpdp import train as gt  # noqa: E402
from hamlab.world_models.jepa import lom as jl  # noqa: E402
from hamlab.world_models.jepa import wm as jw  # noqa: E402
from hamlab.solvers.gp_critic.critic import gp_predict, init_hyper  # noqa: E402
from hamlab.models import make as make_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def collect_stations(seed):
    """Run GPDP (configuration of reports/gpdp_report.tex) and record each iteration's stations; also return the final rule."""
    stations = []
    original = gb.backward_pass_single

    def recording(env, aggregate_s_seq, *a, **kw):
        stations.append(aggregate_s_seq)
        return original(env, aggregate_s_seq, *a, **kw)

    gb.backward_pass_single = recording
    try:
        a = GPDPArgs()
        a.single_vgp, a.alpha, a.num_iterations, a.seed, a.eval_frequency, a.save = True, 1.0, 20, seed, 10_000, False
        policy, _ = gt.train(a)
    finally:
        gb.backward_pass_single = original
    pool = jax.tree.map(lambda *x: jnp.concatenate(x, axis=1), *stations)
    return pool, policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=5, help="seed of the GPDP run that provides the training economies")
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--probe-points", type=int, default=2000)
    p.add_argument("--name", default="d8")
    cli = p.parse_args()

    env = make_model("krusell_smith")
    T = env.params.max_steps_in_episode
    out_dir = ROOT / "runs/endogenous/jepa_wm" / cli.name
    t0 = time.perf_counter()
    pool, final_policy = collect_stations(cli.seed)
    print(f"collected {pool.mu.shape[1]} economies in {time.perf_counter() - t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    wm, log = jw.train_world_model(pool, d=cli.d, k=cli.rollout, steps=cli.steps)
    enc_time = time.perf_counter() - t0
    S_pool = np.asarray(jw.encode(wm, jnp.asarray(pool.mu)), np.float64)
    K_pool = paths(env, pool)[0]
    sub = np.random.default_rng(0).choice(S_pool.reshape(-1, cli.d).shape[0], cli.probe_points, replace=False)
    Xp, yp = S_pool.reshape(-1, cli.d)[sub], K_pool.reshape(-1)[sub]
    probe32, _ = gb.fit_gp_bounded(jnp.asarray(Xp, jnp.float32), jnp.asarray(yp[:, None], jnp.float32),
                                   init_hyper(cli.d, cli.d), 300)
    probe64 = jw.fit_probe(Xp, yp)

    # --- diagnostics on economies the encoder never saw ---
    diag = dict(seed=cli.seed, d=cli.d, rollout=cli.rollout, steps=cli.steps, encoder_train_time_s=enc_time,
                final_pred_mse=log[-1][2], final_sigreg=log[-1][3], populations={})
    pops = dict(converged=final_policy, start_15pct=jnp.float32(0.15))
    for pname, pol in pops.items():
        out = gt.forward(env, pol, jax.random.PRNGKey(777), 32)
        seq = out[0]
        K, z, tau = paths(env, seq)
        S = np.asarray(jw.encode(wm, jnp.asarray(seq.mu)), np.float64)
        flat = S.reshape(-1, cli.d)
        k32 = np.asarray(gp_predict(probe32, jnp.asarray(flat, jnp.float32)))[:, 0]
        k64 = jw.probe_predict(probe64, flat)[0]
        row = dict(
            K_r2_probe32=r2(K.reshape(-1), k32), K_r2_probe64=r2(K.reshape(-1), k64),
            decode_rel_err_pct_probe32=float(100 * np.mean(np.abs(k32 / K.reshape(-1) - 1))),
            decode_rel_err_pct_probe64=float(100 * np.mean(np.abs(k64 / K.reshape(-1) - 1))),
        )
        if pname == "converged":  # one-step K forecasts, both laws of motion fitted on 8 stations of this population
            st = jax.tree.map(lambda x: x[:, :8], gt.forward(env, pol, jax.random.PRNGKey(4242), 8)[0])
            lom, _ = glom.fit_law_of_motion(*glom.transitions(env, st))
            X_e, y_e = (np.asarray(a, np.float64) for a in glom.transitions(env, seq))
            K_next = X_e[:, 1] + y_e[:, 0]
            m_gp, _ = glom.predict_next_K(lom, X_e)
            for probe_name, probe in (("probe32", probe32), ("probe64", probe64)):
                model = jl.JepaModel(wm, probe)
                lgp = jl.fit_latent_law(model, st)
                m_j, v_j = jl.next_K(model, lgp, S[:-1].reshape(-1, cli.d), X_e[:, 0], T - X_e[:, 2])
                row[f"one_step_K_err_pct_jepa_{probe_name}"] = float(100 * np.mean(np.abs(m_j / K_next - 1)))
                row[f"one_step_zscore_std_jepa_{probe_name}"] = float(np.std((K_next - m_j) / np.sqrt(v_j)))
            row["one_step_K_err_pct_K_law_of_motion"] = float(100 * np.mean(np.abs(m_gp / K_next - 1)))
        diag["populations"][pname] = row
        print(pname, json.dumps({k: round(v, 5) for k, v in row.items()}), flush=True)

    best = "probe64" if diag["populations"]["converged"]["decode_rel_err_pct_probe64"] <= diag["populations"]["converged"]["decode_rel_err_pct_probe32"] else "probe32"
    jl.save(jl.JepaModel(wm, probe64 if best == "probe64" else probe32), out_dir / "model.pkl")
    diag["saved_probe"] = best
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"saved {out_dir / 'model.pkl'} (with {best})", flush=True)


if __name__ == "__main__":
    main()
