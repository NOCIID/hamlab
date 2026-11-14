"""
Summary table: DeepHAM (runs deepham_<tag>_s*) against GPDP in the configuration of
reports/gpdp_report.tex (runs report_singlevgp_a1_s*), on the held-out evaluation of experiments/gpdp (64 economies, key 777):
MFAX's exploitability (nearest grid action, as in that report), CE relative errors against the
clairvoyant continuous best response, convergence time.

    python experiments/deepham/summarize_deepham.py [--tags gm1 fm1]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/gpdp/results"
BLOCKS = ["0-31", "32-63", "64-95", "96-111", "112-119", "120-126"]


def convergence(log, deepham, threshold=0.1):
    """Training time and evaluation row at which the gain from deviating (clairvoyant) first falls
    below threshold %. GPDP's row i evaluates the policy of iteration i - 1 (time of row i - 1, as
    experiments/gpdp/summarize_lom.py); DeepHAM's row evaluates the policy after its own train_time."""
    rows = log["iterations"]
    for i, r in enumerate(rows):
        if r.get("gain_clairvoyant_pct", np.inf) < threshold and i > 0:
            return (r["train_time"], r["step"]) if deepham else (rows[i - 1]["train_time"], i)
    return np.nan, np.nan


def fmt(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}f} ± {np.nanstd(x, ddof=1):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["gm1"])
    cli = p.parse_args()
    groups = {"GPDP (configuration of reports/gpdp_report.tex)": ("gpdp", "report_singlevgp_a1_s", "report_evaluate.json", "relative_errors_report.json")}
    for tag in cli.tags:
        groups[f"DeepHAM ({tag})"] = ("deepham", f"deepham_{tag}_s", f"deepham_{tag}_evaluate.json", f"relative_errors_deepham_{tag}.json")
    out = {}
    for name, (algo, prefix, ev_file, re_file) in groups.items():
        ev = json.loads((RES / ev_file).read_text())["runs"]
        re = json.loads((RES / re_file).read_text())["runs"]
        runs = [r for r in (f"{algo}/{prefix}{s}" for s in range(5)) if r in ev and r in re]
        cr = [re[r]["by_best_response"]["clairvoyant_continuous"] for r in runs]
        logs = [json.loads((ROOT / "runs/endogenous" / r / "log.json").read_text()) for r in runs]
        conv = [convergence(lg, algo == "deepham") for lg in logs]
        g = dict(
            n_runs=len(runs),
            mfax_exploitability_nearest=fmt([ev[r]["mfax_nearest_grid"]["exploitability"] for r in runs]),
            mfax_exploitability_split=fmt([ev[r]["mfax"]["exploitability"] for r in runs]),
            gain_from_deviating_pct=fmt([c["gain_from_deviating_pct"] for c in cr], 3),
            ce_relerr_t0_pct=fmt([c["value_relerr_t0_mu0_weighted_pct"] for c in cr], 3),
            ce_relerr_all_periods_pct=fmt([c["value_relerr_all_periods_pop_weighted_pct"] for c in cr], 3),
            ce_relerr_all_periods_each=[round(c["value_relerr_all_periods_pop_weighted_pct"], 3) for c in cr],
            ce_relerr_by_block_pct=" · ".join(f"{np.mean([c['value_relerr_by_block_pct'][b] for c in cr]):.3f}" for b in BLOCKS),
            policy_relerr_pct=fmt([c["policy_relerr_pop_weighted_pct"] for c in cr], 2),
            time_to_converge_s_each=[round(float(c[0]), 1) for c in conv],
            converged_at_each=[c[1] for c in conv],
            total_train_time_s=fmt([lg["train_time"] for lg in logs], 1),
            capital_t100=fmt([ev[r]["K_mean_at"]["100"] for r in runs], 2),
        )
        if algo == "deepham":  # what the generalized moment encodes, and the value networks' accuracy
            last = [lg["iterations"][-1] for lg in logs]
            if "gm_K_r2" in last[0]:
                g["final_R2_of_K_on_Q"] = fmt([r["gm_K_r2"] for r in last], 4)
                g["final_R2_of_date_on_Q"] = fmt([r["gm_time_r2"] for r in last], 3)
            g["value_fit_valid_ce_relerr_pct_last"] = fmt([lg["value_fits"][-1]["valid_ce_relerr_pct"] for lg in logs], 3)
            g["non_finite_passes"] = [lg["iterations"][-1]["non_finite_passes"] for lg in logs]
        out[name] = g
    print(json.dumps(out, indent=1))
    (ROOT / "experiments/deepham/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments/deepham/results/summary_deepham.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
