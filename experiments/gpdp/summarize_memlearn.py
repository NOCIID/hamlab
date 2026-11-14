"""
Summary table: GPDP in the reference configuration of run_report_config.sh (policy GP over (z, K, T - t)) against GPDP without the date
and with a learned memory (memlearn_zkh_s*: inputs (z, K, h); memlearn_h_s*: h only; memlearn_v3_*:
the same with the current defaults: minimum length scale 0.05, clipped logits). Same metrics and convergence rule as summarize_lom.py;
also how often an iteration blows up, and the share at period 126.

    python experiments/gpdp/summarize_memlearn.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_lom import BLOCKS, RES, ROOT, convergence, fmt  # noqa: E402

GROUPS = {
    "report: policy GP (z, K, T - t)": ("report_singlevgp_a1_s", "report_evaluate.json", "relative_errors_report.json"),
    "no date, learned memory: (z, K, h)": ("memlearn_zkh_s", "memlearn_evaluate.json", "relative_errors_memlearn.json"),
    "no date, learned memory: h only": ("memlearn_h_s", "memlearn_evaluate.json", "relative_errors_memlearn.json"),
    # v3: minimum length scale 0.05 (was 0.2) and logits clipped to the targets' range
    "v3 no date, learned memory: (z, K, h)": ("memlearn_v3_zkh_s", "memlearn_v3_evaluate.json", "relative_errors_memlearn_v3.json"),
    "v3 no date, learned memory: h only": ("memlearn_v3_h_s", "memlearn_v3_evaluate.json", "relative_errors_memlearn_v3.json"),
    # v4a05: v3's settings with blending alpha = 0.5 (and the per-iteration CE metric in the logs)
    "v4a05 no date, learned memory: (z, K, h), alpha 0.5": ("memlearn_v4a05_zkh_s", "memlearn_v4a05_evaluate.json", "relative_errors_memlearn_v4a05.json"),
}


def main():
    out = {}
    for name, (prefix, ev_file, re_file) in GROUPS.items():
        ev = json.loads((RES / ev_file).read_text())["runs"]
        re = json.loads((RES / re_file).read_text())["runs"]
        runs = [r for r in (f"gpdp/{prefix}{s}" for s in range(5)) if r in ev and r in re]
        cr = [re[r]["by_best_response"]["clairvoyant_continuous"] for r in runs]
        logs = [json.loads((ROOT / "runs/endogenous" / r / "log.json").read_text())["iterations"] for r in runs]
        conv = [convergence(dict(iterations=lg)) for lg in logs]
        gains = [[row["gain_clairvoyant_pct"] for row in lg[2:]] for lg in logs]
        out[name] = dict(
            n_runs=len(runs),
            mfax_exploitability_nearest=fmt([ev[r]["mfax_nearest_grid"]["exploitability"] for r in runs]),
            gain_from_deviating_pct=fmt([c["gain_from_deviating_pct"] for c in cr], 2),
            ce_relerr_t0_pct=fmt([c["value_relerr_t0_mu0_weighted_pct"] for c in cr], 2),
            ce_relerr_all_periods_pct=fmt([c["value_relerr_all_periods_pop_weighted_pct"] for c in cr], 3),
            ce_relerr_by_block_pct=" · ".join(
                f"{np.mean([c['value_relerr_by_block_pct'][b] for c in cr]):.3f}" for b in BLOCKS
            ),
            policy_relerr_pct=fmt([c["policy_relerr_pop_weighted_pct"] for c in cr], 2),
            time_to_converge_s_median=float(np.nanmedian([c[0] for c in conv])),
            iterations_to_converge_each=[c[1] for c in conv],
            time_per_iteration_s=fmt([np.mean([row["step_time"] for row in lg[1:]]) for lg in logs], 2),
            median_gain_iterations_2_19_pct=float(np.median([np.nanmedian(g) for g in gains])),
            blowup_iterations_each=[sum((not math.isfinite(x)) or x > 1.0 for x in g) for g in gains],
            capital_t100=fmt([ev[r]["K_mean_at"]["100"] for r in runs], 2),
            capital_t127=fmt([ev[r]["K_mean_at"]["127"] for r in runs], 2),
            share_t126=fmt([ev[r]["mean_share_at"]["126"] for r in runs], 3),
        )
    print(json.dumps(out, indent=1))
    (RES / "summary_memlearn.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
