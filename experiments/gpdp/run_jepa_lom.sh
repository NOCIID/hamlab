#!/bin/bash
# GPDP in the reference configuration of run_report_config.sh (single value GP, alpha = 1, 20 iterations,
# 5 seeds) with tomorrow's K from the JEPA world model (forecasting only): a frozen encoder (latent size 16,
# pre-trained by experiments/gpdp/jepa_wm_pretrain.py on a seed-5 GPDP run) + a latent GP law of
# motion refitted each iteration + a probe s -> K. Decisions keep (z, K, T - t).
# Compare with runs lom_s* (the K law-of-motion GP) and report_singlevgp_a1_s* (K' exact).
# TAG names the batch (default d16: runs jepalom_d16_s*); EXTRA adds flags, e.g.
#   TAG=d16refit EXTRA="--jepa-refit-probe" ...   (the probe refitted each iteration)
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_jepa_lom.sh
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
MODEL=${MODEL:-$R/jepa_wm/d16/model.pkl}
TAG=${TAG:-d16}
EXTRA=${EXTRA:-}
mkdir -p "$LOGS"
run() {  # run <log name> <command...>; a training run whose log.json exists is skipped (resume)
    local name=$(echo "${@:2}" | sed -n 's/.*--run-name \([^ ]*\).*/\1/p')
    if [ -n "$name" ] && [ -f "$R/gpdp/$name/log.json" ]; then echo "=== $1  skipped (done)"; return; fi
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}
GPDP="$PY -m hamlab.solvers.gpdp --single-vgp --alpha 1.0 --num-iterations 20 --law-of-motion --lom-model jepa --jepa-model $MODEL $EXTRA"
RUNS=""
for s in 0 1 2 3 4; do
    run gpdp_jepalom_${TAG}_s$s $GPDP --seed $s --run-name jepalom_${TAG}_s$s
    RUNS="$RUNS $R/gpdp/jepalom_${TAG}_s$s"
done
SUF=$([ "$TAG" = "d16" ] && echo "" || echo "_$TAG")
run jepalom${SUF}_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $RUNS --out jepalom${SUF}_evaluate.json
run jepalom${SUF}_relative_errors $PY experiments/gpdp/relative_errors.py --runs $RUNS --out relative_errors_jepalom${SUF}.json
echo "all done"
