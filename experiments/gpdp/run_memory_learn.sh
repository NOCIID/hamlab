#!/bin/bash
# GPDP in the reference configuration of run_report_config.sh (single value GP, alpha = 1, 20 iterations,
# 5 seeds) WITHOUT the date in the policy GP, and with a GRU learned together with the policy GP
# instead (hamlab/solvers/gpdp/memory.py):
#   memlearn_zkh_s*  policy GP inputs (z, K, h)
#   memlearn_h_s*    policy GP inputs h only (RSPG's information set)
# Then the held-out evaluation and relative errors, as for run_report_config.sh. Finished runs are skipped.
# TAG names a batch: TAG=v3 gives memlearn_v3_zkh_s* etc. (no TAG: a batch with a minimum
# length scale of 0.2 and no logit clipping, the defaults at the time).
# ALPHA (default 1.0) is the blending weight; VARIANTS (default "zkh h") picks the versions to run.
# Usage (from the repository root):  TAG=v3 PY=python caffeinate -i bash experiments/gpdp/run_memory_learn.sh
#                                    TAG=v4a05 ALPHA=0.5 VARIANTS=zkh ... (blending, (z, K, h) only)
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
T=${TAG:+${TAG}_}
ALPHA=${ALPHA:-1.0}
VARIANTS=${VARIANTS:-"zkh h"}
mkdir -p "$LOGS"
run() {  # run <log name> <command...>; a training run whose log.json exists is skipped (resume)
    local name=$(echo "${@:2}" | sed -n 's/.*--run-name \([^ ]*\).*/\1/p')
    if [ -n "$name" ] && [ -f "$R/gpdp/$name/log.json" ]; then echo "=== $1  skipped (done)"; return; fi
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}
GPDP="$PY -m hamlab.solvers.gpdp --single-vgp --alpha $ALPHA --num-iterations 20 --no-policy-time --policy-memory-learn"
RUNS=""
for v in $VARIANTS; do
    extra=""; [ "$v" = "h" ] && extra="--no-policy-memory-zk"
    for s in 0 1 2 3 4; do
        run gpdp_memlearn_${T}${v}_s$s $GPDP $extra --seed $s --run-name memlearn_${T}${v}_s$s
        RUNS="$RUNS $R/gpdp/memlearn_${T}${v}_s$s"
    done
done
run memlearn_${T}evaluate $PY experiments/gpdp/step4_evaluate.py --runs $RUNS --out memlearn_${T}evaluate.json
run memlearn_${T}relative_errors $PY experiments/gpdp/relative_errors.py --runs $RUNS --out relative_errors_memlearn${TAG:+_$TAG}.json
echo "all done"
