#!/bin/bash
# DeepHAM on the endogenous model, 5 seeds, then the held-out evaluation of
# experiments/gpdp (64 economies, key 777): MFAX's exploitability and CE relative errors.
# Compare with GPDP's runs from experiments/gpdp/run_report_config.sh (report_singlevgp_a1_s*).
# TAG names the batch (default gm1: runs deepham_gm1_s*); EXTRA adds flags, e.g.
#   TAG=fm1 EXTRA="--n-fm 1 --n-gm 0"   (the paper's "1st moment" variant: mean wealth instead of Q)
# PAR=1 trains the seeds one after the other (default: all in parallel; each process uses ~3 cores).
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/deepham/run_deepham.sh
set -e
PY=${PY:-python}
LOGS=experiments/deepham/results/logs
R=runs/endogenous
TAG=${TAG:-gm1}
EXTRA=${EXTRA:-}
PAR=${PAR:-5}
mkdir -p "$LOGS"
run() {  # run <log name> <command...>; a training run whose log.json exists is skipped (resume)
    local name=$(echo "${@:2}" | sed -n 's/.*--run-name \([^ ]*\).*/\1/p')
    if [ -n "$name" ] && [ -f "$R/deepham/$name/log.json" ]; then echo "=== $1  skipped (done)"; return; fi
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    $1 done in $(( $(date +%s) - t0 ))s"
}
DH="$PY -m hamlab.solvers.deepham $EXTRA"
RUNS=""
for s in 0 1 2 3 4; do
    if [ "$PAR" -gt 1 ]; then
        run deepham_${TAG}_s$s $DH --seed $s --run-name deepham_${TAG}_s$s &
    else
        run deepham_${TAG}_s$s $DH --seed $s --run-name deepham_${TAG}_s$s
    fi
    RUNS="$RUNS $R/deepham/deepham_${TAG}_s$s"
done
wait
run deepham_${TAG}_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $RUNS --out deepham_${TAG}_evaluate.json
run deepham_${TAG}_relative_errors $PY experiments/gpdp/relative_errors.py --runs $RUNS --out relative_errors_deepham_${TAG}.json
echo "all done"
