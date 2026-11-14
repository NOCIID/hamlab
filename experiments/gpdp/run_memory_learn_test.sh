#!/bin/bash
# Test: the learned-memory rules under-consume at the end of the episode.
# Hypothesis: the minimum length scale (0.2) stops the GP from using the slow clocks near the end.
# Here: default length-scale bound (0.05), noise floor kept, logits clipped to the targets' range.
# Fixed-population check, then the two worst runs of run_memory_learn.sh rerun with this setting.
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
$PY experiments/gpdp/memory_checks.py --min-lengthscale 0.05 --out memory_checks_v3.json > $LOGS/memory_checks_v3.log 2>&1
GPDP="$PY -m hamlab.solvers.gpdp --single-vgp --alpha 1.0 --num-iterations 20 --no-policy-time --policy-memory-learn --memory-min-lengthscale 0.05"
$GPDP --seed 3 --run-name memlearn_v3_zkh_s3 > $LOGS/gpdp_memlearn_v3_zkh_s3.log 2>&1
$GPDP --no-policy-memory-zk --seed 4 --run-name memlearn_v3_h_s4 > $LOGS/gpdp_memlearn_v3_h_s4.log 2>&1
$PY experiments/gpdp/step4_evaluate.py --runs $R/gpdp/memlearn_v3_zkh_s3 $R/gpdp/memlearn_v3_h_s4 --out memlearn_v3_evaluate.json > $LOGS/memlearn_v3_evaluate.log 2>&1
$PY experiments/gpdp/relative_errors.py --runs $R/gpdp/memlearn_v3_zkh_s3 $R/gpdp/memlearn_v3_h_s4 --out relative_errors_memlearn_v3.json > $LOGS/memlearn_v3_relative_errors.log 2>&1
echo "all done"
