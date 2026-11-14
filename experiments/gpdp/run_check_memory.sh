#!/bin/bash
# GPDP with RSPG's information set: policy GP on (z, K, 8-dim GRU hidden state) with no
# explicit date, single value GP over (z, K, T - t). The GRU is frozen from an RSPG (+ GP critic) run
# trained with --gru-hidden-size 8 (gpfix_h8_s*, same seed). Compared with that RSPG and with GPDP
# with the date. The RSPG runs are trained first if they are missing.
# Usage (from the repository root):  PY=python caffeinate -i bash experiments/gpdp/run_check_memory.sh
set -e
PY=${PY:-python}
LOGS=experiments/gpdp/results/logs
R=runs/endogenous
mkdir -p "$LOGS"
run() {  # run <log name> <command...>
    echo "=== $1  ($(date +%H:%M:%S))"
    local t0=$(date +%s)
    "${@:2}" > "$LOGS/$1.log" 2>&1
    echo "    done in $(( $(date +%s) - t0 ))s"
}

# --- RSPG + GP critic (experiments/gp_critic/run_all.sh's gpfix_s*) with an 8-dim GRU, if missing ---
RSPG="$PY -m hamlab.solvers.gp_critic.rspg_gp --use-gp --horizon 32 --phase-counts 4 2 1 1 --time-input --gru-hidden-size 8"
for s in 0 1 2 3 4; do
    [ -f $R/hsm_rspg_gp/gpfix_h8_s$s/mf_agent_wrapper.pkl ] || run rspg_gpfix_h8_s$s $RSPG --seed $s --run-name gpfix_h8_s$s
done
echo "RSPG (8-dim GRU) runs ready ($(date +%H:%M:%S))"

GPDP="$PY -m hamlab.solvers.gpdp --single-vgp --no-policy-time"
run gpdp_memory_s0 $GPDP --run-name memory_h8_s0 --policy-memory-run $R/hsm_rspg_gp/gpfix_h8_s0
for s in 1 2 3 4; do
    run gpdp_memory_s$s $GPDP --seed $s --run-name memory_h8_s$s --policy-memory-run $R/hsm_rspg_gp/gpfix_h8_s$s --eval-frequency 5
done

NEW=""; for s in 0 1 2 3 4; do NEW="$NEW $R/gpdp/memory_h8_s$s"; done
RS8=""; for s in 0 1 2 3 4; do RS8="$RS8 $R/hsm_rspg_gp/gpfix_h8_s$s"; done
run check_memory_evaluate $PY experiments/gpdp/step4_evaluate.py --runs $NEW $RS8 --out check_memory_evaluate.json
run check_memory_relative_errors $PY experiments/gpdp/relative_errors.py --runs $NEW $RS8 --out relative_errors_memory.json
echo "all done"
