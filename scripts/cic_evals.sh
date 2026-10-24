#!/usr/bin/env bash
# Plan card-identity-combos: the evaluations behind M1 and M5-M7, on this machine (greedy, with decision exports and
# game logs). The pool-v3 sweep vs AdvancedAI (1,150 games, seed 919001) for P-v3, nvf-a and B; the head to head vs
# P-v3 (256 at seed 919001 + 512 at seed 919777) for nvf-a and B.
#   setsid nohup bash scripts/cic_evals.sh > runs/dojo/cic-evals.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts:src
R=runs/dojo
EVALS=runs/knowledge-v3.1/evaluations
log=$R/cic-evals.log
note() { echo "$(date -Is) $*" >> "$log"; }
PV3=$PWD/$R/npv-v3/final.pt A=$PWD/$R/nvf-a-r4/final.pt B=$PWD/$R/nvf-b-wsd-r-r1/final.pt
evaluate() {  # tag checkpoint reference opponents games [extra]
    local tag=$1 checkpoint=$2 reference=$3 opponents=$4 games=$5
    shift 5
    if [ -f "$EVALS/$tag/results.json" ] && grep -q '"complete": true' "$EVALS/$tag/results.json"; then return 0; fi
    rm -rf "$EVALS/$tag"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_v31.py --checkpoint "$checkpoint" --reference "$reference" \
        --tag "$tag" --threads 16 --games "$games" --opponents "$opponents" --temperature 0 --export --game-log "$@" \
        > "$R/evals/$tag.log" 2>&1
    note "eval $tag exit $?"
}
SWEEP=(--schedule reports/data/dojo-sweep/schedule-v3.tsv --seed 919001)
note start
for m in pv3:$PV3 nvfa:$A b:$B; do
    evaluate "cic-sweep-${m%%:*}" "${m#*:}" "$PV3" advanced 1150 "${SWEEP[@]}"
done
for m in nvfa:$A b:$B; do
    evaluate "cic-h2h-${m%%:*}" "${m#*:}" "$PV3" initial 256
    evaluate "cic-h2h-${m%%:*}-s2" "${m#*:}" "$PV3" initial 512 --seed 919777
done
note done
