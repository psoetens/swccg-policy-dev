#!/usr/bin/env bash
# Plan card-identity-combos: M3, M4, M6 and M7 on the three finals (scripts/identity_probe.py), one after the other.
#   setsid nohup bash scripts/cic_identity.sh > runs/dojo/cic-identity.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts:src
D=reports/data/card-identity-combos
RUNS=(--run runs/dojo/npv-v3 --run runs/dojo/nvf-a-r4 --run runs/dojo/nvf-b-wsd-r-r1)
log=runs/dojo/cic-identity.log
note() { echo "$(date -Is) $*" >> "$log"; }
note start
for step in m4 m6 m7 m3; do
    [ -f "$D/$step.json" ] && continue
    .venv/bin/python scripts/identity_probe.py $step "${RUNS[@]}" --out "$D/$step.json" --n 3000 --decisions 2000 \
        --max 3000 --combos "$D/combo-map.json" > "runs/dojo/cic-identity-$step.log" 2>&1
    note "$step exit $?"
done
note done
