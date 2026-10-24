#!/usr/bin/env bash
# Plan q-critic stage 2, on this machine, unattended (not named *_chain.sh: hostsync counts those as busy jobs):
#   waits for the stage-1 Q-net (runs/q-critic/qnet-s1.pt) and runs/q-critic/READY (made by hand after the learner's
#   --advantage q is reviewed), then trains arm qc from D (24k games, pool v3b, seed 990001, the cn-ctl recipe with
#   --advantage q --qnet qnet-s1.pt; learner and rollouts on this RTX 5090, 12 threads), then plays the evaluations here
#   against cn-ctl (copied to runs/q-critic/ctl/) and writes runs/q-critic/stage2.json with the plan's verdict.
#   setsid nohup bash scripts/qc_stage2.sh > runs/q-critic/stage2.nohup 2>&1 < /dev/null &
# Log runs/q-critic/stage2.log; markers runs/q-critic/stage2.{pid,DONE,STOPPED}.
set -u
cd "$(dirname "$0")/.."
R=runs/dojo Q=runs/q-critic NAME=qc
log=$Q/stage2.log
EVALS=runs/knowledge-v3.1/evaluations
D=$PWD/runs/critic-noise/D.pt FCTL=$PWD/$Q/ctl/final.pt TCTL=$PWD/$Q/ctl/trunk.pt QNET=$PWD/$Q/qnet-s1.pt
note() { echo "$(date -Is) $*" >> "$log"; }
stop() { note "STOP: $*"; echo "$(date -Is) $*" > "$Q/stage2.STOPPED"; exit 1; }
export PYTHONPATH=scripts:src
source env.sh > /dev/null 2>&1
mkdir -p "$R/evals"
[ -f "$Q/stage2.DONE" ] && exit 0
echo $$ > "$Q/stage2.pid"
note "stage 2 (pid $$): waiting for the Q-net and READY"
until [ -f "$QNET" ] && [ -f "$Q/READY" ]; do sleep 60; done
for f in "$D" "$FCTL" "$TCTL" "$QNET"; do [ -f "$f" ] || stop "missing $f"; done
# ===== The arm
if [ ! -f "$R/$NAME.DONE" ] && [ ! -f "$R/$NAME.STOPPED" ]; then
    note "arm $NAME launched"
    NAME=$NAME START=$D GPU=0 ROLLOUT_GPU=0 THREADS=12 GAMES=24000 WARMUP=1000 SAVE_AT=12000 \
        POOL=data/schedules/rl_pool_decipher_v3b.txt EXTRA="--advantage q --qnet $QNET" \
        bash scripts/lr2_arm.sh >> "$Q/arm.nohup" 2>&1
fi
[ -f "$R/$NAME.final-segment" ] || stop "arm $NAME has no final segment ($(tail -1 $R/$NAME.log))"
FQC=$PWD/$R/$(cat "$R/$NAME.final-segment")/final.pt
TQC=$(ls "$PWD/$R/$NAME"/milestones/trunk.pt "$PWD/$R/$NAME"-r[0-9]/milestones/trunk.pt 2>/dev/null | head -1)
note "arm over: $(cat $R/$NAME.final-segment); stale share $(python3 - "$R/$(cat $R/$NAME.final-segment)/learner.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])][5:]
used = sum(r['steps'] for r in rows); stale = sum(r['discarded_stale_steps'] for r in rows)
print(f'{stale / max(used + stale, 1):.4f}')
PY
)"
# ===== Evaluations (one at a time: one GPU, 16 hardware threads)
evaluate() {  # threads tag checkpoint reference opponents games [extra]
    local th=$1 tag=$2 checkpoint=$3 reference=$4 opponents=$5 games=$6
    shift 6
    if [ -f "$EVALS/$tag/results.json" ] && grep -q '"complete": true' "$EVALS/$tag/results.json"; then return 0; fi
    rm -rf "$EVALS/$tag"
    .venv/bin/python scripts/evaluate_v31.py --checkpoint "$checkpoint" --reference "$reference" --tag "$tag" \
        --threads "$th" --games "$games" --opponents "$opponents" --temperature 0 "$@" > "$R/evals/$tag.log" 2>&1
    note "eval $tag exit $?"
    rm -rf "$EVALS/$tag/agent.ts" "$EVALS/$tag/agent.ts.bundle" "$EVALS/$tag/reference.ts" \
        "$EVALS/$tag/reference.ts.bundle" "$EVALS/$tag/classes"
}
evaluate 14 qc-vs-ctl "$FQC" "$FCTL" initial 512 --seed 919001
evaluate 14 qc-vs-ctl-s2 "$FQC" "$FCTL" initial 1024 --seed 919777
evaluate 14 qc-vs-d "$FQC" "$D" initial 256 --seed 919001
evaluate 14 qc-vs-d-s2 "$FQC" "$D" initial 512 --seed 919777
if [ -n "$TQC" ]; then
    evaluate 14 qc-trunk-vs-ctl "$TQC" "$TCTL" initial 256 --seed 919001
    evaluate 14 qc-trunk-vs-ctl-s2 "$TQC" "$TCTL" initial 512 --seed 919777
fi
evaluate 14 qc-sweep "$FQC" "$D" advanced 1110 --schedule reports/data/dojo-sweep/schedule-v3b.tsv --seed 919001
python3 - "$EVALS" > "$Q/stage2.json" <<'PY'
import json, os, sys
E = sys.argv[1]
def rate(tag, side='initial', pair=True):
    w = g = 0
    for t in ((tag, tag + '-s2') if pair else (tag,)):
        f = f'{E}/{t}/results.json'
        if not os.path.exists(f):
            continue
        x = json.load(open(f))
        if not x.get('complete'):
            continue
        x = x['results'][side]; w += x['wins']; g += x['games']
    return {'rate': round(w / g, 4) if g else None, 'games': g}
r = rate('qc-vs-ctl')
out = {'vs_ctl': r, 'vs_d': rate('qc-vs-d'), 'trunk_vs_ctl': rate('qc-trunk-vs-ctl'),
       'sweep': rate('qc-sweep', 'advanced', pair=False),
       'verdict': (None if r['games'] < 1536 else 'the Q-critic trains a better actor' if r['rate'] >= 0.530
                   else 'harmful' if r['rate'] <= 0.470 else 'not shown')}
print(json.dumps(out, indent=1))
PY
note "summary: $(tr -d '\n ' < $Q/stage2.json)"
touch "$Q/stage2.DONE"
note "stage 2 complete"
