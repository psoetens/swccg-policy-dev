#!/usr/bin/env bash
# Plan scratch-24h-ab: the decayed branch of a trunk checkpoint and its checks. From a trunk milestone (with its
# optimizer state) a short run at the trunk's settings whose learning rates fall by a cosine to the floor over
# BRANCH_GAMES games; then its final plays D and Rando Cal (greedy, the held-out Decipher pairings, paired seats). The trunk is paused throughout
# (scripts/arm_pause.sh) unless it has ended. The caller holds the evaluation lock (scripts/pair_watch.sh).
#   bash scripts/branch_decay.sh ARM MARK SOURCE.pt
# Settings come from the environment (pair_watch.sh exports them from runs/dojo/<chain>.conf):
#   BRANCH_GPU (nvidia-smi index: learner and rollouts), BRANCH_THREADS, BRANCH_GAMES, BRANCH_SEED, POOL, OPPONENTS,
#   RECIPE (the trunk's recipe options), CHECK_GAMES, CHECK_THREADS.
# Plan scratch-24h-ab amendment 2: an optional runs/dojo/ARM.branch.env overrides them per arm (e.g. BRANCH_LEARNER_GPU,
# the learner on another GPU than the rollouts, and BRANCH_THREADS) for marks >= BRANCH_ENV_FROM_MARK and the final.
# Output: runs/dojo/ARM-bMARK/final.pt, evaluations bc-ARM-MARK-{d,rando}; one JSON line on stdout:
#   {"arm", "mark", "branch", "status", "branch_games", "stale_share", "d": {...}, "rando": {...}, "paused_seconds"}
set -u
cd "$(dirname "$0")/.."
ARM=${1:?arm} MARK=${2:?mark} SOURCE=${3:?source checkpoint}
R=runs/dojo EVALS=runs/knowledge-v3.1/evaluations
D=$PWD/runs/critic-noise/D.pt
BRANCH=$ARM-b$MARK
if [ -f "$R/$ARM.branch.env" ]; then
    # shellcheck disable=SC1090
    { [ "$MARK" = final ] || [ "$MARK" -ge "$(source "$R/$ARM.branch.env"; echo "${BRANCH_ENV_FROM_MARK:-0}")" ]; } \
        && source "$R/$ARM.branch.env"
fi
BRANCH_LEARNER_GPU=${BRANCH_LEARNER_GPU:-$BRANCH_GPU}
log=$R/$ARM.branch.log
note() { echo "$(date -Is) $*" >> "$log"; }
export PYTHONPATH=scripts:src
source env.sh > /dev/null 2>&1
source scripts/machine_env.sh > /dev/null 2>&1
mkdir -p "$R/evals"
paused=0
resume() { [ "$paused" = 1 ] && note "$(bash scripts/arm_pause.sh resume "$ARM" 2>&1)"; paused=0; }
trap resume EXIT
trap 'exit 1' TERM INT HUP
started=$(date +%s)
if [ ! -f "$R/$ARM.DONE" ] && [ ! -f "$R/$ARM.STOPPED" ]; then
    out=$(bash scripts/arm_pause.sh pause "$ARM" 2>&1) && paused=1
    note "$out"
fi
if [ ! -f "$R/$BRANCH/final.pt" ]; then
    rm -rf "${R:?}/${BRANCH:?}" "/tmp/swccg-v31-$BRANCH"
    note "$BRANCH: from $SOURCE, $BRANCH_GAMES games, learner GPU $BRANCH_LEARNER_GPU, rollouts GPU $BRANCH_GPU, $BRANCH_THREADS threads, seed $BRANCH_SEED"
    # shellcheck disable=SC2086
    .venv/bin/python scripts/v31_stage.py --configuration V31-c6 --purpose training --runs "$R" --variant a10c \
        --logit-cap 30 --opponents "$OPPONENTS" --load-workers 4 --rollout-games 256 --single-tables --critic-stream \
        --prefetch --keep-consumed 4 --minibatch 64 --lr 5e-5 --value-lr 3e-4 --entropy-coef 0.01 \
        --pool "$POOL" --opponent-pool "$POOL" --threads "$BRANCH_THREADS" --actor-fp32 \
        --lr-schedule wsd --lr-schedule-games "$BRANCH_GAMES" --lr-warmup-games 0 --lr-decay-fraction 1.0 --lr-floor 0.05 \
        --max-games "$BRANCH_GAMES" --minutes 60 --learner-device "$BRANCH_LEARNER_GPU" --rollout-gpus "$BRANCH_GPU" \
        --seed "$BRANCH_SEED" --start "$SOURCE" --tag "$BRANCH" $RECIPE --critic-warmup 0 \
        > "$R/$BRANCH.supervisor.log" 2>&1
    note "$BRANCH: stage exit $? ($(cat "$R/$BRANCH/result.json" 2>/dev/null | head -c 200))"
fi
check() {  # name, opponents, extra args -> evaluation tag
    local etag=bc-$ARM-$MARK-$1
    if ! grep -q '"complete": true' "$EVALS/$etag/results.json" 2>/dev/null; then
        rm -rf "${EVALS:?}/${etag:?}"
        # shellcheck disable=SC2086
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$BRANCH_GPU" .venv/bin/python scripts/evaluate_v31.py \
            --checkpoint "$PWD/$R/$BRANCH/final.pt" --reference "$D" --tag "$etag" --threads "$CHECK_THREADS" \
            --games "$CHECK_GAMES" --opponents "$2" --temperature 0 --seed 919001 $3 > "$R/evals/$etag.log" 2>&1
        note "$etag: exit $?"
        rm -rf "$EVALS/$etag/agent.ts" "$EVALS/$etag/agent.ts.bundle" "$EVALS/$etag/reference.ts" \
            "$EVALS/$etag/reference.ts.bundle" "$EVALS/$etag/classes"
    fi
    echo "$etag"
}
status=missing d_tag='' rando_tag=''
if [ -f "$R/$BRANCH/final.pt" ]; then
    status=ok
    d_tag=$(check d initial "")
    rando_tag=$(check rando rando "")
fi
resume
.venv/bin/python - "$ARM" "$MARK" "$BRANCH" "$status" "$R" "$EVALS" "$d_tag" "$rando_tag" "$(( $(date +%s) - started ))" <<'EOF'
import json, sys
from pathlib import Path
arm, mark, branch, status, R, EVALS, d_tag, rando_tag, seconds = sys.argv[1:]
row = {'arm': arm, 'mark': mark, 'branch': branch, 'status': status, 'paused_seconds': int(seconds)}
learner = Path(R) / branch / 'learner.jsonl'
if learner.exists():
    rows = [json.loads(line) for line in learner.open() if line.strip()]
    steps = sum(r['steps'] for r in rows)
    stale = sum(r['discarded_stale_steps'] for r in rows)
    row.update(branch_games=sum(r['games'] for r in rows), stale_share=stale / max(steps + stale, 1),
               actor_lr_last=rows[-1]['actor_lr'] if rows else None)
for name, tag in (('d', d_tag), ('rando', rando_tag)):
    path = Path(EVALS) / tag / 'results.json'
    if tag and path.exists():
        result = json.loads(path.read_text())
        first = next(iter(result['results'].values()))
        row[name] = {'tag': tag, 'rate': first['win_rate_all_games'], 'ci95': first['ci95_all_games'],
                     'games': first['games'], 'unfinished': first['game_length']['unfinished_rate'],
                     'complete': result.get('complete', False)}
print(json.dumps(row))
EOF
