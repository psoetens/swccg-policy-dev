#!/usr/bin/env bash
# Offline learner benchmark (plan .claude/plans/learner-throughput.md): one PPO iteration from a fixed checkpoint on a fixed
# batch (runs/bench/lt/batch.txt: v4 backlog files, ~150k steps), v4 flags, phase timing on, reproducible minibatch order.
#   NAME=base-a scripts/learner_bench.sh [extra learner flags...]   -> runs/bench/lt/NAME/{learner.jsonl,summary.json}
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
NAME=${NAME:?NAME}; CKPT=${CKPT:-runs/rl-v4/ckpt/learner-0424.pt}; ITERS=${ITERS:-1}; SEED=${SEED:-1}
R=runs/bench/lt/$NAME
rm -rf "$R"; mkdir -p "$R/ckpt" "$R/traj" "$R/snaps"
cp "$CKPT" "$R/ckpt/latest.pt"
IT0=$(.venv/bin/python -c "import torch; print(torch.load('$CKPT',map_location='cpu',weights_only=False)['rl_meta']['iteration'])")
while read -r f; do cp "$f" "$R/traj/"; done < runs/bench/lt/batch.txt
# ITERS > 1: the same batch again for every further iteration (offline A/B on identical data)
( i=1; while [ "$i" -lt "$ITERS" ]; do until [ -z "$(ls "$R/traj")" ]; do sleep 1; done; sleep 1
    while read -r f; do cp "$f" "$R/traj/"; done < runs/bench/lt/batch.txt; i=$((i + 1)); done ) &
FEED=$!
T0=$(date +%s.%N)
.venv/bin/python -m swccg_policy.rl learn --run "$R" --batch-steps 150000 --bytes-per-step-estimate 169 --value-warmup 2 \
    --kl-start 0 --kl-min 0 --cards data/cards/card_table_v2.npz --seed "$SEED" --seed-all --profile-phases \
    --max-iterations $((IT0 + ITERS)) "$@" > "$R/learner.log" 2>&1 &
LP=$!
peak=0; while kill -0 $LP 2>/dev/null; do r=$(awk '/VmRSS/{print $2}' /proc/$LP/status 2>/dev/null || echo 0); [ "${r:-0}" -gt "$peak" ] && peak=$r; sleep 1; done
wait $LP; T1=$(date +%s.%N); kill $FEED 2>/dev/null || true
.venv/bin/python - "$R" "$T0" "$T1" "$peak" <<'P'
import json, sys
R, t0, t1, peak = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
L = [json.loads(l) for l in open(f"{R}/learner.jsonl")]
it = L[-1]
steps = sum(d["steps"] for d in L)
iter_s = sum(sum(d.get("phases", {}).values()) for d in L)
out = {"name": R.split("/")[-1], "iterations": len(L), "steps": steps, "process_wall_s": round(t1 - t0, 1),
       "iteration_s": round(iter_s, 1), "steps_per_s_iteration": round(steps / iter_s, 1), "peak_rss_mb": peak // 1024,
       "phases": [d.get("phases") for d in L],
       "stats": {k: round(it[k], 5) for k in ("pg", "vloss", "kl_bc", "entropy", "clipfrac", "approx_kl_old", "value_mean", "return_mean")}}
json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps(out))
P
