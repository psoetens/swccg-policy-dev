#!/usr/bin/env bash
# Rollout throughput benchmark (plan .claude/plans/rollout-throughput.md): one TrainGen run of $GAMES games with the v4 league and
# fixed weights, with CPU / GPU / perf counters sampled alongside. No learner unless LEARNER=1 (then a learner trains on a copy of
# recorded trajectories on the same GPU, as in a real run).
#   NAME=base-t20 THREADS=20 scripts/rollout_bench.sh          -> runs/bench/base-t20/summary.json
# Env: NAME, THREADS (20), GAMES (1500), CKPT (runs/rl-v4/ckpt/learner-0419.pt), BACKEND (python | native; NATIVE_EXTRA), PORT (47400), PAIRSEED (9001),
#      LEARNER (0), SCHEDULE (a schedule file instead of the v4 pool), TRAINGEN_JOPTS / TRAINGEN_HEAP (passed through), PERF (1: perf stat on the JVM and the server).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
NAME=${NAME:?NAME}; THREADS=${THREADS:-20}; GAMES=${GAMES:-1500}; PORT=${PORT:-47400}; PAIRSEED=${PAIRSEED:-9001}
CKPT=${CKPT:-runs/rl-v4/ckpt/learner-0419.pt}; BACKEND=${BACKEND:-python}; LEARNER=${LEARNER:-0}; PERF=${PERF:-1}
POOL=data/schedules/rl_pool_v4.txt
LEAGUE="policy:learner=0.35,policy:snap=0.3,policy:ck:$PWD/runs/rl-v3n/final/pilot-final.pt=0.15,random=0.1,advanced=0.1"
B=runs/bench/$NAME
rm -rf "$B"; mkdir -p "$B/ckpt" "$B/snaps"
cp "$CKPT" "$B/ckpt/latest.pt"
# snapshots: symlinks to the last 6 (all that the server / PolicyEngine load); copying all of them filled the disk (2026-09-26)
for f in $(ls runs/rl-v4/snaps/*.pt | sort | tail -6); do ln -s "$PWD/$f" "$B/snaps/"; done
if [ "$BACKEND" = native ]; then
  # in-JVM policy (PolicyEngine): TorchScript copies (python -m swccg_policy.native ts ...) instead of the Python server
  cp "${CKPT%.pt}.ts" "$B/ckpt/latest.ts"; rm -f "$B"/snaps/*.pt
  for f in $(ls runs/rl-v4/snaps/*.ts | sort | tail -6); do ln -s "$PWD/$f" "$B/snaps/"; done
  NATIVE_JOPTS="-Dpolicy.native.run=$PWD/$B -Dpolicy.native.bundle=$PWD/native/bundle -Dpolicy.native.bc=$PWD/checkpoints/bc-v1/model.ts \
    -Dpolicy.native.lib=$PWD/native/build/libswccgpolicy.so ${NATIVE_EXTRA:-}"
fi
echo "name $NAME threads $THREADS games $GAMES ckpt $CKPT backend $BACKEND learner $LEARNER pairseed $PAIRSEED league $LEAGUE" > "$B/config.txt"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

SERVER=""
if [ "$BACKEND" = python ]; then
  uv run python -m swccg_policy.rl server --run "$B" --port "$PORT" --graphs > "$B/server.log" 2>&1 &
  PIDS+=($!)
  until grep -q "RL server on" "$B/server.log"; do sleep 1; done
  SERVER=$(pgrep -f "[s]wccg_policy.rl server --run $B " | tail -1)
fi
if [ "$LEARNER" = 1 ]; then
  # A learner on real v4 trajectories, looping over the same files (copied back into traj/ whenever it consumed them).
  mkdir -p "$B/traj" "$B/traj-src"; cp runs/rl-v4/traj-consumed/part-*.jsonl.gz "$B/traj-src/"
  ( while true; do [ -z "$(ls "$B/traj" 2>/dev/null)" ] && cp "$B"/traj-src/*.jsonl.gz "$B/traj/"; sleep 2; done ) &
  PIDS+=($!)
  LEARN_TS=(); [ "$BACKEND" = native ] && LEARN_TS=(--native-ts)  # native: the learner writes latest.ts, the JVM swaps it in live
  uv run python -m swccg_policy.rl learn --run "$B" --batch-steps 150000 --bytes-per-step-estimate 169 --kl-start 0 --kl-min 0 \
      "${LEARN_TS[@]}" > "$B/learner.log" 2>&1 &
  PIDS+=($!)
fi

DECKS=(--pool "$POOL"); [ -n "${SCHEDULE:-}" ] && DECKS=(--schedule "$SCHEDULE")
TRAINGEN_JOPTS="${TRAINGEN_JOPTS:-} ${NATIVE_JOPTS:-} -Dpolicy.loopLog=$PWD/$B/loops.tsv" \
scripts/traingen.sh "${DECKS[@]}" --games "$GAMES" --threads "$THREADS" --agent policy:learner --opponent "$LEAGUE" \
    --agentSide alternate --server 127.0.0.1:"$PORT" --cap 6000 --maxGameMs 900000 --pairSeed "$PAIRSEED" --tag "bench-$NAME" \
    --stackLog "$B/exceptions.log" --out "$B/games.csv" > "$B/traingen.log" 2>&1 &
TG=$!; PIDS+=($TG)
until grep -q "decks loaded" "$B/traingen.log" 2>/dev/null || ! kill -0 $TG 2>/dev/null; do sleep 0.5; done
date +%s.%N > "$B/t_start"
JAVA=$(pgrep -P $TG java || echo $TG)
python3 scripts/rollout_sampler.py --out "$B/samples.csv" --pids "java=$JAVA${SERVER:+,server=$SERVER}" &
PIDS+=($!)
nvidia-smi dmon -s u -d 1 > "$B/gpu.txt" 2>&1 &
PIDS+=($!)
if [ "$PERF" = 1 ]; then
  sleep 10  # perf stat -p attaches to the threads that exist now: wait until the game threads run
  date +%s.%N > "$B/t_perf"
  perf stat -x, -e instructions:u,cycles:u,task-clock,context-switches,cpu-migrations -p "$JAVA" -o "$B/perf-java.csv" &
  PIDS+=($!)
  [ -n "$SERVER" ] && { perf stat -x, -e instructions:u,cycles:u,task-clock,context-switches,cpu-migrations -p "$SERVER" -o "$B/perf-server.csv" & PIDS+=($!); }
fi
wait $TG || true
date +%s.%N > "$B/t_end"
for p in "${PIDS[@]}"; do [ "$p" != "$TG" ] && kill -INT "$p" 2>/dev/null || true; done
sleep 2
python3 scripts/rollout_bench_summary.py "$B" | tee "$B/summary.json"
