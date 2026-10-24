#!/usr/bin/env bash
# Train a V3.1 policy further for a games budget (plan dev-distribution): scripts/lr2_arm.sh with the defaults the
# lab used for the shc models on 2026-10-24 (pool v3b, the shc recipe, D as a fixed opponent).
#   NAME=my-arm START=models/shc-r1/policy.pt GAMES=20000 scripts/train_arm.sh
#   (unattended: setsid nohup scripts/train_arm.sh > runs/dojo/my-arm.nohup 2>&1 < /dev/null &)
# Schedule over GAMES: warm-up (WARMUP games), constant, cosine to 5% over the last 10%; full checkpoints at SAVE_AT
# and the trunk (milestones/trunk.pt, with optimizer state) where the decay starts. A crash restarts from the newest
# checkpoint as NAME-r1, -r2, ... with the games left.
# Output: runs/dojo/NAME/ (final.pt, learner.jsonl, report.json, ...); log runs/dojo/NAME.log; markers
# runs/dojo/NAME.{DONE,STOPPED,final-segment}. Stop it cleanly: touch runs/dojo/NAME/STOP (see v31_stage.py).
# Env: NAME, START (required); GPU (0), ROLLOUT_GPU (= GPU), THREADS (12), GAMES (20000), WARMUP (GAMES/10, at most
#   8000), SEED (990001), CAP_MINUTES (1440), SAVE_AT (16000,32000), POOL (pool v3b), OPPONENTS, EXTRA (recipe options).
set -u
cd "$(dirname "$0")/.."
GAMES=${GAMES:-20000}
D=$PWD/models/D/policy.pt
[ -f "${D%.pt}.ts" ] || PYTHONPATH=src:scripts .venv/bin/python scripts/export_torchscript.py "$D" || exit 1
export NAME=${NAME:?NAME=<arm name>} START=${START:?START=<checkpoint.pt>} GPU=${GPU:-0} GAMES
export THREADS=${THREADS:-12} WARMUP=${WARMUP:-$(( GAMES / 10 < 8000 ? GAMES / 10 : 8000 ))}
export CAP_MINUTES=${CAP_MINUTES:-1440} POOL=${POOL:-data/schedules/rl_pool_decipher_v3b.txt}
export OPPONENTS=${OPPONENTS:-policy:learner=0.5,policy:snap=0.2,policy:ck:$D=0.15,rando=0.1,random=0.05}
export EXTRA=${EXTRA:---lam 0.95 --value-lambda 0.95 --shaping damage --shaping-weight 0.05 --shaping-form potential --shaping-target both}
mkdir -p runs/dojo
exec bash scripts/lr2_arm.sh
