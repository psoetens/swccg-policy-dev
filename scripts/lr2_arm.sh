#!/usr/bin/env bash
# Plan links-round2 phase 3: a copy of dna_arm.sh (which the phase-2 arms were running when this was written) with
# ROLLOUT_GPU, rollout inference on another GPU than the learner.
# Plan dilution-names-ab: one arm (C = trunk, A = fix 1, B = fix 1 + 2) on the server. A games budget (GAMES arm
# games after START), the warm-up/stable/decay schedule over it (warm-up 8k, constant, cosine to 5% over the last
# 10%), full checkpoints at +16k and +32k for the in-run checks and the trunk checkpoint where the decay starts. The
# recipe of nvf-b (fp32 actor, pool v3, entropy 0.01, the same seed for every arm). A crash restart continues from the
# newest checkpoint with the remaining games and the schedule offset (games are counted from START's rl_meta).
#   NAME=dna-c START=... GPU=0 setsid nohup bash scripts/dna_arm.sh > runs/dojo/dna-c.nohup 2>&1 < /dev/null &
# Log: runs/dojo/NAME.log; markers runs/dojo/NAME.{pid,final-segment,DONE,STOPPED}.
set -u
cd "$(dirname "$0")/.."
R=runs/dojo
NAME=${NAME:?} START=${START:?} GPU=${GPU:?}
THREADS=${THREADS:-14} GAMES=${GAMES:-60000} WARMUP=${WARMUP:-8000} SEED=${SEED:-990001}
CAP_MINUTES=${CAP_MINUTES:-1500}
# Plan links-round2: milestone marks and extra stage options (e.g. "--gamma 0.9995 --pair-weights FILE"), word-split.
SAVE_AT=${SAVE_AT:-16000,32000} EXTRA=${EXTRA:-}
# Plan links-round2 phase 3: rollout inference on another GPU than the learner (default: the learner's).
ROLLOUT_GPU=${ROLLOUT_GPU:-$GPU}
# Plan scratch-24h-ab: the rollout opponent mix (default: the links-round2 mix).
OPPONENTS=${OPPONENTS:-policy:learner=0.6,policy:snap=0.25,random=0.05,advanced=0.1}
log=$R/$NAME.log
note() { echo "$(date -Is) $*" >> "$log"; }
stop_chain() { note "STOP: $*"; echo "$(date -Is) $*" > "$R/$NAME.STOPPED"; exit 1; }
source scripts/objective_lib.sh
export PYTHONPATH=scripts:src
[ -f "$R/$NAME.DONE" ] && exit 0
echo $$ > "$R/$NAME.pid"
[ -f "$START" ] || stop_chain "missing $START"
checkpoint_games() {
    .venv/bin/python -c "import sys,torch;print(int(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['rl_meta']['games']))" "$1"
}
START_GAMES=$(checkpoint_games "$START")
TRUNK=$(( GAMES * 9 / 10 ))
note "arm start (pid $$): $GAMES games from $START ($START_GAMES), GPU $GPU (rollouts $ROLLOUT_GPU), $THREADS threads, seed $SEED${EXTRA:+, extra: $EXTRA}"
POOL=${POOL:-data/schedules/rl_pool_decipher_v3.txt}  # plan links-round2 phase 2: v3b
.venv/bin/python scripts/dojo_coverage.py "$POOL" >> "$log" 2>&1 || stop_chain "Dojo coverage gate failed"
COMMON=(--configuration V31-c6 --purpose training --runs "$R" --variant a10c --logit-cap 30
        --opponents "$OPPONENTS"
        --load-workers 4 --rollout-games 512 --single-tables --critic-stream --prefetch
        --archive-minutes 40 --keep-consumed 40 --minibatch 64 --lam 0.95 --lr 5e-5 --value-lr 3e-4 --threads "$THREADS"
        --pool "$POOL" --opponent-pool "$POOL" --entropy-coef 0.01 --lr-schedule wsd --lr-schedule-games "$GAMES"
        --lr-warmup-games "$WARMUP" --lr-decay-fraction 0.1 --lr-floor 0.05 --save-at-games "$SAVE_AT"
        --trunk-games "$TRUNK" --actor-fp32 --learner-device "$GPU" --rollout-gpus "$ROLLOUT_GPU")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && COMMON+=($EXTRA)
stage_alive() { pgrep -f "[v]31_stage.py .*--tag $1( |$)" > /dev/null; }
start=$START used=0
for n in 0 1 2 3 4; do
    segment=$NAME; [ "$n" -gt 0 ] && segment=$NAME-r$n
    echo "$segment" > "$R/$NAME.active-segment"
    done_games=$(( $(checkpoint_games "$start") - START_GAMES ))
    left=$((GAMES - done_games)) minutes=$((CAP_MINUTES - used))
    if [ -z "$(run_reason "$R/$segment")" ]; then
        if stage_alive "$segment"; then
            note "$segment: stage still running (chain relaunched); waiting for it"
            while stage_alive "$segment"; do sleep 60; done
        elif [ ! -d "$R/$segment" ]; then
            if [ "$left" -lt 1000 ] || [ "$minutes" -lt 15 ]; then note "$NAME: $left games / $minutes min left; ending"; break; fi
            wait_disk "$segment" 30
            note "$segment: start from $start, $left games (offset $done_games), cap $minutes min"
            .venv/bin/python scripts/v31_stage.py "${COMMON[@]}" --tag "$segment" --minutes "$minutes" \
                --max-games "$left" --lr-schedule-offset-games "$done_games" --seed $((SEED + n)) --start "$start" \
                > "$R/$segment.supervisor.log" 2>&1
        fi
    fi
    reason=$(run_reason "$R/$segment")
    used=$((used + $(run_minutes "$R/$segment")))
    note "$segment ended: ${reason:-no result (dead stage)}; $used min used"
    case "$reason" in
        games_complete|duration_complete) echo "$segment" > "$R/$NAME.final-segment"; break ;;
        user_stop) [ -f "$R/$segment.stalled" ] || stop_chain "$segment stopped by a STOP file" ;;
        resource_guard|gpu_memory_guard) wait_disk "$segment-restart" 40 ;;
    esac
    if [ -f "$R/$segment/final.pt" ]; then
        start=$PWD/$R/$segment/final.pt
    else
        newest=$(ls "$R/$segment"/archive/learner-*.pt 2>/dev/null | sort -V | tail -1)
        [ -n "$newest" ] && start=$PWD/$newest
    fi
    note "$segment: ${reason:-dead}; next segment from $start"
    sleep 60
done
if [ ! -f "$R/$NAME.final-segment" ]; then
    for n in 4 3 2 1 0; do
        segment=$NAME; [ "$n" -gt 0 ] && segment=$NAME-r$n
        if [ -f "$R/$segment/final.pt" ]; then
            note "$NAME: ends on $segment ($(run_reason "$R/$segment")), short of its budget"
            echo "$segment" > "$R/$NAME.final-segment"; break
        fi
    done
fi
[ -f "$R/$NAME.final-segment" ] || stop_chain "no usable final segment"
for segment in $(ls -d "$R/$NAME" "$R/$NAME"-r[0-9] 2>/dev/null | xargs -n1 basename); do
    [ -f "$R/$segment.collapse.txt" ] || .venv/bin/python scripts/check_v31_stage_collapse.py "$R/$segment" --history \
        --rules a13 > "$R/$segment.collapse.txt" 2>&1
done
touch "$R/$NAME.DONE"
note "$NAME arm complete ($(cat "$R/$NAME.final-segment"))"
