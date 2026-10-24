#!/usr/bin/env bash
# Plan eval-throughput: freeze / resume a running arm so an evaluation gets the whole machine (or its GPU pair).
#   bash scripts/arm_pause.sh pause <arm>    SIGSTOP the descendants of runs/dojo/<arm>.pid (stage supervisor, learner,
#                                            rollout JVM: they run in their own sessions, so the tree is walked by
#                                            parent PID); the frozen PIDs go to runs/dojo/<arm>.paused
#   bash scripts/arm_pause.sh resume <arm>   SIGCONT exactly those PIDs (reverse order) and remove the file
#   bash scripts/arm_pause.sh state <arm>    the frozen PIDs and their process states
# Nothing restarts: rollouts in flight, the queue and the policy lag stay as they were. Wall-clock timers of the arm
# (archive interval, the stage's time cap) keep counting during a pause.
set -u
cd "$(dirname "$0")/.."
ACTION=${1:?pause|resume|state} ARM=${2:?arm}
R=runs/dojo
file=$R/$ARM.paused
descendants() {  # pid -> its descendants, parents first
    local child
    for child in $(ps -o pid= --ppid "$1"); do
        echo "$child"
        descendants "$child"
    done
}
case "$ACTION" in
    pause)
        [ -f "$file" ] && { echo "already paused: $file" >&2; exit 1; }
        root=$(cat "$R/$ARM.pid" 2>/dev/null) || { echo "no $R/$ARM.pid" >&2; exit 1; }
        kill -0 "$root" 2>/dev/null || { echo "arm $ARM (pid $root) is not running" >&2; exit 1; }
        pids=$(descendants "$root")
        [ -n "$pids" ] || { echo "arm $ARM has no processes to pause" >&2; exit 1; }
        echo "$pids" > "$file"
        # shellcheck disable=SC2086
        kill -STOP $pids 2>/dev/null
        # a child started between the walk and the signal: one more pass (the frozen supervisor starts no more)
        for pid in $(descendants "$root"); do
            grep -qx "$pid" "$file" || { kill -STOP "$pid" 2>/dev/null; echo "$pid" >> "$file"; pids="$pids $pid"; }
        done
        echo "paused $ARM: $(echo $pids | wc -w) processes";;
    resume)
        [ -f "$file" ] || exit 0
        # shellcheck disable=SC2046
        kill -CONT $(tac "$file") 2>/dev/null
        rm -f "$file"
        echo "resumed $ARM";;
    state)
        [ -f "$file" ] || { echo "not paused"; exit 0; }
        # shellcheck disable=SC2046
        ps -o pid=,stat=,comm= -p $(paste -sd, "$file");;
    *) echo "unknown action $ACTION" >&2; exit 2;;
esac
