#!/usr/bin/env bash
# Plan scratch-24h-ab amendment 2: end lr2_arm.sh trunks at a wall-clock time. At TIME, a STOP file in each arm's
# active segment (runs/dojo/<arm>.active-segment, read then): the stage ends with reason user_stop and writes final.pt,
# lr2_arm.sh writes runs/dojo/<arm>.STOPPED. A paused trunk sees the STOP once it is resumed.
#   setsid nohup bash scripts/stop_at.sh '2026-10-25 12:00' s24-a s24-b > /dev/null 2>&1 < /dev/null &
# Log runs/dojo/stop_at.log.
set -u
cd "$(dirname "$0")/.."
R=runs/dojo
WHEN=${1:?time} ; shift
target=$(date -d "$WHEN" +%s) || exit 1
echo "$(date -Is) stop_at (pid $$): $* at $(date -d "@$target" -Is)" >> "$R/stop_at.log"
while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
for arm in "$@"; do
    segment=$(cat "$R/$arm.active-segment" 2>/dev/null) || { echo "$(date -Is) $arm: no active segment" >> "$R/stop_at.log"; continue; }
    [ -f "$R/$arm.DONE" ] || [ -f "$R/$arm.STOPPED" ] && { echo "$(date -Is) $arm: already ended" >> "$R/stop_at.log"; continue; }
    touch "$R/$segment/STOP"
    echo "$(date -Is) $arm: STOP in $R/$segment" >> "$R/stop_at.log"
done
