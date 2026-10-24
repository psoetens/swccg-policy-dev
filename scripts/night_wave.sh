#!/usr/bin/env bash
# Run a wave of night_probe.py jobs, N at a time. Each JOBS line: NAME ARGS... (output -> OUT_DIR/NAME.json).
# Usage: scripts/night_wave.sh JOBS_FILE OUT_DIR [PARALLEL]
set -u
cd "$(dirname "$0")/.."
jobs="$1"
out="$2"
parallel="${3:-4}"
mkdir -p "$out"
echo "$(date -Is) wave start $jobs" >> "$out/wave.log"
grep -v '^\s*#' "$jobs" | grep -v '^\s*$' | while read -r name rest; do
    printf '%s\0' "$name $rest"
done | xargs -0 -P "$parallel" -I{} bash -c '
    set -- {}
    name="$1"; shift
    if [ -f "'"$out"'/$name.json" ] && grep -q "\"status\": \"complete\"" "'"$out"'/$name.json"; then exit 0; fi
    rm -f -- "'"$out"'/$name.json"
    .venv/bin/python scripts/night_probe.py --out "'"$out"'/$name.json" "$@" > "'"$out"'/$name.log" 2>&1
    echo "$(date -Is) $name exit $?" >> "'"$out"'/wave.log"'
echo "$(date -Is) wave done $jobs" >> "$out/wave.log"
