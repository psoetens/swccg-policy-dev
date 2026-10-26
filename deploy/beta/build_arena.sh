#!/usr/bin/env bash
# Plan beta-server Amendment 4: WebGL build of GempArena for the beta (/GempArena/), from a clean commit.
#   deploy/beta/build_arena.sh [GEMPARENA_REPO] [OUT_PARENT]   # -> OUT_PARENT/webgl-<sha>; prints that path
# Unity (UNITY, default the Hub's 6000.2.13f1) with WebGL Build Support, batch mode, the project's own build profile
# "New Web Profile" (gzip with decompression fallback: Caddy needs no Content-Encoding headers). In a browser GempArena
# takes its Gemp server from the page's address, so the build is not tied to a host name.
set -euo pipefail
REPO=$(cd "${1:-$HOME/src/GempArena}" && pwd)
PARENT=${2:-$HOME/src/GempArena-builds}
UNITY=${UNITY:-$HOME/Unity/Hub/Editor/6000.2.13f1/Editor/Unity}
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "$REPO has uncommitted changes" >&2; exit 1; }
SHA=$(git -C "$REPO" rev-parse --short=9 HEAD)
OUT=$PARENT/webgl-$SHA
mkdir -p "$PARENT"
# The texture importer's crunch compressor can crash on a fresh import (sem_wait failure); rerunning resumes it.
nice -n 10 "$UNITY" -batchmode -quit -nographics -projectPath "$REPO" \
    -activeBuildProfile "Assets/Settings/Build Profiles/New Web Profile.asset" -build "$OUT" -logFile "$PARENT/build-$SHA.log" >&2 \
    || { echo "Unity failed; log $PARENT/build-$SHA.log" >&2; exit 1; }
[ -f "$OUT/index.html" ] || { echo "no index.html in $OUT" >&2; exit 1; }
printf 'commit=%s\nbranch=%s\nbuilt=%s\nunity=%s\n' "$(git -C "$REPO" rev-parse HEAD)" "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" \
    "$(date -Is)" "$(basename "$(dirname "$(dirname "$UNITY")")")" > "$OUT/BUILD_INFO"
echo "$OUT"
