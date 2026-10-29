#!/usr/bin/env bash
# Plan beta-server Amendment 4: WebGL build of GempArena for the beta (/GempArena/), from a clean commit.
#   deploy/beta/build_arena.sh [--allow-dirty] [GEMPARENA_REPO] [OUT_PARENT]   # -> OUT_PARENT/webgl-<sha>; prints that path
# --allow-dirty (rehearsal only, plan brainiac-gemparena): builds uncommitted changes into webgl-<sha>-dirty, which
# deploy.sh arena refuses.
# Unity (UNITY, default the Hub's 6000.2.13f1) with WebGL Build Support, batch mode, the project's own build profile
# "New Web Profile" (gzip with decompression fallback: Caddy needs no Content-Encoding headers). In a browser GempArena
# takes its Gemp server from the page's address, so the build is not tied to a host name.
set -euo pipefail
DIRTY_OK=0
[ "${1:-}" = --allow-dirty ] && { DIRTY_OK=1; shift; }
REPO=$(cd "${1:-$HOME/src/GempArena}" && pwd)
PARENT=${2:-$HOME/src/GempArena-builds}
UNITY=${UNITY:-$HOME/Unity/Hub/Editor/6000.2.13f1/Editor/Unity}
# Unity imports the card mesh (Assets/Art/Meshes/card.blend) by running Blender; without it every card is broken (a
# bare "Sample Action Choice Text" label instead of the board). BLENDER_DIR: the folder holding the blender executable.
export PATH=${BLENDER_DIR:-$HOME/src/tools/blender-4.2.3-linux-x64}:$PATH
command -v blender > /dev/null || { echo "blender not found (set BLENDER_DIR)" >&2; exit 1; }
SHA=$(git -C "$REPO" rev-parse --short=9 HEAD)
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    [ "$DIRTY_OK" = 1 ] || { echo "$REPO has uncommitted changes" >&2; exit 1; }
    SHA=$SHA-dirty; SUFFIX=" (with uncommitted changes)"
fi
OUT=$PARENT/webgl-$SHA
mkdir -p "$PARENT"
# Unity leaves the files of earlier builds (hashed names) in the output folder: start from an empty one
rm -rf "${OUT:?}"
# The texture importer's crunch compressor can crash on a fresh import (sem_wait failure); rerunning resumes it.
nice -n 10 "$UNITY" -batchmode -quit -nographics -projectPath "$REPO" \
    -activeBuildProfile "Assets/Settings/Build Profiles/New Web Profile.asset" -build "$OUT" -logFile "$PARENT/build-$SHA.log" >&2 \
    || { echo "Unity failed; log $PARENT/build-$SHA.log" >&2; exit 1; }
[ -f "$OUT/index.html" ] || { echo "no index.html in $OUT" >&2; exit 1; }
! grep -q "Blender could not be launched" "$PARENT/build-$SHA.log" \
    || { echo "the card mesh was not imported (Blender); log $PARENT/build-$SHA.log" >&2; exit 1; }
# GempArena plan web-texture-formats: a second data file with the textures in ASTC, for browsers without DXT (iPad,
# iPhone, most Android). Same code, so only the data file is kept; the page template picks it (const astcDataFile).
ASTC_PROFILE="Assets/Settings/Build Profiles/Web ASTC Profile.asset"
if [ -f "$REPO/$ASTC_PROFILE" ]; then
    ASTC_OUT=$PARENT/astc-$SHA
    rm -rf "${ASTC_OUT:?}"
    nice -n 10 "$UNITY" -batchmode -quit -nographics -projectPath "$REPO" \
        -activeBuildProfile "$ASTC_PROFILE" -build "$ASTC_OUT" -logFile "$PARENT/build-$SHA-astc.log" >&2 \
        || { echo "Unity failed (ASTC); log $PARENT/build-$SHA-astc.log" >&2; exit 1; }
    for f in "$OUT"/Build/*.framework.js* "$OUT"/Build/*.wasm*; do
        cmp -s "$f" "$ASTC_OUT/Build/$(basename "$f")" \
            || { echo "the ASTC build's code differs from the DXT build's ($(basename "$f"))" >&2; exit 1; }
    done
    ASTC_DATA=$(cd "$ASTC_OUT/Build" && ls *.data*)
    cp "$ASTC_OUT/Build/$ASTC_DATA" "$OUT/Build/"
    grep -q 'const astcDataFile = "";' "$OUT/index.html" \
        || { echo "index.html has no astcDataFile line (GempArena page template)" >&2; exit 1; }
    sed -i "s/const astcDataFile = \"\";/const astcDataFile = \"$ASTC_DATA\";/" "$OUT/index.html"
    rm -rf "${ASTC_OUT:?}"
fi
"$(dirname "$0")/inject_clientlog.sh" "$OUT"  # plan client-logs
printf 'commit=%s%s\nbranch=%s\nbuilt=%s\nunity=%s\n' "$(git -C "$REPO" rev-parse HEAD)" "${SUFFIX:-}" "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" \
    "$(date -Is)" "$(basename "$(dirname "$(dirname "$UNITY")")")" > "$OUT/BUILD_INFO"
echo "$OUT"
