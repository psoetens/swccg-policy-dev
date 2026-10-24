#!/usr/bin/env bash
# Run a previously compiled overlay; use efficient_java_check.sh to build it first.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/machine_env.sh
# Plan engine-on-feat-brainiac: the engine tree (default v5 since 2026-10-24, built from feat/brainiac; v4 = tag engine-v4); an older tree for comparisons.
SWCCG_ENGINE_TREE="${SWCCG_ENGINE_TREE:-$PWD/.cache/gemp-knowledge-v5}"
SWCCG_V31_DEPS=$(tr ':' '\n' < "$SWCCG_ENGINE_DEPS" | grep -v /com/gempukku/ | paste -sd:)
SWCCG_V31_CP="${SWCCG_V31_CLASSES:?Compiled overlay directory required}"
if [[ -n "${SWCCG_FROZEN_SCHEMA_CLASSES:-}" ]]; then
    SWCCG_V31_CP="$SWCCG_FROZEN_SCHEMA_CLASSES:$SWCCG_V31_CP"
fi
SWCCG_V31_CP="$SWCCG_V31_CP:$SWCCG_ENGINE_TREE/gemp-swccg-server/target/test-classes"
for module in server cards logic common; do
    SWCCG_V31_CP="$SWCCG_V31_CP:$SWCCG_ENGINE_TREE/gemp-swccg-$module/target/classes"
done
# Plan critic-followups BL-49: extra JVM options for the game JVM only (e.g. a deterministic identity hash for exact
# replays: "-XX:+UnlockExperimentalVMOptions -XX:hashCode=2"; javac must not get it). Unset: unchanged.
# shellcheck disable=SC2086
exec java --add-opens java.base/java.util=ALL-UNNAMED -Xmx6g ${SWCCG_GAME_JAVA_OPTS:-} \
    -cp "$SWCCG_V31_CP:$SWCCG_V31_DEPS" "$@"
