#!/usr/bin/env bash
# Run the Gemp TrainGen harness (fork gemp-swccg-agent/gemp-swccg-public, test scope) with the benchmark JDK.
# Build it first: ~/src/gemp-swccg-agent/ops/bench.sh build  (the classpath is written to build/cp-train.txt below).
#   scripts/traingen.sh --schedule data/schedules/train.tsv --threads 8 --export data/shards/v1-train --out ... [--tag X]
set -euo pipefail
AGENT="${GEMP_AGENT:-$HOME/src/gemp-swccg-agent}"
source "$AGENT/ops/bench-env.sh"
B="$AGENT/build/gemp"
CP="$B/gemp-swccg-server/target/test-classes:$B/gemp-swccg-server/target/classes:$B/gemp-swccg-cards/target/classes:$B/gemp-swccg-logic/target/classes:$B/gemp-swccg-common/target/classes:$(tr ':' '\n' < "$AGENT/build/cp.txt" | grep -v /com/gempukku/ | paste -sd:)"
exec java -Xmx${TRAINGEN_HEAP:-12g} ${TRAINGEN_JOPTS:-} --add-opens java.base/java.util=ALL-UNNAMED -cp "$CP" com.gempukku.swccgo.bench.TrainGen --decks / "$@"
