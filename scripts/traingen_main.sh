#!/usr/bin/env bash
# Run any main class of the Gemp harness (fork, test scope) with the benchmark JDK and the same classpath as scripts/traingen.sh.
#   scripts/traingen_main.sh com.gempukku.swccgo.bench.SamplerCheck in.json out.json
set -euo pipefail
AGENT="${GEMP_AGENT:-$HOME/src/gemp-swccg-agent}"
source "$AGENT/ops/bench-env.sh"
B="$AGENT/build/gemp"
CP="$B/gemp-swccg-server/target/test-classes:$B/gemp-swccg-server/target/classes:$B/gemp-swccg-cards/target/classes:$B/gemp-swccg-logic/target/classes:$B/gemp-swccg-common/target/classes:$(tr ':' '\n' < "$AGENT/build/cp.txt" | grep -v /com/gempukku/ | paste -sd:)"
MAIN=$1; shift
exec java -Xmx${TRAINGEN_HEAP:-2g} ${TRAINGEN_JOPTS:-} --add-opens java.base/java.util=ALL-UNNAMED -cp "$CP" "$MAIN" "$@"
