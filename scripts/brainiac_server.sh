#!/usr/bin/env bash
# Local Gemp server with Brainiac: play against a policy release in the browser (plan brainiac-v6; dev-distribution).
# The build is Gemp feat/brainiac (the clone scripts/setup.sh makes in .cache/gemp-swccg) plus scripts/brainiac_overlay.py
# (knowledge engine hooks, the shared knowledge package, the v6 Brainiac and its web worker from engine/). It serves
# v6 releases only (scripts/brainiac_release.py; ready ones in models/brainiac/releases/), one release per server.
#
#   scripts/brainiac_server.sh sync     # feat/brainiac (committed tree) -> .cache/brainiac-build, overlay, compile
#   scripts/brainiac_server.sh db       # build + start the MariaDB container from Gemp's own docker/gemp_db.Dockerfile
#   RELEASE=models/brainiac/releases/shc-r1-2026-10-24 scripts/brainiac_server.sh start
#                                       # server on http://localhost:${PORT:-17101}/gemp-swccg/ (background)
#   scripts/brainiac_server.sh stop
#
# Env (or .env): BUILD (.cache/brainiac-build), PORT (17101), RELEASE (release dir, required for start),
#   SWCCG_GEMP_REPO (.cache/gemp-swccg), SWCCG_GEMP_BRANCH (feat/brainiac), LOG (runs/brainiac-e2e/server.log),
#   HALL_DECISION_TIMEOUT_SECONDS (300: stock timers), MVN_OFFLINE (unset; 1 = mvn -o),
#   BRAINIAC_DB_CONTAINER (swccg-brainiac-db), BRAINIAC_DB_PORT (35001), BRAINIAC_DB_PASSWORD (gemp-dev-local),
#   ADMIN_USER / ADMIN_PASSWORD (asdf / asdf: Gemp's seeded admin; sets the fresh hall operational).
# Log in as asdf / asdf (or any seeded user, see the Gemp clone's src/db-scripts/initial_user_setup.sql), open
# "vs Bot" in the hall and pick "Brainiac (<release>)".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/machine_env.sh"
GEMP_REPO="${SWCCG_GEMP_REPO:-$ROOT/.cache/gemp-swccg}"
GEMP_BRANCH="${SWCCG_GEMP_BRANCH:-feat/brainiac}"
B="${BUILD:-$ROOT/.cache/brainiac-build}"
PORT="${PORT:-17101}"
RELEASE="${RELEASE:-}"
LOG="${LOG:-$ROOT/runs/brainiac-e2e/server.log}"
PIDFILE="$B/server.pid"
DB="${BRAINIAC_DB_CONTAINER:-swccg-brainiac-db}"
DB_PORT="${BRAINIAC_DB_PORT:-35001}"
DB_PASSWORD="${BRAINIAC_DB_PASSWORD:-gemp-dev-local}"
MVN=(mvn -q); [ -n "${MVN_OFFLINE:-}" ] && MVN+=(-o)

case "${1:-}" in
  sync)
    mkdir -p "$B"
    # The branch's committed tree, not the clone's working tree (the clone may have another branch checked out).
    EXPORT="$(mktemp -d)"
    git -C "$GEMP_REPO" archive "$GEMP_BRANCH" src | tar -x -m -C "$EXPORT"  # -m: fresh mtimes, Maven recompiles
    rsync -a --delete --exclude target --exclude server.pid --exclude cp-async.txt "$EXPORT/src/" "$B/"
    rm -rf "$EXPORT"
    cd "$B"
    # One reactor build (-am) for both goals: sibling modules resolve from the reactor, not from ~/.m2 (no `mvn install`
    # needed). The async module is built last, so its classpath is the one left in cp-async.txt; the reactor's own
    # module folders are dropped (the start command lists them itself).
    nice -n 19 "${MVN[@]}" -DskipTests compile dependency:build-classpath -pl gemp-swccg-async -am \
        -Dmdep.outputFile="$B/cp-async.txt"
    tr ':' '\n' < "$B/cp-async.txt" | grep -v "^$B/" | paste -sd: > "$B/cp-async.tmp" && mv "$B/cp-async.tmp" "$B/cp-async.txt"
    # Plan brainiac-v6: knowledge hooks, knowledge package and v6 Brainiac compiled into $B/overlay-classes.
    (cd "$ROOT" && PYTHONPATH=scripts:src .venv/bin/python scripts/brainiac_overlay.py "$B")
    ;;
  db)
    [ -f "$B/docker/gemp_db.Dockerfile" ] || { echo "$B has no docker/gemp_db.Dockerfile; run sync first"; exit 1; }
    if ! docker container inspect "$DB" >/dev/null 2>&1; then
      docker build -q -t swccg-brainiac-db -f "$B/docker/gemp_db.Dockerfile" "$B" >/dev/null
      # Gemp's schema and seeded users load on the first start (docker-entrypoint-initdb.d); data in a named volume.
      docker run -d --name "$DB" -p "127.0.0.1:$DB_PORT:3306" -v "$DB-data:/var/lib/mysql" \
        -e MYSQL_ROOT_PASSWORD="$DB_PASSWORD" -e MYSQL_DATABASE=gemp-swccg -e MYSQL_USER=gemp \
        -e MYSQL_PASSWORD="$DB_PASSWORD" swccg-brainiac-db >/dev/null
    else
      docker start "$DB" >/dev/null
    fi
    for i in $(seq 1 60); do
      docker exec "$DB" mariadb -ugemp -p"$DB_PASSWORD" gemp-swccg -e 'select count(*) from player' >/dev/null 2>&1 \
        && { echo "database up: $DB on 127.0.0.1:$DB_PORT"; exit 0; }
      sleep 2
    done
    echo "database did not come up; docker logs $DB"; exit 1
    ;;
  start)
    [ -n "$RELEASE" ] || { echo "RELEASE=<release dir> required (models/brainiac/releases/*)"; exit 1; }
    RELEASE="$(cd "$RELEASE" && pwd)"
    [ -d "$B/overlay-classes" ] || { echo "$B has no overlay (a failed sync?); run sync first"; exit 1; }
    "$0" db
    CLASSES="$B/overlay-classes:$B/gemp-swccg-async/target/classes:$B/gemp-swccg-server/target/classes:$B/gemp-swccg-cards/target/classes:$B/gemp-swccg-logic/target/classes:$B/gemp-swccg-common/target/classes"
    THIRD_PARTY="$(tr ':' '\n' < "$B/cp-async.txt" | grep -v '/com/gempukku/' | paste -sd:)"
    mkdir -p "$(dirname "$LOG")"
    cd "$B"
    DB_CONTAINER_NAME="127.0.0.1:$DB_PORT" MYSQL_DATABASE=gemp-swccg MYSQL_USER=gemp MYSQL_PASSWORD="$DB_PASSWORD" \
    INTERNAL_PORT="$PORT" WEB_PATH="$B/gemp-swccg-async/src/main/web/" BRAINIAC_RELEASE_PATH="$RELEASE" \
    HALL_DECISION_TIMEOUT_SECONDS="${HALL_DECISION_TIMEOUT_SECONDS:-300}" \
      nohup nice -n 5 java -Xmx4g --add-opens java.base/java.util=ALL-UNNAMED -Dlog4j.configurationFile="$ROOT/scripts/brainiac-log4j.xml" \
      -Dpolicy.knowledge=true -Dpolicy.context=5 -Dpolicy.unknownKeysDir="$ROOT/runs/unknown-keys" -cp "$CLASSES:$THIRD_PARTY" com.gempukku.swccgo.async.SwccgoAsyncServer > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for i in $(seq 1 60); do
      if curl -sf -o /dev/null "http://localhost:$PORT/gemp-swccg/"; then
        # A fresh hall refuses games until an admin sets it operational.
        JAR="$(mktemp)"; URL="http://localhost:$PORT/gemp-swccg-server"; REF="Referer: http://localhost:$PORT/gemp-swccg/"
        curl -sf -o /dev/null -c "$JAR" -H "$REF" --data "login=${ADMIN_USER:-asdf}&password=${ADMIN_PASSWORD:-asdf}" "$URL/login" \
          && curl -sf -o /dev/null -b "$JAR" -H "$REF" --data "enabled=false" "$URL/admin/shutdown" \
          && echo "hall operational" || echo "hall NOT operational (admin login failed)"
        rm -f "$JAR"
        echo "server up on http://localhost:$PORT/gemp-swccg/ (pid $(cat "$PIDFILE")), release $(basename "$RELEASE")"; exit 0
      fi
      sleep 2
    done
    echo "server did not come up; see $LOG"; exit 1
    ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && echo stopped || echo "not running"
    ;;
  *) echo "usage: $0 sync | db | start | stop"; exit 1 ;;
esac
