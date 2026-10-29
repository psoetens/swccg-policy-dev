#!/usr/bin/env bash
# Plan beta-server: build the two images of the public beta server from the dev repo and Gemp feat/brainiac.
#   deploy/beta/build.sh            # -> images brainiac-beta-gemp:<tag>, brainiac-beta-db:<tag>; prints the tag
# The tag is dev-<dev repo HEAD>[-dirty]-gemp-<feat/brainiac commit>; the live server runs only clean, pushed commits
# (plan Amendment 1; deploy.sh checks that). The Gemp build is scripts/brainiac_server.sh sync into
# .cache/beta-build; the image gets its compiled classes, the third-party jars and the web folder with the beta banner
# (deploy/beta/banner.py; Google Tag Manager removed). The DB image is Gemp's own docker/gemp_db.Dockerfile plus
# deploy/beta/db-init (the official server's other library decks, seeded users removed, Librarian locked, bot tables
# and registration on, the ops admin).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/beta"
source "$ROOT/scripts/machine_env.sh"
GEMP_REPO="${SWCCG_GEMP_REPO:-$ROOT/.cache/gemp-swccg}"
GEMP_BRANCH="${SWCCG_GEMP_BRANCH:-feat/brainiac}"
DEV_SHA=$(git -C "$ROOT" rev-parse --short=9 HEAD)
# Uncommitted changes (a rehearsal before the commit) give a -dirty tag, which deploy.sh never ships.
[ -z "$(git -C "$ROOT" status --porcelain)" ] || DEV_SHA=$DEV_SHA-dirty
GEMP_SHA=$(git -C "$GEMP_REPO" rev-parse --short=9 "$GEMP_BRANCH")
TAG=dev-$DEV_SHA-gemp-$GEMP_SHA
B=$ROOT/.cache/beta-build
STAGE=$ROOT/.cache/beta-image
BUILD=$B "$ROOT/scripts/brainiac_server.sh" sync >&2
# Peter: no decision ("request") timer in games against a bot (deploy/beta/patch_mediator.py, on the build copy).
"$ROOT/.venv/bin/python" "$HERE/patch_mediator.py" "$B" >&2
(cd "$B" && nice -n 19 mvn -q -DskipTests compile -pl gemp-swccg-server -am >&2)
rm -rf "${STAGE:?}"
mkdir -p "$STAGE/classes" "$STAGE/lib"
cp -a "$B/overlay-classes" "$STAGE/classes/0-overlay"
n=1
for module in async server cards logic common; do
    cp -a "$B/gemp-swccg-$module/target/classes" "$STAGE/classes/$n-$module"; n=$((n + 1))
done
tr ':' '\n' < "$B/cp-async.txt" | grep -v '/com/gempukku/' | grep . | while read -r jar; do cp "$jar" "$STAGE/lib/"; done
cp -a "$B/gemp-swccg-async/src/main/web" "$STAGE/web"
# GempArena plan card-data-from-gemp: the card data file GempArena loads (web/gemparena/cards.json), made by the Gemp
# branch's own generator (Gemp's Maven build runs it too); it downloads swccg-card-json, at build time only.
GEN=$(mktemp)
git -C "$GEMP_REPO" show "$GEMP_BRANCH:bin/gemparena_cards" > "$GEN" \
    || { echo "gemp $GEMP_BRANCH has no bin/gemparena_cards (GempArena card data)" >&2; exit 1; }
python3 "$GEN" --web "$STAGE/web" >&2 || { echo "bin/gemparena_cards failed" >&2; exit 1; }
rm -f "$GEN"
"$ROOT/.venv/bin/python" "$HERE/banner.py" "$STAGE/web" "dev $DEV_SHA · gemp $GEMP_SHA" >&2
cp "$HERE/log4j.xml" "$HERE/entrypoint.sh" "$STAGE/"
docker build -q -t "brainiac-beta-gemp:$TAG" -f "$HERE/gemp.Dockerfile" "$STAGE" >&2
# Gemp's DB image, with our init scripts after Gemp's own (10-40)
DBCTX=$ROOT/.cache/beta-db
rm -rf "${DBCTX:?}"; mkdir -p "$DBCTX/docker"
cp -a "$B/db-scripts" "$DBCTX/"
cp "$B/docker/gemp_db.Dockerfile" "$DBCTX/docker/"
cp "$HERE"/db-init/* "$DBCTX/db-scripts/"
{ cat "$B/docker/gemp_db.Dockerfile"; echo
  for f in "$HERE"/db-init/*; do echo "COPY ./db-scripts/$(basename "$f") /docker-entrypoint-initdb.d/$(basename "$f")"; done
} > "$DBCTX/docker/beta_db.Dockerfile"
docker build -q -t "brainiac-beta-db:$TAG" -f "$DBCTX/docker/beta_db.Dockerfile" "$DBCTX" >&2
echo "$TAG"
