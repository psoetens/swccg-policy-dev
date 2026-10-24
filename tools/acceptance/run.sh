#!/usr/bin/env bash
# Fresh-user acceptance (plan dev-distribution §1) in a container. From the host:
#   tools/acceptance/run.sh DEV_REPO GEMP_REPO WORKDIR
# DEV_REPO / GEMP_REPO are mounted read-only and cloned (file://); WORKDIR (host) holds the clone and the logs.
# Ports: Brainiac on 17111, its DB on 35011 (host network; the lab's own Gemp uses 17101 / 35001).
set -euo pipefail
DEV=$(cd "$1" && pwd) GEMP=$(cd "$2" && pwd) WORK=$3
mkdir -p "$WORK/home"  # the container user's home (uv Pythons, ~/.m2) persists with the work dir
docker build -q -t swccg-dev-acceptance "$(dirname "$0")" >/dev/null
exec docker run --rm --gpus all --network host --shm-size 8g \
    --group-add "$(stat -c %g /var/run/docker.sock)" -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$DEV:/src/dev:ro" -v "$GEMP:/src/gemp:ro" -v "$WORK:/work" -v "$WORK/home:/home/dev" \
    -e SWCCG_GEMP_URL=file:///src/gemp -e PORT=17111 -e BRAINIAC_DB_PORT=35011 \
    -e BRAINIAC_DB_CONTAINER="swccg-acceptance-db-$(basename "$WORK")" \
    swccg-dev-acceptance bash /src/dev/tools/acceptance/steps.sh
