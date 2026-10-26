#!/usr/bin/env bash
# Plan beta-server §1.1: the beta server on this machine, same images and compose file as the VPS, plain HTTP on
# http://localhost:18080 (localhost only). State in .cache/beta-rehearsal (its .env holds random passwords).
#   deploy/beta/rehearse.sh up [RELEASE_DIR]   # build (if the tag's images are missing), start
#   deploy/beta/rehearse.sh check              # deploy/beta/check.sh against it, with the ops admin
#   deploy/beta/rehearse.sh arena BUILD_DIR    # serve a GempArena build (build_arena.sh, also --allow-dirty) at /GempArena/
#   deploy/beta/rehearse.sh down               # stop (volumes kept)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/beta" S="$ROOT/.cache/beta-rehearsal"
compose() { docker compose -f "$HERE/compose.yml" --env-file "$S/.env" "$@"; }
rand() { head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32; }
case "${1:-}" in
  up)
    RELEASE=$(cd "${2:-$ROOT/models/brainiac/releases/v-cards-shc095-2026-10-23}" && pwd)
    TAG=${TAG:-$("$HERE/build.sh")}
    mkdir -p "$S"
    if [ ! -f "$S/.env" ]; then
        umask 077
        printf 'PROJECT=brainiac-beta-rehearsal\nDB_ROOT_PASSWORD=%s\nDB_PASSWORD=%s\nOPS_USER=betaops\nOPS_PASSWORD=%s\nGEMP_PORT=17181\nSITE=http://localhost:18080\nCADDYFILE=%s\n' \
            "$(rand)" "$(rand)" "$(rand)" "$HERE/Caddyfile.rehearse" > "$S/.env"
    fi
    sed -i '/^TAG=/d; /^RELEASE_DIR=/d; /^BRAINIAC_DIR=/d; /^RELEASE_NAME=/d' "$S/.env"
    printf 'TAG=%s\nRELEASE_DIR=%s\nBRAINIAC_DIR=%s\nRELEASE_NAME=%s\n' "$TAG" "$RELEASE" "$(dirname "$(dirname "$RELEASE")")" \
        "$(basename "$RELEASE")" >> "$S/.env"
    grep -q '^GEMPARENA_DIR=' "$S/.env" || echo "GEMPARENA_DIR=$S/gemparena" >> "$S/.env"
    grep -q '^CLIENTLOG_DIR=' "$S/.env" || printf 'CLIENTLOG_DIR=%s\nCLIENTLOG_PORT=17182\n' "$S/clientlogs" >> "$S/.env"
    mkdir -p "$S/gemparena" "$S/clientlogs"
    compose up -d
    for _ in $(seq 1 90); do
        compose logs gemp 2>/dev/null | grep -q 'beta: hall' && break; sleep 2
    done
    compose logs gemp | grep 'beta: hall' || { echo "gemp did not report; compose logs gemp"; exit 1; }
    echo "rehearsal up: http://localhost:18080/ ($TAG, release $(basename "$RELEASE"))"
    ;;
  check)
    set -a; source "$S/.env"; set +a
    "$HERE/check.sh" http://localhost:18080 "$RELEASE_DIR"
    ;;
  arena)
    BUILD=$(cd "${2:?GempArena WebGL build folder (build_arena.sh output)}" && pwd)
    [ -f "$BUILD/index.html" ] || { echo "$BUILD is not a WebGL build"; exit 1; }
    mkdir -p "$S/gemparena"
    rm -rf "${S:?}/gemparena/$(basename "$BUILD")"
    cp -a "$BUILD" "$S/gemparena/"
    "$HERE/inject_clientlog.sh" "$S/gemparena/$(basename "$BUILD")"
    ln -sfn "$(basename "$BUILD")" "$S/gemparena/current"
    echo "http://localhost:18080/GempArena/ serves $(basename "$BUILD")"
    ;;
  down) compose down ;;
  *) echo "usage: $0 up [RELEASE_DIR] | check | arena BUILD_DIR | down"; exit 1 ;;
esac
