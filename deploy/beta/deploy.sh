#!/usr/bin/env bash
# Plan beta-server: put the beta server on the VPS over SSH (images copied with docker save/load: no registry, no
# builds on the VPS). Everything lives in /opt/brainiac on the VPS.
#   deploy/beta/deploy.sh push [RELEASE_DIR]   # images of the current (pushed) commits + compose files + the release
#   deploy/beta/deploy.sh init                 # first time only: /opt/brainiac/.env with passwords made on the VPS
#   deploy/beta/deploy.sh up [RELEASE_NAME]    # (re)start on the pushed tag with that release
#   deploy/beta/deploy.sh promote NAME         # make a registered account admin (Peter's own)
#   deploy/beta/deploy.sh backup-install       # nightly DB dump at 04:00, 14 kept (/opt/brainiac/backups)
#   deploy/beta/deploy.sh harden               # SSH: keys only
#   deploy/beta/deploy.sh status
# Env: BETA_SSH (ssh target, default root@2a01:4f9:c014:53f4::1), BETA_KEY (~/.ssh/claude-swccg), SITE for init
# (brainiac-beta.duckdns.org). The live server runs pushed commits only (plan Amendment 1): push refuses otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/beta"
source "$ROOT/scripts/machine_env.sh"  # .env: SWCCG_GEMP_REPO etc.
TARGET=${BETA_SSH:-root@2a01:4f9:c014:53f4::1}
SSH=(ssh -i "${BETA_KEY:-$HOME/.ssh/claude-swccg}" -o IdentitiesOnly=yes "$TARGET")
R=/opt/brainiac
GEMP_REPO="${SWCCG_GEMP_REPO:-$ROOT/.cache/gemp-swccg}"
GEMP_BRANCH="${SWCCG_GEMP_BRANCH:-feat/brainiac}"
DEV_SHA=$(git -C "$ROOT" rev-parse --short=9 HEAD)
GEMP_SHA=$(git -C "$GEMP_REPO" rev-parse --short=9 "$GEMP_BRANCH")
TAG=dev-$DEV_SHA-gemp-$GEMP_SHA
remote() { "${SSH[@]}" "$@"; }
pushed() {  # repo commit url-pattern: is the commit on a branch of the remote whose URL matches (after a fetch)?
    local remote
    remote=$(git -C "$1" remote -v | awk -v p="$3" '$2 ~ p {print $1; exit}')
    [ -n "$remote" ] && git -C "$1" fetch -q "$remote" && [ -n "$(git -C "$1" branch -r --contains "$2" --list "$remote/*")" ]
}
case "${1:-}" in
  push)
    RELEASE=$(cd "${2:-$ROOT/models/brainiac/releases/v-cards-shc095-2026-10-23}" && pwd)
    [ -z "$(git -C "$ROOT" status --porcelain)" ] || { echo "dev repo has uncommitted changes"; exit 1; }
    pushed "$ROOT" HEAD psoetens/swccg-policy-dev || { echo "dev commit $DEV_SHA is not pushed (live runs pushed commits only)"; exit 1; }
    pushed "$GEMP_REPO" "$GEMP_BRANCH" psoetens/gemp-swccg || { echo "gemp $GEMP_SHA is not pushed"; exit 1; }
    git -C "$ROOT" ls-files --error-unmatch "$RELEASE/manifest.json" > /dev/null 2>&1 \
        || { echo "release $RELEASE is not committed in the dev repo"; exit 1; }
    docker image inspect "brainiac-beta-gemp:$TAG" > /dev/null 2>&1 || "$HERE/build.sh" > /dev/null
    docker image inspect caddy:2.8.4 > /dev/null 2>&1 || docker pull -q caddy:2.8.4
    remote "mkdir -p $R/brainiac/releases $R/brainiac/runtime $R/backups"
    echo "copying images ($TAG)"
    docker save "brainiac-beta-gemp:$TAG" "brainiac-beta-db:$TAG" caddy:2.8.4 | gzip -1 | remote "gunzip | docker load"
    tar -C "$HERE" -cf - compose.yml Caddyfile backup.sh | remote "tar -C $R -xf -"
    echo "copying release $(basename "$RELEASE")"
    RUNTIME=$("$ROOT/.venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime"])' "$RELEASE/manifest.json")
    tar -C "$(dirname "$(dirname "$RELEASE")")" -cf - "releases/$(basename "$RELEASE")" "runtime/$RUNTIME" \
        | remote "mkdir -p $R/brainiac && tar -C $R/brainiac -xf -"
    echo "pushed $TAG; release $(basename "$RELEASE"); next: deploy.sh up $(basename "$RELEASE")"
    ;;
  init)
    remote "set -e; cd $R; [ -f .env ] && { echo '.env exists; left as is'; exit 0; }
        umask 077; rand() { head -c 32 /dev/urandom | base64 | tr -dc A-Za-z0-9 | head -c 32; }
        printf 'PROJECT=brainiac-beta\nDB_ROOT_PASSWORD=%s\nDB_PASSWORD=%s\nOPS_USER=betaops\nOPS_PASSWORD=%s\nGEMP_PORT=17101\nSITE=%s\nCADDYFILE=./Caddyfile\n' \
            \$(rand) \$(rand) \$(rand) '${SITE:-brainiac-beta.duckdns.org}' > .env; echo '.env written'"
    ;;
  up)
    NAME=${2:?release name (in $R/brainiac/releases)}
    remote "set -e; cd $R; test -f brainiac/releases/$NAME/manifest.json; docker image inspect brainiac-beta-gemp:$TAG > /dev/null
        sed -i '/^TAG=/d; /^BRAINIAC_DIR=/d; /^RELEASE_NAME=/d' .env
        printf 'TAG=%s\nBRAINIAC_DIR=%s\nRELEASE_NAME=%s\n' $TAG $R/brainiac $NAME >> .env
        docker compose --env-file .env up -d
        for i in \$(seq 1 90); do docker compose logs gemp 2>/dev/null | grep -q 'beta: hall' && break; sleep 2; done
        docker compose logs gemp | grep 'beta: hall'"
    ;;
  promote)
    NAME=${2:?account name}
    [[ "$NAME" =~ ^[A-Za-z0-9_-]{1,10}$ ]] || { echo "unexpected account name"; exit 1; }
    remote "cd $R && set -a && . ./.env && docker compose exec -T db mariadb -uroot -p\"\$DB_ROOT_PASSWORD\" gemp-swccg \
        -e \"UPDATE player SET type = CONCAT(type, 'a') WHERE name = '$NAME' AND type NOT LIKE '%a%'; SELECT name, type FROM player WHERE name = '$NAME';\""
    # Gemp caches logged-in players (CachedPlayerDAO): clear it, as the ops admin, so the change shows at the next reload
    remote "cd $R && set -a && . ./.env && set +a && docker compose exec -T -e OPS_USER -e OPS_PASSWORD gemp bash -c '
        J=\$(mktemp); U=http://localhost:8080/gemp-swccg-server; R=\"Referer: http://localhost:8080/gemp-swccg/\"
        curl -sf -o /dev/null -c \$J -H \"\$R\" --data-urlencode \"login=\$OPS_USER\" --data-urlencode \"password=\$OPS_PASSWORD\" \$U/login \
            && curl -s -b \$J -H \"\$R\" -X POST \$U/admin/clearcache; echo; rm -f \$J'"
    ;;
  backup-install)
    remote "chmod 700 $R/backup.sh && printf '0 4 * * * root $R/backup.sh >> $R/backups/backup.log 2>&1\n' > /etc/cron.d/brainiac-backup && echo installed"
    ;;
  harden)  # SSH keys only (the Hetzner image allows passwords); key login is checked before and after
    remote "printf '# Plan beta-server: SSH keys only\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n' \
        > /etc/ssh/sshd_config.d/00-beta-keys-only.conf && sshd -t && systemctl reload ssh && sshd -T | grep -E '^passwordauthentication'"
    remote true && echo "key login still works"
    ;;
  status)
    remote "cd $R && grep -E '^(TAG|RELEASE_NAME|SITE)=' .env; docker compose ps --format 'table {{.Service}}\t{{.Status}}'; free -h | sed -n 2p; df -h / | tail -1"
    ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
