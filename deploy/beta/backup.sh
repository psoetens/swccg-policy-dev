#!/usr/bin/env bash
# Plan beta-server: nightly dump of the Gemp database on the VPS (cron, deploy.sh backup-install); 14 kept.
set -euo pipefail
cd /opt/brainiac
set -a; . ./.env; set +a
out=backups/gemp-$(date +%Y%m%d-%H%M).sql.gz
docker compose exec -T db mariadb-dump -uroot -p"$DB_ROOT_PASSWORD" --single-transaction gemp-swccg | gzip > "$out"
ls -1t backups/gemp-*.sql.gz | tail -n +15 | xargs -r rm --
echo "$(date -Is) $out $(du -h "$out" | cut -f1)"
