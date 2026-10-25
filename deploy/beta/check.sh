#!/usr/bin/env bash
# Plan beta-server §1: the checks of a running beta server, from outside (through Caddy).
#   deploy/beta/check.sh BASE_URL RELEASE_DIR        # e.g. http://localhost:18080 models/brainiac/releases/<name>
# Env: OPS_USER / OPS_PASSWORD (optional): also switch registration off (register must then fail with 405) and on.
# 1. every served page carries the beta banner and no Google Tag Manager; 2. Gemp's seeded logins are refused;
# 3. a fresh account registers, logs in, the library lists 132 decks (db-init 45: the official server's 90; 46: the 42
# Brainiac test-rig decks), and the hall offers the release with bot tables on; 4. the served model's
# SHA-256 is the release manifest's, and every browser runtime file (onnxruntime-web) is served unchanged. Exit 0 only if all pass.
set -uo pipefail
BASE=${1:?base url} RELEASE=${2:?release dir}
WEB=$BASE/gemp-swccg SRV=$BASE/gemp-swccg-server REF="Referer: $WEB/"
PY=${PY:-$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python}
fails=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

for page in index hall game deckBuild admin merchant soloDraft; do
    html=$(curl -sf "$WEB/$page.html") || { bad "$page.html not served"; continue; }
    grep -q 'brainiac-beta-banner-ribbon' <<< "$html" && ! grep -q googletagmanager <<< "$html" \
        && ok "$page.html: banner, no analytics" || bad "$page.html: banner missing or analytics present"
done
grep -q 'brainiac-beta-banner-notice' <<< "$(curl -sf "$WEB/index.html")" && ok "index.html: beta notice" || bad "index.html: no notice"

for user in asdf qwer Librarian; do
    for pw in asdf qwer test "$user"; do
        [ "$(code -H "$REF" --data "login=$user&password=$pw" "$SRV/login")" = 200 ] && bad "seeded login $user/$pw accepted"
    done
done
ok "seeded logins refused (asdf, qwer, Librarian with their known/obvious passwords)"

USER=t$(date +%s | tail -c 9) PASS=$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')
JAR=$(mktemp)
c=$(code -c "$JAR" -H "$REF" --data "login=$USER&password=$PASS" "$SRV/register")
[ "$c" = 200 ] && ok "registered $USER" || bad "register $USER: HTTP $c"
c=$(code -c "$JAR" -H "$REF" --data "login=$USER&password=$PASS" "$SRV/login")
[ "$c" = 200 ] && ok "login $USER" || bad "login $USER: HTTP $c"
hall=$(curl -s -b "$JAR" -H "$REF" "$SRV/hall")
decks=$(curl -s "$SRV/deck/libraryList" | grep -oE '<(dark|light)Deck>' | wc -l)
[ "$decks" = "${LIBRARY_DECKS:-132}" ] && ok "library: $decks sample decks" || bad "library: $decks sample decks (not ${LIBRARY_DECKS:-132})"
grep -q 'aiTablesEnabledBoolean="true"' <<< "$hall" && ok "bot tables on" || bad "bot tables off"
attr=$(grep -o 'brainiac="[^"]*"' <<< "$hall" | head -1)
read -r name sha <<< "$("$PY" -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["name"],m["sha256"])' "$RELEASE/manifest.json")"
[[ "$attr" == "brainiac=\"$name|$sha|"* ]] && ok "hall offers $name" || bad "hall offers ${attr:-nothing}, not $name"
got=$(curl -s -b "$JAR" -H "$REF" "$SRV/brainiac/model/$sha.onnx" | sha256sum | cut -d' ' -f1)
[ "$got" = "$sha" ] && ok "served model = manifest sha256" || bad "served model sha256 $got != $sha"
runtime=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime"])' "$RELEASE/manifest.json")
before=$fails n=0
for f in "$(dirname "$(dirname "$RELEASE")")/runtime/$runtime"/*; do
    got=$(curl -s "$SRV/brainiac/runtime/$runtime/$(basename "$f")" | sha256sum | cut -d' ' -f1)
    [ "$got" = "$(sha256sum < "$f" | cut -d' ' -f1)" ] || bad "runtime file $(basename "$f") not served as in the release tree"
    n=$((n + 1))
done
[ "$n" -gt 0 ] || bad "no runtime files for $runtime next to the release"
[ "$fails" = "$before" ] && ok "browser runtime $runtime served ($n files, contents identical)"
rm -f "$JAR"

if [ -n "${OPS_USER:-}" ]; then
    OPS=$(mktemp)
    code -c "$OPS" -H "$REF" --data-urlencode "login=$OPS_USER" --data-urlencode "password=$OPS_PASSWORD" "$SRV/login" > /dev/null
    curl -s -o /dev/null -b "$OPS" -H "$REF" --data "enabled=false" "$SRV/admin/settings/newaccounts"
    c=$(code -H "$REF" --data "login=x${USER:1}&password=$PASS" "$SRV/register")
    [ "$c" = 405 ] && ok "registration off: register refused (405)" || bad "registration off: register HTTP $c"
    curl -s -o /dev/null -b "$OPS" -H "$REF" --data "enabled=true" "$SRV/admin/settings/newaccounts"
    c=$(code -H "$REF" --data "login=y${USER:1}&password=$PASS" "$SRV/register")
    [ "$c" = 200 ] && ok "registration back on" || bad "registration back on: register HTTP $c"
    rm -f "$OPS"
fi
echo "test accounts made: $USER$([ -n "${OPS_USER:-}" ] && echo " y${USER:1}")"
[ "$fails" = 0 ] && echo "ALL CHECKS PASSED" || echo "$fails CHECK(S) FAILED"
[ "$fails" = 0 ]
