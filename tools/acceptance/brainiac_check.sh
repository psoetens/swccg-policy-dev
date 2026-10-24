#!/usr/bin/env bash
# The served hall offers the release, and the browser gets exactly the release's model and runtime.
#   tools/acceptance/brainiac_check.sh PORT RELEASE_DIR
set -euo pipefail
PORT=$1 RELEASE=$2
URL="http://localhost:$PORT/gemp-swccg-server" REF="Referer: http://localhost:$PORT/gemp-swccg/"
JAR=$(mktemp)
curl -sf -o /dev/null -c "$JAR" -H "$REF" --data "login=asdf&password=asdf" "$URL/login"
attr=$(curl -sf -b "$JAR" -H "$REF" "$URL/hall" | grep -o 'brainiac="[^"]*"' | head -1)
echo "hall: $attr"
name=$("${PY:-.venv/bin/python}" -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["name"],m["sha256"],m["runtime"])' "$RELEASE/manifest.json")
set -- $name
[[ "$attr" == "brainiac=\"$1|$2|"* ]] || { echo "hall does not offer release $1 ($2)"; exit 1; }
got=$(curl -sf -b "$JAR" -H "$REF" "$URL/brainiac/model/$2.onnx" | sha256sum | cut -d' ' -f1)
[ "$got" = "$2" ] || { echo "served model sha256 $got != manifest $2"; exit 1; }
echo "model: served $2.onnx matches the manifest"
curl -sf -o /dev/null -b "$JAR" -H "$REF" "$URL/brainiac/runtime/$3/ort.webgpu.min.js" && echo "runtime: $3 served"
rm -f "$JAR"
