#!/usr/bin/env bash
# Plan client-logs: load /clientlog.js first in a GempArena WebGL build's index.html (the build's copy; GempArena's
# source is not changed). Idempotent.  deploy/beta/inject_clientlog.sh BUILD_DIR
set -euo pipefail
INDEX=${1:?GempArena build folder}/index.html
grep -q 'src="/clientlog.js"' "$INDEX" && exit 0
grep -qi '<head>' "$INDEX" || { echo "$INDEX: no <head>" >&2; exit 1; }
sed -i '0,/<head>/I s##&\n    <script src="/clientlog.js" data-client="gemparena"></script>#' "$INDEX"
grep -q 'src="/clientlog.js"' "$INDEX"
