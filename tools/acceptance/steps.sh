#!/usr/bin/env bash
# Inside the acceptance container: plan dev-distribution §1, items 1-5 (item 5 up to a served hall; the browser game
# itself is a manual check). Each step logs to /work/logs/<n>-<step>.log; a summary goes to /work/logs/summary.txt.
set -uo pipefail
mkdir -p /work/logs
S=/work/logs/summary.txt; : > "$S"
step() {  # name, command...
    local name=$1; shift
    local start=$(date +%s)
    echo "== $name: $*" | tee -a "$S"
    if "$@" > "/work/logs/$name.log" 2>&1; then r=ok; else r="FAILED ($?)"; fi
    echo "   $r in $(( $(date +%s) - start )) s" | tee -a "$S"
    [ "$r" = ok ]
}
cd /work
[ -d swccg-policy ] || step 1-clone git clone file:///src/dev swccg-policy || exit 1
cd swccg-policy
step 2-setup scripts/setup.sh || exit 1
{ echo "   clone: .git $(du -sh .git | cut -f1) (pack $(git count-objects -vH | sed -n 's/size-pack: //p'), lfs $(du -sh .git/lfs | cut -f1))"; } | tee -a "$S"
PY=.venv/bin/python
export PYTHONPATH=src:scripts
step 3-pytest $PY -m pytest -q tests
step 4-evaluate $PY scripts/evaluate_v31.py --checkpoint models/shc-r1/policy.pt --reference models/D/policy.pt \
    --tag accept-shc-r1 --opponents rando,advanced --games 32 --threads 8
step 5-train env NAME=dev-smoke START=models/shc-r1/policy.pt GAMES=1000 THREADS=12 scripts/train_arm.sh
FINAL=$(ls runs/dojo/dev-smoke*/final.pt 2>/dev/null | sort | tail -1)
echo "   final: ${FINAL:-none}; $(tail -3 runs/dojo/dev-smoke.log 2>/dev/null | tr '\n' ' ')" | tee -a "$S"
[ -n "$FINAL" ] && step 6-evaluate-trained $PY scripts/evaluate_v31.py --checkpoint "$FINAL" \
    --reference models/D/policy.pt --tag accept-dev-smoke --opponents rando --games 16 --threads 8
step 7-brainiac-sync scripts/brainiac_server.sh sync
step 8-brainiac-start env RELEASE=models/brainiac/releases/shc-r1-2026-10-24 scripts/brainiac_server.sh start
step 9-brainiac-page bash -c 'curl -sf http://localhost:17111/gemp-swccg/ | head -c 300; grep -i brainiac runs/brainiac-e2e/server.log | head -5'
scripts/brainiac_server.sh stop >> /work/logs/8-brainiac-start.log 2>&1
echo "done" | tee -a "$S"
