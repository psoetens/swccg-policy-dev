# Shared helpers for objective_chain.sh / objective_chain2.sh (sourced; needs $R, note()).
# The disk can be shared with other users, which can temporarily take ~20 GB (2026-10-03 14:49):
# wait for space instead of failing, and set aside + retry runs the supervisor's resource guard refused.

free_gb() { df --output=avail -BG . | tail -1 | tr -dc 0-9; }

wait_disk() {  # label [min GB, default 24]: block until enough is free (checked every 5 min, at most 12 h)
    local waited=0 need=${2:-24}
    while [ "$(free_gb)" -lt "$need" ]; do
        [ "$waited" -eq 0 ] && note "waiting for >= ${need}G free disk before $1 (now $(free_gb)G)"
        waited=$((waited + 1))
        if [ "$waited" -gt 144 ]; then note "disk still < 24G after 12 h before $1; chain stops"; exit 1; fi
        sleep 300
    done
    [ "$waited" -gt 0 ] && note "disk ok ($(free_gb)G) after $((waited * 5)) min; continuing with $1"
    return 0
}

run_reason() {  # run dir -> result reason ('' if none)
    .venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['reason'])" "$1/result.json" 2>/dev/null
}

run_minutes() {  # run dir -> elapsed minutes as an integer (0 if unknown)
    .venv/bin/python -c "import json,sys;print(int(json.load(open(sys.argv[1]))['elapsed_seconds']//60))" "$1/result.json" 2>/dev/null || echo 0
}

supervised() {  # tag, then v31_stage.py args (without --tag). Retries an early resource-guard refusal up to 3 times.
    # One stage at a time on this machine (2026-10-04: two concurrent stages exhausted the shared disk).
    local tag=$1 attempt rc reason
    shift
    exec 9>"$R/.stage.lock"
    flock 9
    for attempt in 1 2 3; do
        reason=$(run_reason "$R/$tag")
        if [ -n "$reason" ] && [ "$reason" != resource_guard ]; then flock -u 9; return 0; fi
        if [ -d "$R/$tag" ]; then
            if [ "$(run_minutes "$R/$tag")" -gt 5 ]; then
                note "$tag: stopped by the resource guard after $(run_minutes "$R/$tag") min of training; not retried from scratch, chain stops for a decision"
                exit 1
            fi
            mv "$R/$tag" "$R/$tag-guard$(date +%s)"
            note "$tag: previous attempt refused by the resource guard; set aside"
        fi
        wait_disk "$tag" "$([ "$attempt" -gt 1 ] && echo 30 || echo 24)"
        .venv/bin/python scripts/v31_stage.py "$@" --tag "$tag" > "$R/$tag.supervisor.log" 2>&1
        rc=$?
        reason=$(run_reason "$R/$tag")
        note "$tag exit $rc ($reason)"
        [ "$reason" != resource_guard ] && { flock -u 9; return 0; }
    done
    note "$tag: refused by the resource guard 3 times; chain stops"
    exit 1
}

completed() {  # run dir -> success only if the run trained for its full duration
    [ "$(run_reason "$1")" = duration_complete ]
}
