"""Exporter validation (plan building block #3).

(a) Round trip: for every exported decision, rebuild the answer string from the chosen options (option answer field) and
    compare it with the answer the bot actually gave (TrainGen --traceDump). Multi-card answers are compared as sets.
(b) Hidden information: every token must be something the deciding player could know, by Gemp's own client rule —
    opponent cards only from public zones, own cards from public or owner-visible zones, and the own-unseen multiset only
    for own cards. No opponent card may appear from HAND, RESERVE_DECK, FORCE_PILE or USED_PILE.

    uv run python -m swccg_policy.audit --shards data/shards/smoke --trace data/shards/smoke/trace.tsv
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import sys
from pathlib import Path

import orjson


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--trace", help="TrainGen --traceDump file for the round-trip check")
    a = ap.parse_args()
    zones = json.load(open(Path(a.shards) / "zones.json"))
    public = {i for i, z in enumerate(zones) if z["public"]}
    owner_visible = {i for i, z in enumerate(zones) if z["visibleByOwner"]}
    name = {i: z["name"] for i, z in enumerate(zones)}
    forbidden_opp = {i for i, z in enumerate(zones) if z["name"] in ("HAND", "RESERVE_DECK", "FORCE_PILE", "USED_PILE", "SABACC_HAND")}
    unseen = next(i for i, z in enumerate(zones) if z["name"] == "OWN_UNSEEN")
    trace = {}
    if a.trace:
        for line in open(a.trace):
            gid, dec, player, dtype, answer = (line.rstrip("\n").split("\t") + [""])[:5]
            trace[(gid, int(dec))] = answer
    viol = collections.Counter()
    notes = collections.Counter()
    zone_use = collections.Counter()
    n = rt_ok = rt_bad = 0
    bad_examples = []
    for shard in sorted(glob.glob(str(Path(a.shards) / "shard-*.jsonl.gz"))):
        with gzip.open(shard, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                n += 1
                for t in r[8]:
                    kind, zone, mine = t[0], t[2], t[3]
                    zone_use[(name[zone], "mine" if mine else "opp")] += 1
                    if zones[zone].get("faceDown") and t[1] not in ("-1_1", "-1_2"):
                        if t[1].endswith("_BACK"):
                            # A face-down double-sided card shows its other face; Gemp sends the same to the client.
                            notes["face-down double-sided card shows its back face (as in the Gemp client)"] += 1
                        else:
                            viol[f"face-down card in {name[zone]} shows its blueprint"] += 1
                    if kind == 2:
                        if not (mine == 1 and zone == unseen):
                            viol["unseen token not own"] += 1
                    elif mine:
                        if zone not in public and zone not in owner_visible:
                            viol[f"own card in hidden zone {name[zone]}"] += 1
                    else:
                        if zone not in public:
                            viol[f"opponent card in non-public zone {name[zone]}"] += 1
                        if zone in forbidden_opp:
                            viol[f"opponent card from {name[zone]}"] += 1
                key = (r[1], r[2])
                if a.trace and key in trace:
                    opts, chosen = r[9], r[10]
                    rebuilt = {opts[i][4] for i in chosen}
                    actual = set(x for x in trace[key].split(",")) if trace[key] != "" else {""}
                    if r[4] == "INTEGER":
                        ok = bool(chosen) and (opts[chosen[0]][4] == trace[key] or chosen[0] == len(opts) - 1)
                    else:
                        ok = rebuilt == actual
                    if ok:
                        rt_ok += 1
                    else:
                        rt_bad += 1
                        if len(bad_examples) < 5:
                            bad_examples.append((key, r[4], sorted(rebuilt), sorted(actual)))
    print(f"records: {n}")
    print("token zones:", dict(sorted(zone_use.items())))
    print("hidden-info violations:", dict(viol) if viol else "none")
    if notes:
        print("notes:", dict(notes))
    if a.trace:
        print(f"round trip: {rt_ok} ok, {rt_bad} mismatched, {n - rt_ok - rt_bad} not in trace")
        for b in bad_examples:
            print("  mismatch:", b)
    return 1 if viol or rt_bad else 0


if __name__ == "__main__":
    sys.exit(main())
