#!/usr/bin/env python3
"""Deck pool, held-out set and evaluation schedules for the corpus v3 RL pilot (weekend-modern-retrain Addendum 3).

Held out from training: rl-v1's 4 held-out decks, plus 10 % of the new PC decks (corpus v3 rows with deck_id "pcm-*"),
stratified by side x era, fixed seed. Writes:
  data/schedules/rl_pool_v3.txt        training pool ("DS|LS<TAB>path"), for TrainGen --pool
  data/schedules/v3_heldout.txt        held-out new decks (one deck_id per line)
  data/schedules/eval_v3_heldout.tsv   600 games: 300 random held-out DS x held-out LS pairings, each twice (both seats)
The corpus v2 regression schedule is the existing data/schedules/eval_vs_bc.tsv (v2 decks, 300 pairings x both seats).

    python3 scripts/make_v3_pool.py [--seed 3] [--fraction 0.10]
"""
import argparse
import collections
import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data/decks/corpus")
SCHED = os.path.join(ROOT, "data/schedules")
RL_V1_HELDOUT = {"pc-2021_MPC_Day_1-LingrellDS-Court", "swdb-59-Roger_Roger", "local-Freeman_LS_Family_Anakin",
                 "swdb-73-NABOO_-_LS_(Theme)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--pairings", type=int, default=300)
    ap.add_argument("--out", default="eval_v3_heldout.tsv", help="held-out eval schedule file name (in data/schedules)")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data/decks/manifest.csv"))) if r["status"] == "passed"]
    assert len(rows) == len(os.listdir(CORPUS)), "manifest and corpus disagree"
    rnd = random.Random(a.seed)
    strata = collections.defaultdict(list)
    for r in rows:
        if r["deck_id"].startswith("pcm-"):
            strata[(r["side"], r["era"])].append(r["deck_id"])
    heldout = set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        heldout.update(rnd.sample(ids, max(1, round(a.fraction * len(ids)))))
    side = {r["deck_id"]: r["side"] for r in rows}
    path = lambda d: os.path.join(CORPUS, d + ".xml")
    pool = [r for r in rows if r["deck_id"] not in heldout and r["deck_id"] not in RL_V1_HELDOUT]
    with open(os.path.join(SCHED, "rl_pool_v3.txt"), "w") as f:
        for r in sorted(pool, key=lambda r: r["deck_id"]):
            f.write(("DS" if r["side"] == "DARK" else "LS") + "\t" + path(r["deck_id"]) + "\n")
    with open(os.path.join(SCHED, "v3_heldout.txt"), "w") as f:
        f.writelines(d + "\n" for d in sorted(heldout))
    ds = sorted(d for d in heldout if side[d] == "DARK")
    ls = sorted(d for d in heldout if side[d] == "LIGHT")
    with open(os.path.join(SCHED, a.out), "w") as f:
        for _ in range(a.pairings):
            d, l = rnd.choice(ds), rnd.choice(ls)
            f.write(f"{path(d)}\t{path(l)}\n" * 2)
    by = collections.Counter((side[d], next(r["era"] for r in rows if r["deck_id"] == d)) for d in heldout)
    print(f"pool {len(pool)} decks (DS {sum(r['side'] == 'DARK' for r in pool)}, LS {sum(r['side'] == 'LIGHT' for r in pool)}); "
          f"held-out new {len(heldout)} {dict(by)}; {a.out} {2 * a.pairings} games")


if __name__ == "__main__":
    main()
