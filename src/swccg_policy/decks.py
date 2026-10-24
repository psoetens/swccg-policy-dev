"""Deck pool, held-out decks and balanced game schedules for TrainGen.

The pool is every deck the deck-corpus screen passed (data/decks/manifest.csv, status "passed"), plus decks listed in
--reinstate (decks rejected only because of a harness bug that has since been fixed; their XML stays in data/decks/rejected/).
Held-out decks: per side, the decks with the most blueprints that appear in no other pool deck, so evaluation on them
tests cards the policy never saw in training.

Schedules are TSV files, one game per line ("ds_path<TAB>ls_path"). Within each side decks are used round-robin in a
reshuffled order, so every deck of a side gets the same number of games (±1). The two sides have different deck counts, so
each LS deck necessarily plays more games than each DS deck; per-deck counts are written next to each schedule.

    uv run python -m swccg_policy.decks --train-games 5000 --out data/schedules
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECKS = ROOT / "data/decks"
# Decks the deck-corpus screen could not accept only because of the GempBench null-parameter NPE (since fixed) and that
# passed the re-screen (swccg_policy.rescreen -> data/schedules/rescreen.json). Their XML stays in data/decks/rejected/.
def rescreened() -> list[str]:
    path = ROOT / "data/schedules/rescreen.json"
    if not path.exists():
        return []
    return sorted(k for k, v in json.load(open(path)).items() if v["pass"])


REINSTATED = rescreened()


def strip_bp(bp: str) -> str:
    return bp.rstrip("*^")


def deck_blueprints(path: Path) -> list[str]:
    return [strip_bp(b) for b in re.findall(r'<card blueprintId="([^"]+)"', path.read_text())]


def deck_era(row: dict) -> str:
    """Era of a deck: the manifest's "era" column when present (corpus v2), otherwise derived from the source/event."""
    if row.get("era"):
        return row["era"]
    ev = f"{row.get('event', '')} {row.get('deck_id', '')}"
    if "2020" in ev or "2021" in ev:
        return "2020-21"
    if row.get("deck_id", "").startswith("local-"):
        return "2022"
    return "unknown"


def load_pool(reinstate: list[str]) -> dict[str, dict]:
    pool = {}
    for r in csv.DictReader(open(DECKS / "manifest.csv")):
        if r["status"] == "passed":
            path = DECKS / "corpus" / f"{r['deck_id']}.xml"
        elif r["deck_id"] in reinstate and r["status"] in ("provisional_exception", "rejected"):
            path = DECKS / "rejected" / f"{r['deck_id']}.xml"
        else:
            continue
        pool[r["deck_id"]] = {"side": "DS" if r["side"] == "DARK" else "LS", "path": str(path), "reinstated": r["status"] != "passed",
                              "cards": deck_blueprints(path), "era": deck_era(r)}
    return pool


def pick_heldout(pool: dict[str, dict], per_side: int) -> list[str]:
    """Per side, the decks with the most blueprints found in no other pool deck. With an era mix, one deck per era per
    side (most exclusive cards within that era), so the held-out test covers both eras."""
    freq = collections.Counter(b for d in pool.values() for b in set(d["cards"]))
    eras = sorted({d["era"] for d in pool.values()})
    held = []
    for side in ("DS", "LS"):
        groups = [[k for k, d in pool.items() if d["side"] == side and d["era"] == e] for e in eras] if per_side == len(eras) \
            else [[k for k, d in pool.items() if d["side"] == side]]
        for keys in groups:
            scored = sorted(((sum(1 for b in set(pool[k]["cards"]) if freq[b] == 1), k) for k in keys), key=lambda x: (-x[0], x[1]))
            held += [k for _, k in scored[:per_side // len(groups)]]
    return held


def balanced(ds: list[str], ls: list[str], games: int, rng: random.Random) -> list[tuple[str, str]]:
    def stream(xs):
        while True:
            order = xs[:]
            rng.shuffle(order)
            yield from order
    a, b = stream(ds), stream(ls)
    return [(next(a), next(b)) for _ in range(games)]


def write_schedule(pool, pairs, path: Path) -> dict:
    """Schedule TSV plus <name>.games.json (per game: deck ids and eras) and <name>.counts.json (games per deck)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d, l in pairs:
            f.write(f"{pool[d]['path']}\t{pool[l]['path']}\n")
    json.dump([{"game": i, "ds": d, "ls": l, "ds_era": pool[d]["era"], "ls_era": pool[l]["era"]} for i, (d, l) in enumerate(pairs)],
              open(path.with_suffix(".games.json"), "w"))
    counts = collections.Counter([d for d, _ in pairs] + [l for _, l in pairs])
    json.dump(dict(sorted(counts.items())), open(path.with_suffix(".counts.json"), "w"), indent=1)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-games", type=int, default=5000)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--heldout-games-per-deck", type=int, default=100)
    ap.add_argument("--heldout-per-side", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default=str(ROOT / "data/schedules"))
    a = ap.parse_args()
    pool = load_pool(REINSTATED)
    held = pick_heldout(pool, a.heldout_per_side)
    train_ds = sorted(k for k, d in pool.items() if d["side"] == "DS" and k not in held)
    train_ls = sorted(k for k, d in pool.items() if d["side"] == "LS" and k not in held)
    out = Path(a.out)
    rng = random.Random(a.seed)
    c_train = write_schedule(pool, balanced(train_ds, train_ls, a.train_games, rng), out / "train.tsv")
    c_eval = write_schedule(pool, balanced(train_ds, train_ls, a.eval_games, random.Random(a.seed + 1)), out / "eval_seen.tsv")
    pairs = []
    rh = random.Random(a.seed + 2)
    for h in held:
        for _ in range(a.heldout_games_per_deck):
            pairs.append((h, rh.choice(train_ls)) if pool[h]["side"] == "DS" else (rh.choice(train_ds), h))
    write_schedule(pool, pairs, out / "eval_heldout.tsv")
    # Play-strength schedules: random balanced pairings of training decks, and the 3 benchmark matchups (local decks).
    write_schedule(pool, balanced(train_ds, train_ls, 300, random.Random(a.seed + 3)), out / "play_random.tsv")
    # The 3 benchmark matchups use the original local deck files (comparable with the gemp-swccg-agent benchmark reports).
    local = Path.home() / "src/gemp-swccg-agent/decks"
    bench = [("Krueger DS BHBM.txt", "HatoumLS Yoda Communing.txt"), ("HatoumDS SC.txt", "Krueger LS OA.txt"),
             ("NelsonDS Ralltiir Operations (V).txt", "Nelson LS Qui-Gon Communing.txt")]
    with open(out / "play_bench.tsv", "w") as f:
        for i in range(300):
            d, l = bench[i % 3]
            f.write(f"{local / d}\t{local / l}\n")
    train_cards = {b for k in train_ds + train_ls for b in pool[k]["cards"]}
    info = {
        "pool": len(pool), "pool_ds": sum(d["side"] == "DS" for d in pool.values()), "pool_ls": sum(d["side"] == "LS" for d in pool.values()),
        "reinstated": [k for k, d in pool.items() if d["reinstated"]],
        "heldout": {h: {"side": pool[h]["side"], "cards": len(set(pool[h]["cards"])),
                        "cards_absent_from_training": len(set(pool[h]["cards"]) - train_cards)} for h in held},
        "train_decks": {"DS": len(train_ds), "LS": len(train_ls)},
        "eras": dict(collections.Counter(d["era"] for d in pool.values())),
        "unique_blueprints_training_decks": len(train_cards),
        "per_deck_games_train": {"DS": sorted(set(c_train[k] for k in train_ds)), "LS": sorted(set(c_train[k] for k in train_ls))},
    }
    json.dump(info, open(out / "pool.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
