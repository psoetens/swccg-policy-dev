"""Plan v-cards §1.3: recent virtual tournament decks, split into gathering and held-out decks, with schedules.

Selection (data/decks/v4): status passed, era modern, event year >= 2025, at least one virtual card (set >= 200), every
card in the grown card table, no crash in the AdvancedAI screen. Each deck's objective (or "none") comes from its
objective card. Per (side, objective) about 20% of the decks are held out (seeded); an objective with one deck stays in
gathering. Writes reports/data/v-cards/decks.json and runs/v-cards/schedules/:
  gather-ds.tsv / gather-ls.tsv  every gathering deck 4 times on the policy's side (DS / LS rows for --agent-side ds / ls),
                                 opponents drawn from the other side's gathering decks;
  heldout.tsv                    held-out DS x LS pairings, each twice in a row (paired seats with --agent-side alternate).
    .venv/bin/python scripts/vc_select_decks.py --cards runs/v-cards/tables-shc095-cards/cards.npz \\
        --export runs/v-cards/cards-all.npz [--heldout-games 2400]
"""
import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("data/decks/v4/corpus")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards", type=Path, required=True, help="grown card table (cards.npz)")
    parser.add_argument("--export", type=Path, required=True, help="cards.py output (for card categories)")
    parser.add_argument("--heldout-share", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--heldout-games", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    known = {str(blueprint) for blueprint in np.load(args.cards)["ids"]}
    export = np.load(args.export)
    names = [str(name) for name in export["feature_names"]]
    objective_column = names.index("cardCategory=OBJECTIVE")
    objectives = {str(blueprint) for blueprint, row in zip(export["ids"], export["features"]) if row[objective_column]}
    titles = {str(blueprint): str(title) for blueprint, title in zip(export["ids"], export["titles"])}
    decks = []
    skipped = collections.Counter()
    for row in csv.DictReader(open("data/decks/v4/manifest.csv")):
        if row["status"] != "passed" or row["era"] != "modern":
            continue
        if not row["date"][:4].isdigit() or int(row["date"][:4]) < 2025:
            skipped["date"] += 1
            continue
        path = CORPUS / f"{row['deck_id']}.xml"
        ids = re.findall(r'blueprintId="([^"]+)"', path.read_text())
        if any(blueprint not in known for blueprint in ids):
            skipped["unknown card"] += 1
            continue
        virtual = [b for b in ids if b.split("_", 1)[0].isdigit() and int(b.split("_", 1)[0]) >= 200]
        if not virtual:
            skipped["no virtual card"] += 1
            continue
        if row["screen_crash"] not in ("0", ""):
            skipped["screen crash"] += 1
            continue
        objective = next((b for b in ids if b in objectives), None)
        decks.append({"deck": str(path), "side": row["side"], "event": row["event"], "date": row["date"],
                      "objective": objective, "objective_title": titles.get(objective, "none"),
                      "virtual_cards": len(virtual)})
    groups = collections.defaultdict(list)
    for deck in decks:
        groups[(deck["side"], deck["objective"])].append(deck)
    for key in sorted(groups, key=str):
        members = sorted(groups[key], key=lambda d: d["deck"])
        rng.shuffle(members)
        held = round(len(members) * args.heldout_share) if len(members) > 1 else 0
        for index, deck in enumerate(members):
            deck["split"] = "heldout" if index < held else "gathering"
    out = Path("runs/v-cards/schedules")
    out.mkdir(parents=True, exist_ok=True)
    by = {(split, side): [d["deck"] for d in decks if d["split"] == split and d["side"] == side]
          for split in ("gathering", "heldout") for side in ("DARK", "LIGHT")}
    gather = {}
    for side, other, name in (("DARK", "LIGHT", "gather-ds"), ("LIGHT", "DARK", "gather-ls")):
        lines = []
        for deck in by[("gathering", side)]:
            for _ in range(args.repeats):
                opponent = rng.choice(by[("gathering", other)])
                lines.append(f"{deck}\t{opponent}" if side == "DARK" else f"{opponent}\t{deck}")
        (out / f"{name}.tsv").write_text("\n".join(lines) + "\n")
        gather[name] = len(lines)
    dark, light = by[("heldout", "DARK")], by[("heldout", "LIGHT")]
    # every held-out deck at least once, then random pairings up to the game count
    pairings = [(ds, rng.choice(light)) for ds in dark] + [(rng.choice(dark), ls) for ls in light]
    pairings += [(rng.choice(dark), rng.choice(light)) for _ in range(max(0, args.heldout_games // 2 - len(pairings)))]
    rng.shuffle(pairings)
    lines = [f"{ds}\t{ls}" for ds, ls in pairings for _ in range(2)]
    (out / "heldout.tsv").write_text("\n".join(lines) + "\n")
    held_used = {deck for line in lines for deck in line.split("\t")}
    summary = {"decks": len(decks), "skipped": dict(skipped),
               "by_split_side": {f"{split}/{side}": len(v) for (split, side), v in by.items()},
               "objectives": len({(d["side"], d["objective"]) for d in decks}),
               "objectives_heldout": len({(d["side"], d["objective"]) for d in decks if d["split"] == "heldout"}),
               "events": dict(collections.Counter(d["event"] for d in decks).most_common()),
               "gather_rows": gather, "heldout_rows": len(lines),
               "heldout_decks_in_schedule": len(held_used), "heldout_decks": sum(len(by[("heldout", s)]) for s in ("DARK", "LIGHT")),
               "seed": args.seed}
    Path("reports/data/v-cards").mkdir(parents=True, exist_ok=True)
    Path("reports/data/v-cards/decks.json").write_text(json.dumps({"summary": summary, "decks": decks}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
