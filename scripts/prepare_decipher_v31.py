"""Build the approved Decipher-only-first pool, held-out schedule and immutable row subset."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

from swccg_policy.v4_tables import CardTable


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/knowledge-v3.1/decipher"


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    source = RUN / "cards.npz"
    cards = CardTable.load(source).cards
    ids = set(map(str, cards["ids"]))
    export = json.loads((ROOT / "reports/data/knowledge-v3.1/card-database-export.json").read_text())
    database = Path(export["output"])
    db = {row["cardId"]: row for row in json.loads(database.read_text())}
    corpus = ROOT / "data/decks/v4"
    decks = []
    for row in csv.DictReader((corpus / "manifest.csv").open()):
        if row["status"] != "passed":
            continue
        path = corpus / "corpus" / (row["deck_id"] + ".xml")
        blueprints = re.findall(r'<card blueprintId="([^"]+)"', path.read_text())
        if any(int(blueprint.split("_", 1)[0]) >= 200 for blueprint in blueprints):
            continue
        objectives = [db[bp]["title"] for bp in blueprints if db.get(bp, {}).get("cardCategory") == "OBJECTIVE"]
        starting = [db[bp]["title"] for bp in blueprints
                    if db.get(bp, {}).get("cardCategory") == "INTERRUPT"
                    and db[bp].get("cardSubtype") in {"STARTING", "USED_OR_STARTING", "LOST_OR_STARTING"}]
        family = "OBJ:" + objectives[0] if objectives else "SI:" + starting[0] if starting else "NONE"
        decks.append({"id": row["deck_id"], "path": str(path), "cards": set(blueprints),
                      "side": "DS" if row["side"] == "DARK" else "LS", "family": family, "era": row["era"]})
    families = defaultdict(list)
    for deck in decks:
        families[(deck["side"], deck["family"])].append(deck)
    rng = random.Random(3104)
    held_ids = {rng.choice(sorted(group, key=lambda deck: deck["id"]))["id"]
                for _, group in sorted(families.items()) if len(group) >= 4}
    pool = [deck for deck in decks if deck["id"] not in held_ids]
    heldout = [deck for deck in decks if deck["id"] in held_ids]
    for side in ("DS", "LS"):
        group = [deck for deck in pool if deck["side"] == side]
        frequency = Counter(bp for deck in group for bp in deck["cards"])
        family_counts = Counter(deck["family"] for deck in group)
        for deck in group:
            deck["rare_share"] = sum(frequency[bp] < 0.02 * len(group) for bp in deck["cards"]) / len(deck["cards"])
            deck["weight"] = (1 + 2 * deck["rare_share"]) / family_counts[deck["family"]]
        minimum = min(deck["weight"] for deck in group)
        for deck in group:
            deck["copies"] = max(1, min(20, round(deck["weight"] / minimum)))
    schedule_root = ROOT / "data/schedules"
    pool_path = schedule_root / "rl_pool_decipher.txt"
    weights_path = schedule_root / "rl_pool_decipher_weights.csv"
    heldout_path = schedule_root / "eval_decipher_heldout.tsv"
    expected_pool = "".join(f"{deck['side']}\t{deck['path']}\n" * deck["copies"] for deck in pool)
    if pool_path.exists() and pool_path.read_text() != expected_pool:
        raise ValueError("Refreshed database changed the already-collected pool")
    pool_path.write_text(expected_pool)
    with weights_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["deck_id", "side", "family", "era", "rare_share", "weight", "copies"])
        for deck in sorted(pool, key=lambda row: (row["side"], row["family"], row["id"])):
            writer.writerow([deck[key] for key in ("id", "side", "family", "era", "rare_share", "weight", "copies")])
    # Both seats use held-out decks, so no seed deck can occur in the coverage set.
    held_sides = {side: sorted(deck["path"] for deck in heldout if deck["side"] == side) for side in ("DS", "LS")}
    if not all(held_sides.values()):
        raise ValueError("Decipher-only held-out set requires both sides")
    pairs = [(held_sides["DS"][index % len(held_sides["DS"])],
              held_sides["LS"][index % len(held_sides["LS"])]) for index in range(64)]
    heldout_path.write_text("".join(dark + "\t" + light + "\n" for dark, light in pairs))
    report = {
        "eligible_decks": len(decks), "pool_decks": len(pool), "held_out_decks": len(heldout),
        "pool_sides": dict(Counter(deck["side"] for deck in pool)),
        "held_out_sides": dict(Counter(deck["side"] for deck in heldout)),
        "pool_families": len({(deck["side"], deck["family"]) for deck in pool}),
        "distinct_cards": len(set.union(*(deck["cards"] for deck in pool))),
        "decipher_blueprints": len(ids) - 4,
        "missing_card_rows": sorted(set.union(*(deck["cards"] for deck in decks)) - ids),
        "held_out_ids": sorted(held_ids),
        "selection": "All passed decipher decks; one held out per side/family with at least four decks",
        "weighting": "(1 + 2 * rare_share) / family_count; copies clipped to 1..20, same as make_v4_pool.py",
        "source_sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), source, corpus / "manifest.csv", database)},
        "deck_sha256": {deck["path"]: hashlib.sha256(Path(deck["path"]).read_bytes()).hexdigest() for deck in decks},
    }
    (ROOT / "reports/data/knowledge-v3.1/decipher-pool.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("eligible_decks", "pool_decks", "held_out_decks", "pool_sides",
                                                 "pool_families", "decipher_blueprints")}))


if __name__ == "__main__":
    main()
