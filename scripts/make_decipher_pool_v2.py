"""Plan actor-v6-inputs: the Decipher RL pool v2 = today's pool decks + the staged objective-coverage decks.

Staged decks join when they are Decipher-only (manifest `decipher_only`), use no virtual card (set >= 200) and every
card has a row in the card table, and pass the Dojo coverage gate. The weighting is prepare_decipher_v31.py's rule over
the combined decks per side: family = first objective, else starting interrupt, else NONE; weight (1 + 2 rare_share) /
family count; copies round(weight / min) clipped to 1..20. The held-out decks of the existing pool stay out, and
eval_decipher_heldout.tsv is not touched.

    PYTHONPATH=scripts:src .venv/bin/python scripts/make_decipher_pool_v2.py [--version v3]

v3 = the same rule after the Dojo registry gained the dojo-three-objectives items (more staged decks pass the gate);
v2 stays as it was built (the actor-v6 run used it).
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from swccg_policy.table_bundle import TableBundle

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "data/decks/objective-coverage"
SCHEDULES = ROOT / "data/schedules"
TABLES = ROOT / "runs/knowledge-v3.1/tables-v5"


def blueprints(path):
    return re.findall(r'<card blueprintId="([^"]+)"', path.read_text())


def family(cards, db):
    objectives = [db[bp]["title"] for bp in cards if db.get(bp, {}).get("cardCategory") == "OBJECTIVE"]
    starting = [db[bp]["title"] for bp in cards
                if db.get(bp, {}).get("cardCategory") == "INTERRUPT"
                and db[bp].get("cardSubtype") in {"STARTING", "USED_OR_STARTING", "LOST_OR_STARTING"}]
    return "OBJ:" + objectives[0] if objectives else "SI:" + starting[0] if starting else "NONE"


def refused_by_gate(paths):
    """Deck files the Dojo coverage gate refuses (scripts/dojo_coverage.py, run on a schedule of these decks)."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as schedule:
        schedule.write("".join(f"DS\t{path}\n" for path in paths))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/dojo_coverage.py"), schedule.name],
                            capture_output=True, text=True, cwd=ROOT)
    Path(schedule.name).unlink()
    return set(re.findall(r"[A-Za-z0-9_.()'-]+\.xml", result.stdout + result.stderr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", help="output name: rl_pool_decipher_<version>.txt")
    version = parser.parse_args().version
    export = json.loads((ROOT / "reports/data/knowledge-v3.1/card-database-export.json").read_text())
    db = {row["cardId"]: row for row in json.loads(Path(export["output"]).read_text())}
    known = TableBundle.load(TABLES).cards.index
    old = json.loads((ROOT / "reports/data/knowledge-v3.1/decipher-pool.json").read_text())
    held = set(old["held_out_ids"])
    decks = []
    for row in csv.DictReader((SCHEDULES / "rl_pool_decipher_weights.csv").open()):
        path = Path("data/decks/v4/corpus") / (row["deck_id"] + ".xml")
        decks.append({"id": row["deck_id"], "path": str(path), "cards": set(blueprints(ROOT / path)),
                      "side": row["side"], "family": row["family"], "era": row["era"], "source": "pool"})
    candidates, skipped = [], Counter()
    for row in csv.DictReader((STAGED / "manifest.csv").open()):
        path = Path("data/decks/objective-coverage/staged") / (row["deck_id"] + ".xml")
        cards = blueprints(ROOT / path)
        stripped = [re.sub(r"[*^]+$", "", bp) for bp in cards]
        if row["decipher_only"] != "True":
            skipped["not Decipher-only"] += 1
        elif any(int(bp.split("_", 1)[0]) >= 200 for bp in stripped):
            skipped["virtual card"] += 1
        elif any(bp not in known for bp in stripped):
            skipped["card not in the card table"] += 1
        elif row["deck_id"] in held:
            skipped["held out"] += 1
        else:
            candidates.append({"id": row["deck_id"], "path": str(path), "cards": set(cards),
                               "side": "DS" if row["side"] == "DARK" else "LS", "family": family(cards, db),
                               "era": "staged", "source": "staged"})
    refused = refused_by_gate([deck["path"] for deck in candidates])
    skipped["Dojo coverage gate"] = sum(Path(deck["path"]).name in refused for deck in candidates)
    decks += [deck for deck in candidates if Path(deck["path"]).name not in refused]
    for side in ("DS", "LS"):
        group = [deck for deck in decks if deck["side"] == side]
        frequency = Counter(bp for deck in group for bp in deck["cards"])
        family_counts = Counter(deck["family"] for deck in group)
        for deck in group:
            deck["rare_share"] = sum(frequency[bp] < 0.02 * len(group) for bp in deck["cards"]) / len(deck["cards"])
            deck["weight"] = (1 + 2 * deck["rare_share"]) / family_counts[deck["family"]]
        minimum = min(deck["weight"] for deck in group)
        for deck in group:
            deck["copies"] = max(1, min(20, round(deck["weight"] / minimum)))
    pool_path = SCHEDULES / f"rl_pool_decipher_{version}.txt"
    weights_path = SCHEDULES / f"rl_pool_decipher_{version}_weights.csv"
    for path in (pool_path, weights_path):
        if path.exists():
            raise FileExistsError(path)
    pool_path.write_text("".join(f"{deck['side']}\t{deck['path']}\n" * deck["copies"] for deck in decks))
    with weights_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["deck_id", "side", "family", "era", "source", "rare_share", "weight", "copies"])
        for deck in sorted(decks, key=lambda row: (row["side"], row["family"], row["id"])):
            writer.writerow([deck[key] for key in ("id", "side", "family", "era", "source", "rare_share", "weight",
                                                   "copies")])
    report = {"pool_decks": len(decks), "from_pool": sum(deck["source"] == "pool" for deck in decks),
              "from_staged": sum(deck["source"] == "staged" for deck in decks),
              "staged_skipped": dict(skipped), "sides": dict(Counter(deck["side"] for deck in decks)),
              "families": len({(deck["side"], deck["family"]) for deck in decks}),
              "new_families": sorted({deck["family"] for deck in decks if deck["source"] == "staged"}
                                     - {deck["family"] for deck in decks if deck["source"] == "pool"}),
              "pool_lines": sum(deck["copies"] for deck in decks),
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), STAGED / "manifest.csv",
                                             SCHEDULES / "rl_pool_decipher_weights.csv")}}
    out = ROOT / f"reports/data/actor-v6-inputs/pool-{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=1))


if __name__ == "__main__":
    main()
