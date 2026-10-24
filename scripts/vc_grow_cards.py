"""Plan v-cards §1.2: grow a checkpoint's own card table with every virtual card of the rebased engine's export.

The checkpoint's rows (Decipher and specials) stay as they are. The added rows are the export's cards of sets 200–227
and 301 (501 playtest and 601 Legacy excluded), with features mapped by column name onto the checkpoint's schema: a value
without a column is dropped and listed (its game text still carries it). Writes a child table bundle (same vocabulary)
and a report comparing every added card with the September preparation (decipher/virtual-additions.npz) and every
Decipher row with the fresh export (kept frozen either way).
    PYTHONPATH=scripts:src .venv/bin/python scripts/vc_grow_cards.py CHECKPOINT runs/v-cards/cards-all.npz OUT_BUNDLE REPORT
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import unpack_tables
from swccg_policy.v4_tables import CardTable, ROW_FIELDS

PREPARED = Path("runs/knowledge-v3.1/decipher/virtual-additions.npz")
EXCLUDED_SETS = (501, 601)


def card_set(blueprint):
    head = str(blueprint).split("_", 1)[0]
    return int(head) if head.lstrip("-").isdigit() else None


def virtual(blueprint):
    number = card_set(blueprint)
    return number is not None and (200 <= number <= 227 or number == 301)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("export", type=Path, help="cards.py output for the fresh export")
    parser.add_argument("out", type=Path, help="new table bundle directory")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    owned = unpack_tables(payload["tables"])
    base = owned.cards.cards
    fresh = dict(np.load(args.export))
    names = [str(name) for name in base["feature_names"]]
    fresh_names = [str(name) for name in fresh["feature_names"]]
    column = {name: index for index, name in enumerate(names)}
    fresh_ids = [str(blueprint) for blueprint in fresh["ids"]]
    known = set(owned.cards.index)
    selected = [row for row, blueprint in enumerate(fresh_ids) if virtual(blueprint) and blueprint not in known]

    features = np.zeros((len(selected), len(names)), dtype=base["features"].dtype)
    dropped = {}
    for out_row, row in enumerate(selected):
        for fresh_column, value in enumerate(fresh["features"][row]):
            if value == 0:
                continue
            name = fresh_names[fresh_column]
            if name in column:
                features[out_row, column[name]] = value
            else:
                dropped.setdefault(fresh_ids[row], {})[name] = float(value)
    additions = {key: value for key, value in base.items() if key not in ROW_FIELDS}
    additions.update(ids=fresh["ids"][selected], titles=fresh["titles"][selected], sides=fresh["sides"][selected],
                     features=features, text_emb=fresh["text_emb"][selected].astype(base["text_emb"].dtype),
                     text_hashes=fresh["text_hashes"][selected])
    grown_cards = owned.cards.grow(additions)
    bundle = TableBundle(grown_cards, owned.vocabulary, owned.sha, (*owned.ancestors, owned.sha))
    owned.assert_prefix_of(bundle)
    bundle.save(args.out)

    # The September preparation: same blueprint, same schema (861 columns) -> did upstream change the card since?
    prepared = dict(np.load(PREPARED))
    prepared_index = {str(blueprint): row for row, blueprint in enumerate(prepared["ids"])}
    new_index = {str(blueprint): row for row, blueprint in enumerate(additions["ids"])}
    vs_prepared = {}
    for blueprint, row in prepared_index.items():
        if not virtual(blueprint):
            continue
        if blueprint not in new_index:
            vs_prepared[blueprint] = "not in the fresh export"
            continue
        new_row = new_index[blueprint]
        diff = {names[c]: [float(prepared["features"][row][c]), float(features[new_row][c])]
                for c in np.nonzero(prepared["features"][row] != features[new_row])[0]}
        text = str(prepared["text_hashes"][row]) != str(additions["text_hashes"][new_row])
        if diff or text:
            vs_prepared[blueprint] = {"features": diff, "text_changed": text}
    new_since_prepared = sorted(set(new_index) - set(prepared_index))
    # Decipher rows: kept frozen; list what the fresh export would change.
    fresh_index = {blueprint: row for row, blueprint in enumerate(fresh_ids)}
    frozen = {}
    for blueprint, row in owned.cards.index.items():
        if blueprint not in fresh_index or card_set(blueprint) is None or card_set(blueprint) < 0:
            continue
        fresh_row = fresh_index[blueprint]
        mapped = np.zeros(len(names), dtype=base["features"].dtype)
        for fresh_column in np.nonzero(fresh["features"][fresh_row])[0]:
            if fresh_names[fresh_column] in column:
                mapped[column[fresh_names[fresh_column]]] = fresh["features"][fresh_row][fresh_column]
        diff = {names[c]: [float(base["features"][row][c]), float(mapped[c])]
                for c in np.nonzero(base["features"][row] != mapped)[0]}
        text = str(base["text_hashes"][row]) != str(fresh["text_hashes"][fresh_row])
        if diff or text:
            frozen[blueprint] = {"title": str(base["titles"][row]), "features": diff, "text_changed": text}
    by_set = {}
    for blueprint in additions["ids"]:
        by_set[str(card_set(blueprint))] = by_set.get(str(card_set(blueprint)), 0) + 1
    excluded = sorted(blueprint for blueprint in fresh_ids if card_set(blueprint) in EXCLUDED_SETS)
    report = {"checkpoint": str(args.checkpoint), "parent_table_sha": owned.sha, "table_sha": bundle.sha,
              "card_sha": grown_cards.sha, "rows_before": len(owned.cards.ids), "rows_after": len(grown_cards.ids),
              "added": len(selected), "added_by_set": dict(sorted(by_set.items(), key=lambda t: int(t[0]))),
              "excluded_sets": list(EXCLUDED_SETS), "excluded_cards": len(excluded),
              "fresh_columns_without_schema_column": sorted(set(fresh_names) - set(names)),
              "dropped_values": dropped, "new_since_september": new_since_prepared,
              "changed_since_september": vs_prepared, "decipher_rows_frozen_with_upstream_changes": frozen,
              "export": str(args.export), "export_sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("rows_before", "rows_after", "added", "added_by_set",
                                                  "fresh_columns_without_schema_column")}))
    print(json.dumps({"dropped_cards": len(dropped), "new_since_september": len(new_since_prepared),
                      "changed_since_september": len(vs_prepared), "decipher_frozen_changed": len(frozen)}))


if __name__ == "__main__":
    main()
