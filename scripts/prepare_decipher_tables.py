"""Select fresh exported Decipher-only rows, preserving old relative order, and audit changed features."""

import hashlib
import json
from pathlib import Path

import numpy as np

from swccg_policy.v4_tables import CardTable, ROW_FIELDS


ROOT = Path("runs/knowledge-v3.1")
REPORTS = Path("reports/data/knowledge-v3.1")


def differences(before, after, blueprint):
    before_row = before["ids"].tolist().index(blueprint)
    after_row = after["ids"].tolist().index(blueprint)
    before_features = dict(zip(before["feature_names"], before["features"][before_row], strict=True))
    after_features = dict(zip(after["feature_names"], after["features"][after_row], strict=True))
    changed = {str(name): {"before": float(before_features.get(name, 0)), "after": float(after_features.get(name, 0))}
               for name in sorted(before_features.keys() | after_features.keys())
               if before_features.get(name, 0) != after_features.get(name, 0)}
    return {"features": changed,
            "text_changed": str(before["text_hashes"][before_row]) != str(after["text_hashes"][after_row]),
            "embedding_max_abs_difference": float(np.max(np.abs(
                before["text_emb"][before_row] - after["text_emb"][after_row])))}


def main():
    audit = json.loads((REPORTS / "card-database-export.json").read_text())
    if not audit["passed"]:
        raise ValueError("Passing database-export audit required")
    source = Path(audit["output"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit["output_sha256"]:
        raise ValueError("Audited export changed")
    fresh = dict(np.load(ROOT / "card-database-export/cards-all.npz"))
    old = dict(np.load("data/cards/card_table_knowledge_v2.npz"))
    ids = fresh["ids"].tolist()
    decipher = {blueprint for blueprint in ids[4:] if int(blueprint.split("_", 1)[0]) < 200}
    ordered = [blueprint for blueprint in old["ids"].tolist()[4:] if blueprint in decipher]
    ordered += [blueprint for blueprint in ids[4:] if blueprint in decipher and blueprint not in ordered]
    selected_ids = ids[:4] + ordered
    index = {blueprint: row for row, blueprint in enumerate(ids)}
    rows = [index[blueprint] for blueprint in selected_ids]
    metadata = {"database_sha256": np.array(audit["output_sha256"]), "gemp_commit": np.array(audit["gemp_commit"])}
    fresh.update(metadata)
    cards = {key: value[rows] if key in ROW_FIELDS else value for key, value in fresh.items()}
    output = ROOT / "decipher"
    output.mkdir(exist_ok=False)
    table = CardTable(cards)
    table.save(output / "cards.npz")
    # The extension source shares exactly the feature schema and original exported rows.
    virtual_rows = [row for row, blueprint in enumerate(ids) if row < 4 or
                    200 <= int(blueprint.split("_", 1)[0]) <= 226 or int(blueprint.split("_", 1)[0]) == 301]
    extension = {key: value[virtual_rows] if key in ROW_FIELDS else value for key, value in fresh.items()}
    np.savez_compressed(output / "virtual-additions.npz", **extension)
    grown = table.grow(extension)
    grown.save(output / "cards-virtual-extension.npz")
    shared = set(old["ids"].tolist()) & decipher
    changed = {blueprint: differences(old, cards, blueprint) for blueprint in sorted(shared)}
    changed_features = {blueprint: value["features"] for blueprint, value in changed.items() if value["features"]}
    manual = dict(np.load(ROOT / "legacy/cards-v1.npz"))
    report = {"decipher_cards": len(decipher), "special_rows": 4, "virtual_cards": len(virtual_rows) - 4,
              "feature_columns": cards["features"].shape[1], "old_feature_columns": old["features"].shape[1],
              "new_decipher_ids": sorted(decipher - set(old["ids"].tolist())),
              "changed_decipher_features": changed_features,
              "changed_decipher_text": {blueprint: value for blueprint, value in changed.items()
                                        if value["text_changed"]},
              "hunting_party_manual_vs_export": differences(manual, cards, "7_252"),
              "card_sha": table.sha, "virtual_extension_sha": grown.sha,
              "database_sha256": audit["output_sha256"], "gemp_commit": audit["gemp_commit"],
              "excluded_sets": [227, 501, 601],
              "count_note": "The fresh export removes stale virtual 225_4, so the actual extension has 1,156 cards.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (REPORTS / "decipher-card-rebuild.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "card-ids.txt").write_text("\n".join(ordered) + "\n")
    (output / "extended-card-ids.txt").write_text("\n".join(grown.ids[4:]) + "\n")
    print(json.dumps({key: report[key] for key in ("decipher_cards", "virtual_cards", "feature_columns",
                                                  "new_decipher_ids", "hunting_party_manual_vs_export")}))
    print(json.dumps({"changed_decipher_features": list(changed_features)}))


if __name__ == "__main__":
    main()
