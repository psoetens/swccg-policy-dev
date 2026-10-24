"""Append the user-approved Hunting Party row using the existing cards.py feature/text pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from swccg_policy.cards import build_features, card_text, embed_texts
from swccg_policy.v4_tables import CardTable


ROOT = Path("runs/knowledge-v3.1/legacy")


def main():
    raise RuntimeError("Superseded by Amendment 4: export_card_database.py and prepare_decipher_tables.py")


def archived_main():
    cards = dict(np.load(ROOT / "cards.npz"))
    row = json.loads((ROOT / "missing-card.jsonl").read_text())
    features, names = build_features([row])
    index = {str(name): column for column, name in enumerate(cards["feature_names"])}
    mapped = np.zeros((1, cards["features"].shape[1]), dtype=cards["features"].dtype)
    for column, name in enumerate(names):
        if name not in index:
            if features[0, column] != 0:
                raise ValueError(f"New printed feature requires schema change: {name}")
        else:
            mapped[0, index[name]] = features[0, column]
    text = card_text(row)
    embedding = embed_texts([text], str(cards["text_model"]))
    additions = {"ids": [row["cardId"]], "titles": [row["title"]], "sides": [row["side"]],
                 "features": mapped, "text_emb": embedding,
                 "text_hashes": [hashlib.sha256(text.encode()).hexdigest()]}
    for key, values in additions.items():
        cards[key] = np.concatenate((cards[key], np.asarray(values, dtype=cards[key].dtype)), axis=0)
    table = CardTable(cards)
    table.save(ROOT / "cards-v1.npz")
    metadata = json.loads((ROOT / "title-display-metadata.json").read_text())
    metadata.append({"blueprint": row["cardId"], "title": row["title"], "display": row["title"],
                     "perSystem": False, "virtual": False, "alternateImage": False, "unavailableInEngine": False})
    (ROOT / "title-display-v1.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {"card": row["cardId"], "title": row["title"], "rows": len(table.ids), "sha": table.sha,
              "user_approved": True, "text_model": str(cards["text_model"]), "normalized_embeddings": True,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "printed_source_sha256": hashlib.sha256((ROOT / "missing-card.jsonl").read_bytes()).hexdigest()}
    Path("reports/data/knowledge-v3.1/legacy-missing-card.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
