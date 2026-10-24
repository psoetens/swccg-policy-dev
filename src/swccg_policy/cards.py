"""Card table: one row per Gemp blueprint with exact structured features and a frozen local text embedding.

Sources: Gemp's card database (card_blueprint_database.json in the cards module), which carries the blueprint data
(types, stats, icons, keywords, personas, species, lore, game text). Text embeddings are computed locally with a
sentence-transformers model; nothing leaves the machine.

    uv run python -m swccg_policy.cards --db PATH/card_blueprint_database.json --out data/cards/card_table.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

DEFAULT_DB = Path.home() / "src/gemp-swccg-agent/gemp-swccg-public/src/gemp-swccg-cards/src/main/resources/card_blueprint_database.json"
# Index 0: padding, 1: unknown blueprint, 2/3: Light/Dark card backs (face-down cards as the engine shows them).
SPECIAL = ["<pad>", "<unk>", "-1_1", "-1_2"]
NUMERIC = ["deployCost", "power", "ability", "forfeit", "destiny", "armor", "maneuver", "landspeed", "hyperspeed",
           "politics", "ferocity", "defenseValue", "pilotCapacity", "passengerCapacity", "vehicleCapacity",
           "astromechCapacity"]
BOOLEAN = ["isComboCard", "deploysLikeStarfighter", "movesLikeCharacter", "movesLikeStarfighter",
           "doesNotCountTowardDeckLimit", "hasVirtualSuffix", "isFrontOfDoubleSidedCard", "mayNotBePlacedInReserveDeck"]
CATEGORICAL = ["side", "cardCategory", "cardSubtype", "uniqueness", "species"]
MULTI = ["cardTypes", "keywords", "personas", "modelTypes"]


def load_db(path: Path) -> list[dict]:
    return json.load(open(path))


def card_text(c: dict) -> str:
    parts = [c.get("title") or ""]
    kind = " ".join(x for x in [c.get("side"), c.get("cardCategory"), " ".join(c.get("cardTypes") or []), c.get("cardSubtype")] if x)
    parts.append(kind.lower())
    for k in ("gameText", "locationDarkSideGameText", "locationLightSideGameText"):
        if c.get(k):
            parts.append(c[k])
    if c.get("lore"):
        parts.append("Lore: " + c["lore"])
    return ". ".join(p for p in parts if p)


# The card database writes "unlimited" capacity (Executor, Flagship Executor, Supremacy, Home One) as Java's Integer.MAX_VALUE.
# Scaled like a number it gave feature values of ~4e8 and card embeddings of ~8e8 (logits in the tens of thousands whenever such
# a ship was an option; reports/learner-throughput.md §5). Unlimited now counts as 16 (Peter: in practice at most 58 passengers,
# realistically far fewer; the largest finite capacity of any card is 8).
UNLIMITED = 2147483647
UNLIMITED_CAPACITY = 16.0


def numeric_value(c: dict, k: str) -> float | None:
    """The numeric attribute k of card c as a float (None if absent / NaN); unlimited capacity -> UNLIMITED_CAPACITY. Any other
    value >= 1000 is rejected: a new sentinel must be handled explicitly, not scaled into a huge feature."""
    v = c.get(k)
    if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return None
    if v == UNLIMITED and k.endswith("Capacity"):
        return UNLIMITED_CAPACITY
    if abs(v) >= 1000:
        raise ValueError(f"card {c.get('id')} {c.get('title')}: {k} = {v} (unhandled sentinel?)")
    return float(v)


def build_features(cards: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Structured feature matrix (rows = cards in order) and column names."""
    cols: list[str] = []
    vocab = {}
    for k in CATEGORICAL:
        vocab[k] = sorted({str(c.get(k)) for c in cards if c.get(k) is not None})
        cols += [f"{k}={v}" for v in vocab[k]]
    for k in MULTI:
        vocab[k] = sorted({v for c in cards for v in (c.get(k) or [])})
        cols += [f"{k}:{v}" for v in vocab[k]]
    icons = sorted({i["icon"] for c in cards for i in (c.get("icons") or [])})
    cols += [f"icon:{v}" for v in icons]
    cols += [f"num:{k}" for k in NUMERIC] + [f"has:{k}" for k in NUMERIC] + [f"bool:{k}" for k in BOOLEAN]
    index = {c: i for i, c in enumerate(cols)}
    x = np.zeros((len(cards), len(cols)), dtype=np.float32)
    for r, c in enumerate(cards):
        for k in CATEGORICAL:
            if c.get(k) is not None:
                x[r, index[f"{k}={c[k]}"]] = 1
        for k in MULTI:
            for v in c.get(k) or []:
                x[r, index[f"{k}:{v}"]] = 1
        for i in c.get("icons") or []:
            x[r, index[f"icon:{i['icon']}"]] = float(i.get("count", 1))
        for k in NUMERIC:
            v = numeric_value(c, k)
            if v is not None:
                x[r, index[f"num:{k}"]] = v / 5.0
                x[r, index[f"has:{k}"]] = 1
        for k in BOOLEAN:
            x[r, index[f"bool:{k}"]] = 1.0 if c.get(k) else 0.0
    return x, cols


def embed_texts(texts: list[str], model_name: str, batch: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="data/cards/card_table.npz")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--extra", help="JSONL of extra card rows (same fields as the database) appended after it")
    a = ap.parse_args()
    cards = load_db(Path(a.db))
    if a.extra:
        # Rows for blueprints the JSON database lacks, dumped from Gemp's blueprints (bench.CardDump); appended so the
        # indices of all database cards stay the same.
        known = {c["cardId"] for c in cards}
        extra = [json.loads(l) for l in open(a.extra) if l.strip()]
        cards += [c for c in extra if c["cardId"] not in known]
    feats, cols = build_features(cards)
    texts = [card_text(c) for c in cards]
    emb = embed_texts(texts, a.model)
    ids = SPECIAL + [c["cardId"] for c in cards]
    n_special = len(SPECIAL)
    feat_all = np.zeros((len(ids), feats.shape[1]), dtype=np.float32)
    feat_all[n_special:] = feats
    emb_all = np.zeros((len(ids), emb.shape[1]), dtype=np.float32)
    emb_all[n_special:] = emb
    titles = SPECIAL + [c.get("title") or "" for c in cards]
    sides = ["", "", "LIGHT", "DARK"] + [c.get("side") or "" for c in cards]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, ids=np.array(ids), titles=np.array(titles), sides=np.array(sides), features=feat_all,
                        text_emb=emb_all, feature_names=np.array(cols), text_model=np.array(a.model),
                        text_hashes=np.array([''] * n_special + [hashlib.sha256(t.encode()).hexdigest() for t in texts]),
                        feature_schema=np.array(2), unlimited_capacity=np.array(UNLIMITED_CAPACITY))
    print(f"{len(ids)} blueprints ({len(cards)} cards + {n_special} special), {feats.shape[1]} structured features, "
          f"text embedding {emb.shape[1]}-d ({a.model}) -> {a.out}")


if __name__ == "__main__":
    main()
