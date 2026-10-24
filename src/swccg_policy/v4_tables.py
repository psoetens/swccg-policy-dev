"""Versioned append-only card rows and exact prefix compatibility for ABI 4."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .cards import SPECIAL


ROW_FIELDS = ("ids", "titles", "sides", "features", "text_emb", "text_hashes")


def array_digest(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + b"\n")
    digest.update(json.dumps(array.shape).encode() + b"\n")
    if array.dtype.kind in "USO":
        digest.update(json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode())
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def card_content_sha(cards):
    fields = {key: array_digest(value) for key, value in sorted(cards.items())}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CardTable:
    def __init__(self, cards, parent_sha=None, ancestors=()):
        self.cards = {key: np.asarray(value) for key, value in cards.items()}
        self.ids = list(map(str, self.cards["ids"]))
        if self.ids[:4] != SPECIAL or len(set(self.ids)) != len(self.ids):
            raise ValueError("Invalid card IDs/special rows")
        if any(len(self.cards[key]) != len(self.ids) for key in ROW_FIELDS):
            raise ValueError("Card row lengths disagree")
        if self.cards["text_emb"].shape[1] != 768:
            raise ValueError("Card text priors require 768 dimensions")
        self.index = {blueprint: index for index, blueprint in enumerate(self.ids)}
        self.parent_sha = parent_sha
        self.ancestors = tuple(ancestors)
        if bool(parent_sha) != bool(self.ancestors) or parent_sha and self.ancestors[-1] != parent_sha:
            raise ValueError("Invalid card parent chain")
        self.content_sha = card_content_sha(self.cards)
        self.sha = hashlib.sha256(json.dumps({"content": self.content_sha, "parent": parent_sha,
            "ancestors": self.ancestors}, sort_keys=True).encode()).hexdigest()

    def lookup(self, blueprint):
        stripped = blueprint.rstrip("*^")
        if stripped in self.index:
            return self.index[stripped]
        if stripped.split("_", 1)[0].isdigit() and int(stripped.split("_", 1)[0]) >= 200:
            raise ValueError(f"Virtual blueprint {blueprint} requires explicit table growth")
        raise ValueError(f"Blueprint absent from this card table: {blueprint}")

    def assert_prefix_of(self, newer):
        if self.sha != newer.sha and self.sha not in newer.ancestors:
            raise ValueError("Unrelated card table SHA")
        if newer.ids[:len(self.ids)] != self.ids:
            raise ValueError("Card table IDs are not a prefix")
        if set(self.cards) != set(newer.cards):
            raise ValueError("Card table fields changed")
        for key, value in self.cards.items():
            other = newer.cards[key][:len(self.ids)] if key in ROW_FIELDS else newer.cards[key]
            # Unicode arrays may widen when appending longer IDs; compare values exactly.
            if not np.array_equal(value, other):
                raise ValueError(f"Existing card table content changed: {key}")

    def grow(self, additions):
        additions = {key: np.asarray(value) for key, value in additions.items()}
        for key, value in self.cards.items():
            if key not in ROW_FIELDS and not np.array_equal(value, additions[key]):
                raise ValueError(f"Card feature/text schema changed: {key}")
        indices = [index for index, blueprint in enumerate(additions["ids"]) if str(blueprint) not in self.index]
        for index, blueprint in enumerate(additions["ids"]):
            if str(blueprint) in self.index:
                old = self.index[str(blueprint)]
                for key in ROW_FIELDS:
                    if not np.array_equal(self.cards[key][old], additions[key][index]):
                        raise ValueError(f"Growth tried to replace existing card {blueprint}: {key}")
        cards = {key: np.concatenate((value, additions[key][indices]), axis=0) if key in ROW_FIELDS else value
                 for key, value in self.cards.items()}
        result = CardTable(cards, self.sha, (*self.ancestors, self.sha))
        self.assert_prefix_of(result)
        return result

    def save(self, path):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.cards)
        manifest = {"version": len(self.ancestors) + 1, "sha": self.sha, "parent_sha": self.parent_sha,
                    "ancestors": self.ancestors, "content_sha": self.content_sha, "rows": len(self.ids)}
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads(path.with_suffix(".json").read_text())
        table = cls(dict(np.load(path)), manifest["parent_sha"], manifest["ancestors"])
        if table.sha != manifest["sha"] or table.content_sha != manifest["content_sha"]:
            raise ValueError("Card table content/SHA mismatch")
        return table
