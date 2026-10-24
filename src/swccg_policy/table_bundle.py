"""Own-table bundles and prefix checks shared by checkpoint and trajectory loaders."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .soft_vocabulary import SoftVocabulary
from .v4_tables import CardTable


class TableBundle:
    def __init__(self, cards, vocabulary, parent_sha=None, ancestors=()):
        self.cards = cards
        self.vocabulary = vocabulary
        self.parent_sha = parent_sha
        self.ancestors = tuple(ancestors)
        if parent_sha != (self.ancestors[-1] if self.ancestors else None):
            raise ValueError("Invalid bundle ancestry")
        identity = {"card_sha": cards.sha, "vocabulary_sha": vocabulary.sha,
                    "parent_sha": parent_sha, "ancestors": self.ancestors}
        self.sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def assert_prefix_of(self, newer):
        if self is newer:
            return
        if self.sha != newer.sha and self.sha not in newer.ancestors:
            raise ValueError("Unrelated acting/checkpoint table SHA")
        self.cards.assert_prefix_of(newer.cards)
        self.vocabulary.assert_prefix_of(newer.vocabulary)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.cards.save(directory / "cards.npz")
        (directory / "vocab.json").write_text(json.dumps(self.vocabulary.payload, ensure_ascii=True, indent=2) + "\n")
        np.savez_compressed(directory / "template_priors.npz", **self.vocabulary.priors)
        manifest = {"schema_version": 4, "tensor_abi": 4, "table_sha": self.sha, "card_sha": self.cards.sha,
                    "vocabulary_sha": self.vocabulary.sha, "parent_sha": self.parent_sha,
                    "ancestors": self.ancestors}
        (directory / "tables.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "tables.json").read_text())
        if (manifest.get("schema_version"), manifest.get("tensor_abi")) != (4, 4):
            raise ValueError("Table bundle schema/ABI mismatch")
        cards = CardTable.load(directory / "cards.npz")
        payload = json.loads((directory / "vocab.json").read_text())
        vocabulary = SoftVocabulary(payload, dict(np.load(directory / "template_priors.npz")))
        bundle = cls(cards, vocabulary, manifest["parent_sha"], manifest["ancestors"])
        if bundle.sha != manifest["table_sha"] or cards.sha != manifest["card_sha"]:
            raise ValueError("Table bundle hash mismatch")
        if vocabulary.sha != manifest["vocabulary_sha"]:
            raise ValueError("Table bundle vocabulary mismatch")
        return bundle
