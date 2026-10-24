"""Append-only schema-4 template tables with offline text priors and bounded OOV telemetry."""

from collections import Counter
import copy
import hashlib
import json

import numpy as np

from .v4_vocabulary import FAMILIES, HASH, OOV_BUCKETS, buckets, fnv1a
from .v4_tables import array_digest


NORMALIZER = "knowledge-v3.1-amendment3-v1"
# Plan titles-and-learner-ablation: v2 reads the engine's marked card names (Java NameMarks / TitleMatcher).
NORMALIZER_V2 = "knowledge-v3.1-amendment3-v2"
NORMALIZERS = (NORMALIZER, NORMALIZER_V2)
PRIOR_MODEL = "BAAI/bge-base-en-v1.5"


def readable_text(family, key):
    return key.replace("_", " ").replace(".", " ") if family == "context" else key


def fingerprint(payload):
    content = {key: value for key, value in payload.items() if key != "sha"}
    def ordered(value):
        if isinstance(value, dict):
            return {key: ordered(value[key]) for key in sorted(value, key=lambda key: key.encode("utf-16-be"))}
        if isinstance(value, list):
            return [ordered(item) for item in value]
        return value
    return hashlib.sha256(json.dumps(ordered(content), ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


class SoftVocabulary:
    def __init__(self, payload, priors, max_unknown=100_000):
        self.payload = copy.deepcopy(payload)
        self.priors = {family: np.asarray(priors[family]) for family in FAMILIES}
        self.total = Counter()
        self.unknown = {family: Counter() for family in FAMILIES}
        self.overflow = Counter()
        self.max_unknown = max_unknown
        self.privileged_context_ids = np.asarray(
            [index for key, index in payload["context"].items() if key.startswith(("rules.", "priv."))], dtype=np.int32)
        expected = (4, 4, 2, OOV_BUCKETS, HASH, PRIOR_MODEL)
        actual = tuple(payload.get(key) for key in ("schema_version", "tensor_abi", "table_format",
                       "oov_buckets", "oov_hash", "prior_model"))
        if (actual != expected or payload.get("normalizer_version") not in NORMALIZERS
                or payload.get("sha") != fingerprint(payload)):
            raise ValueError("Soft vocabulary version/hash mismatch")
        if payload["version"] != len(payload["ancestors"]) + 1:
            raise ValueError("Vocabulary version/ancestor chain mismatch")
        if payload["parent_sha"] != (payload["ancestors"][-1] if payload["ancestors"] else None):
            raise ValueError("Vocabulary parent chain mismatch")
        for family in FAMILIES:
            table = payload[family]
            if len(table) > np.iinfo(np.int32).max:
                raise ValueError("Vocabulary exceeds ABI-4 int32 IDs")
            if sorted(table.values()) != list(range(len(table))):
                raise ValueError("Vocabulary IDs must be contiguous and unique")
            reserved = buckets(payload, family)
            if any(table.get(f"<oov:{index:03d}>") != index for index in range(reserved)) or any(
                    key.startswith("<oov:") and index >= reserved for key, index in table.items()):
                raise ValueError("Reserved bucket IDs changed")
            prior = self.priors[family]
            if prior.shape != (len(table), 768) or prior.dtype != np.float16 or not np.isfinite(prior).all():
                raise ValueError("Template prior shape/dtype/value mismatch")
            if np.any(prior[:reserved]) or array_digest(prior) != payload["prior_sha256"][family]:
                raise ValueError("Template prior hash or OOV rows mismatch")
            cursor = reserved
            for batch in payload["batches"]:
                start, end = batch[family]
                if start != cursor or end < start:
                    raise ValueError("Non-contiguous append batch")
                keys = [key for key, index in sorted(table.items(), key=lambda item: item[1]) if start <= index < end]
                if keys != sorted(keys, key=lambda key: key.encode("utf-16-be")):
                    raise ValueError("Vocabulary batch is not UTF-16 sorted")
                cursor = end
            if cursor != len(table):
                raise ValueError("Vocabulary batches do not cover all rows")

    @property
    def sha(self):
        return self.payload["sha"]

    def lookup(self, family, key, count=1):
        if count < 1:
            raise ValueError("Lookup count must be positive")
        self.total[family] += count
        if key in self.payload[family]:
            return self.payload[family][key]
        if not buckets(self.payload, family):
            raise KeyError(f"{family} key outside a vocabulary without OOV buckets: {key}")
        unknown = self.unknown[family]
        if key in unknown or sum(len(values) for values in self.unknown.values()) < self.max_unknown:
            unknown[key] += count
        else:
            self.overflow[family] += count
        return fnv1a(key) % OOV_BUCKETS

    def coverage(self):
        return {family: {"total": self.total[family],
                         "unknown": sum(self.unknown[family].values()) + self.overflow[family],
                         "keys": dict(self.unknown[family]), "overflow_lookups": self.overflow[family]}
                for family in FAMILIES}

    def assert_prefix_of(self, newer):
        if self.sha != newer.sha and self.sha not in newer.payload["ancestors"]:
            raise ValueError("Unrelated vocabulary SHA")
        if self.payload["batches"] != newer.payload["batches"][:len(self.payload["batches"])]:
            raise ValueError("Vocabulary batch history changed")
        for family in FAMILIES:
            table = self.payload[family]
            if any(newer.payload[family].get(key) != index for key, index in table.items()):
                raise ValueError("Vocabulary IDs are not a prefix")
            if not np.array_equal(self.priors[family], newer.priors[family][:len(table)]):
                raise ValueError("Existing template priors changed")

    @classmethod
    def build(cls, keys, embed, source, parent=None, context_buckets=OOV_BUCKETS, normalizer=None):
        """context_buckets (new vocabularies only): the context family's reserved OOV rows (plan actor-v6-inputs:
        0 for context version 5). normalizer: the child's normalizer version (plan titles-and-learner-ablation:
        NORMALIZER_V2 for tables-v7); default the parent's, or v1."""
        if parent is None:
            payload = {"schema_version": 4, "tensor_abi": 4, "table_format": 2,
                       "version": 1, "normalizer_version": NORMALIZER,
                       "oov_buckets": OOV_BUCKETS, "oov_hash": HASH, "prior_model": PRIOR_MODEL,
                       "parent_sha": None, "ancestors": [], "batches": [], "prior_sha256": {}}
            priors = {}
            if context_buckets != OOV_BUCKETS:
                payload["context_oov_buckets"] = context_buckets
            for family in FAMILIES:
                reserved = buckets(payload, family)
                payload[family] = {f"<oov:{index:03d}>": index for index in range(reserved)}
                priors[family] = np.zeros((reserved, 768), dtype=np.float16)
        else:
            payload = copy.deepcopy(parent.payload)
            payload.update(version=payload["version"] + 1, parent_sha=parent.sha,
                           ancestors=[*payload["ancestors"], parent.sha])
            priors = dict(parent.priors)
        if normalizer is not None:
            if normalizer not in NORMALIZERS:
                raise ValueError(f"Unknown normalizer {normalizer}")
            payload["normalizer_version"] = normalizer
        batch = {"source": source}
        for family in FAMILIES:
            table = payload[family]
            incoming = set(keys.get(family, ()))
            if any(key.startswith("<oov:") for key in incoming):
                raise ValueError("Reserved OOV names cannot be seeded")
            additions = sorted(incoming - set(table), key=lambda key: key.encode("utf-16-be"))
            start = len(table)
            if additions:
                vectors = np.asarray(embed([readable_text(family, key) for key in additions]), dtype=np.float16)
                if vectors.shape != (len(additions), 768):
                    raise ValueError("Text encoder dimension mismatch")
                priors[family] = np.concatenate((priors[family], vectors), axis=0)
            table.update({key: start + index for index, key in enumerate(additions)})
            batch[family] = [start, len(table)]
            payload["prior_sha256"][family] = array_digest(priors[family])
        payload["batches"].append(batch)
        payload["sha"] = fingerprint(payload)
        result = cls(payload, priors)
        if parent:
            parent.assert_prefix_of(result)
        return result
