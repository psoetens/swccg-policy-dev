"""Frozen schema-4 vocabularies with observable, collision-prone OOV buckets."""

from collections import Counter
import hashlib


FAMILIES = ("q", "o", "context")
OOV_BUCKETS = 512
HASH = "fnv1a32-utf8"


def buckets(payload, family):
    """Plan actor-v6-inputs: reserved OOV buckets of one family. Context version 5 tables can have none
    (`context_oov_buckets: 0`, keys from closed engine enums); every other family and older table has 512."""
    return payload.get("context_oov_buckets", OOV_BUCKETS) if family == "context" else OOV_BUCKETS


def fnv1a(text):
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


class Vocabulary:
    def __init__(self, payload):
        if payload.get("schema_version") != 4 or payload.get("tensor_abi") != 4:
            raise ValueError("schema-4 vocabulary contract mismatch")
        if payload.get("oov_buckets") != OOV_BUCKETS or payload.get("oov_hash") != HASH:
            raise ValueError("unsupported frozen OOV contract")
        self.payload = payload
        self.total = Counter()
        self.unknown = {family: Counter() for family in FAMILIES}
        for family in FAMILIES:
            table = payload[family]
            reserved = {f"<oov:{index:03d}>": index for index in range(OOV_BUCKETS)}
            if any(table.get(key) != value for key, value in reserved.items()):
                raise ValueError(f"invalid reserved {family} buckets")
            if sorted(table.values()) != list(range(len(table))) or len(table) > 2**31:
                raise ValueError(f"invalid {family} vocabulary IDs")

    def lookup(self, family, text, count=1):
        if count < 1:
            raise ValueError("lookup count must be positive")
        self.total[family] += count
        value = self.payload[family].get(text)
        if value is not None:
            return value
        examples = self.unknown[family]
        key = text if text in examples or len(examples) < 100 else "<other>"
        examples[key] += count
        return fnv1a(text) % OOV_BUCKETS

    def coverage(self):
        return {
            family: {
                "total": self.total[family],
                "unknown": sum(self.unknown[family].values()),
                "examples": dict(self.unknown[family]),
            }
            for family in FAMILIES
        }

    @staticmethod
    def build(q=(), o=(), context=()):
        payload = {"schema_version": 4, "tensor_abi": 4, "oov_buckets": OOV_BUCKETS, "oov_hash": HASH}
        for family, names in zip(FAMILIES, (q, o, context), strict=True):
            table = {f"<oov:{index:03d}>": index for index in range(OOV_BUCKETS)}
            keys = sorted(set(names), key=lambda key: key.encode("utf-16-be"))
            if any(key in table for key in keys):
                raise ValueError("template collides with reserved bucket name")
            table.update({key: OOV_BUCKETS + index for index, key in enumerate(keys)})
            payload[family] = table
        Vocabulary(payload)
        return payload


def fingerprint(payload):
    if payload.get("schema_version") != 4 or payload.get("tensor_abi") != 4:
        raise ValueError("schema-4 vocabulary contract mismatch")
    digest = hashlib.sha256()
    digest.update(
        f"schema=4\nabi=4\noov_buckets={payload['oov_buckets']}\noov_hash={payload['oov_hash']}\n".encode()
    )
    for family in FAMILIES:
        for key, value in sorted(payload[family].items(), key=lambda item: item[0].encode("utf-16-be")):
            digest.update(f"{family}\t{key}\t{value}\n".encode())
    return digest.digest()
