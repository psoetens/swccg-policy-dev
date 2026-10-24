import pytest

from swccg_policy.v4_vocabulary import Vocabulary, fingerprint, fnv1a


def test_frozen_oov_buckets_and_known_ids():
    payload = Vocabulary.build(q=["Deploy <c>"], context=["usage.game|DEPLOY"])
    vocabulary = Vocabulary(payload)
    assert vocabulary.lookup("q", "Deploy <c>") == 512
    assert fnv1a("") == 0x811C9DC5
    assert fnv1a("hello") == 0x4F9F2CAB
    assert vocabulary.lookup("q", "hello") == 0x4F9F2CAB % 512
    assert vocabulary.lookup("q", "another") != vocabulary.lookup("q", "hello")
    assert vocabulary.coverage()["q"]["unknown"] == 3
    assert vocabulary.coverage()["q"]["total"] == 4


def test_vocabulary_contract_is_frozen_and_versioned():
    payload = Vocabulary.build(q=["second", "first"])
    assert fingerprint(payload) == fingerprint(Vocabulary.build(q=["first", "second", "first"]))
    for key, value in (("tensor_abi", 3), ("oov_buckets", 256), ("oov_hash", "other")):
        invalid = {**payload, key: value}
        with pytest.raises(ValueError):
            Vocabulary(invalid)
    assert fingerprint({**payload, "oov_buckets": 256}) != fingerprint(payload)
    assert fingerprint({**payload, "oov_hash": "other"}) != fingerprint(payload)


def test_oov_collisions_are_not_identities():
    seen = {}
    collision = None
    for index in range(513):
        text = f"unseen {index}"
        bucket = fnv1a(text) % 512
        if bucket in seen:
            collision = (seen[bucket], text)
            break
        seen[bucket] = text
    assert collision is not None
    vocabulary = Vocabulary(Vocabulary.build())
    assert vocabulary.lookup("context", collision[0]) == vocabulary.lookup("context", collision[1])
    assert vocabulary.coverage()["context"]["unknown"] == 2


def test_indexed_lookup_counts_match_individual_lookups():
    indexed = Vocabulary(Vocabulary.build(q=["known"]))
    individual = Vocabulary(indexed.payload)
    for key, count in (("known", 10), ("missing", 73), ("other", 42)):
        value = indexed.lookup("q", key, count=count)
        for _ in range(count):
            assert individual.lookup("q", key) == value
    assert indexed.coverage() == individual.coverage()


@pytest.mark.parametrize("replacement", [0, -1, 1000])
def test_frozen_ids_cannot_alias_buckets_or_leave_gaps(replacement):
    payload = Vocabulary.build(q=["known"])
    payload["q"]["known"] = replacement
    with pytest.raises(ValueError, match="vocabulary IDs"):
        Vocabulary(payload)
