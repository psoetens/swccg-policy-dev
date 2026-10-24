import copy

import numpy as np
import pytest
import torch

from swccg_policy.soft_vocabulary import SoftVocabulary, fingerprint, readable_text
from swccg_policy.text_priors import TextPriorEmbedding


def embed(texts):
    return np.stack([np.full(768, (sum(map(ord, text)) % 17) / 17) for text in texts])


def test_prior_initialization_and_growth_outputs():
    old = SoftVocabulary.build({"q": ["Deploy <c>", "Use <n> Force"]}, embed, "seed")
    module = TextPriorEmbedding(old.priors["q"], 16)
    table = module.table().detach().clone()
    assert torch.equal(module.residual.weight[512:], torch.zeros_like(module.residual.weight[512:]))
    expected = module.projection(torch.nn.functional.pad(module.priors[512:].float(), (0, 0, 0, 62)))[:2]
    assert torch.equal(table[512:], expected)
    # Include enough rows to change the number of projection blocks.
    grown = SoftVocabulary.build({"q": [f"Novel option {index}" for index in range(130)]}, embed, "extension", old)
    old.assert_prefix_of(grown)
    assert old.lookup("q", "Novel option 17") < 512
    assert grown.lookup("q", "Novel option 17") >= 512
    module.grow(grown.priors["q"])
    assert torch.equal(module.table()[:len(table)], table)
    assert readable_text("context", "rules._someField.N") == "rules  someField N"


def test_reject_nonprefix_and_modified_priors():
    old = SoftVocabulary.build({"q": ["a"]}, embed, "seed")
    other = SoftVocabulary.build({"q": ["b"]}, embed, "different")
    with pytest.raises(ValueError, match="Unrelated"):
        old.assert_prefix_of(other)
    grown = SoftVocabulary.build({"q": ["b"]}, embed, "next", old)
    bad = copy.deepcopy(grown.payload)
    bad["q"]["a"], bad["q"]["b"] = bad["q"]["b"], bad["q"]["a"]
    bad["sha"] = fingerprint(bad)
    tampered = SoftVocabulary(bad, grown.priors)
    with pytest.raises(ValueError, match="prefix"):
        old.assert_prefix_of(tampered)
    corrupt = {key: value.copy() for key, value in old.priors.items()}
    corrupt["q"][512, 0] += 1
    with pytest.raises(ValueError, match="hash"):
        SoftVocabulary(old.payload, corrupt)


def test_bounded_complete_unknown_telemetry():
    table = SoftVocabulary.build({}, embed, "empty")
    table.max_unknown = 2
    for key in ("first", "second", "third", "first"):
        table.lookup("q", key)
    assert table.coverage()["q"] == {
        "total": 4, "unknown": 4, "keys": {"first": 2, "second": 1}, "overflow_lookups": 1
    }


def test_normalizer_v2_child_keeps_the_prefix():
    # Plan titles-and-learner-ablation: tables-v7 is a v1 bundle's child that reads marked card names.
    from swccg_policy.soft_vocabulary import NORMALIZER, NORMALIZER_V2
    old = SoftVocabulary.build({"o": ["Stack 3B3-888"]}, embed, "seed")
    grown = SoftVocabulary.build({"o": ["Stack <c>"]}, embed, "marks", old, normalizer=NORMALIZER_V2)
    old.assert_prefix_of(grown)
    assert old.payload["normalizer_version"] == NORMALIZER
    assert grown.payload["normalizer_version"] == NORMALIZER_V2
    assert grown.lookup("o", "Stack <c>") >= 512 and grown.lookup("o", "Stack 3B3-888") >= 512
    SoftVocabulary(grown.payload, grown.priors)
    bad = copy.deepcopy(grown.payload)
    bad["normalizer_version"] = "knowledge-v3.1-amendment3-v9"
    bad["sha"] = fingerprint(bad)
    with pytest.raises(ValueError, match="version"):
        SoftVocabulary(bad, grown.priors)
    with pytest.raises(ValueError, match="Unknown normalizer"):
        SoftVocabulary.build({"o": ["x"]}, embed, "bad", old, normalizer="v9")
