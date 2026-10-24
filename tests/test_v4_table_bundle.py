import json

import numpy as np
import pytest

from swccg_policy.cards import SPECIAL
from swccg_policy.soft_vocabulary import SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_tables import CardTable


def cards(ids):
    ids = [*SPECIAL, *ids]
    count = len(ids)
    return {"ids": np.array(ids), "titles": np.array(ids), "sides": np.array([""] * count),
            "features": np.zeros((count, 2), np.float32), "text_emb": np.zeros((count, 768), np.float32),
            "text_hashes": np.array([""] * count), "feature_names": np.array(["a", "b"])}


def embed(texts):
    return np.ones((len(texts), 768), np.float16)


def test_bundle_save_growth_and_directional_compatibility(tmp_path):
    old = TableBundle(CardTable(cards(["1_1"])), SoftVocabulary.build({"q": ["known"]}, embed, "seed"))
    with pytest.raises(ValueError, match="explicit table growth"):
        old.cards.lookup("200_1")
    new = TableBundle(old.cards.grow(cards(["1_1", "200_1"])),
                      SoftVocabulary.build({"q": ["new"]}, embed, "growth", old.vocabulary),
                      old.sha, [old.sha])
    old.assert_prefix_of(new)
    assert new.cards.lookup("200_1") == 5
    assert old.vocabulary.lookup("q", "new") < 512 <= new.vocabulary.lookup("q", "new")
    with pytest.raises(ValueError, match="Unrelated"):
        new.assert_prefix_of(old)
    old.save(tmp_path / "old")
    new.save(tmp_path / "new")
    loaded = TableBundle.load(tmp_path / "new")
    TableBundle.load(tmp_path / "old").assert_prefix_of(loaded)
    assert loaded.sha == new.sha
    manifest = json.loads((tmp_path / "new/tables.json").read_text())
    manifest["vocabulary_sha"] = "wrong"
    (tmp_path / "new/tables.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        TableBundle.load(tmp_path / "new")


def test_card_growth_refuses_replacement_and_nonprefix():
    old = CardTable(cards(["1_1"]))
    modified = cards(["1_1", "200_1"])
    modified["features"][4, 0] = 1
    with pytest.raises(ValueError, match="replace existing card"):
        old.grow(modified)
    reordered = CardTable(cards(["200_1", "1_1"]), old.sha, [old.sha])
    with pytest.raises(ValueError, match="not a prefix"):
        old.assert_prefix_of(reordered)
