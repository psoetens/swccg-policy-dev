import copy

import numpy as np
import pytest

from test_efficient_model import fixture
from swccg_policy.soft_vocabulary import SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_contract import validate
from swccg_policy.v4_data import parse
from swccg_policy.v4_trajectory import TrajectoryReader, load_binary, write_binary


def game(tables, record):
    actor = parse(record, tables)
    critic_record = copy.deepcopy(record)
    critic_record.extend([record[8], 123, record[11]])
    critic_record[12] = [token for token in record[8] if token[0] != 10]
    # Context source is before the removed option reference; no target remapping needed.
    critic = parse(critic_record, tables, critic=True)
    return {"items": [actor], "critic_items": [critic], "seqs": [[1]], "excluded": [None],
            "policy_versions": [0], "old_logp": [-0.75], "knowledge_sequences": [123], "g": "game"}


@pytest.mark.parametrize("level", [0, 1])
def test_binary_exact_and_prefix_acceptance(tmp_path, level):
    tables, _, record = fixture()
    original = game(tables, record)
    path = tmp_path / "part-0.traj"
    write_binary(path, [original], tables, level)
    vocabulary = SoftVocabulary.build({"q": ["new"]},
                                      lambda texts: np.ones((len(texts), 768), np.float16), "growth", tables.vocabulary)
    grown = TableBundle(tables.cards, vocabulary, tables.sha, [tables.sha])
    loaded = load_binary(path, grown, acting_tables=tables)[0]
    registered = TrajectoryReader(grown, [tables]).load(path)[0]
    assert registered["table_sha"] == tables.sha
    with pytest.raises(ValueError, match="explicit registry"):
        TrajectoryReader(grown).load(path)
    for view in ("items", "critic_items"):
        for key in ("tok", "tokf", "head", "opt", "kgrid"):
            np.testing.assert_array_equal(original[view][0][key], loaded[view][0][key])
    assert loaded["table_sha"] == tables.sha
    assert loaded["knowledge_sequences"] == [123]
    with pytest.raises(ValueError, match="Privileged payload"):
        validate(loaded["critic_items"][0], tables.vocabulary, len(tables.cards.ids), actor=True)
    with pytest.raises(ValueError, match="acting table SHA"):
        load_binary(path, grown)
    unrelated = TableBundle(tables.cards, vocabulary)
    with pytest.raises(ValueError, match="Unrelated"):
        TrajectoryReader(unrelated, [tables])
    with pytest.raises(ValueError, match="Unrelated"):
        load_binary(path, unrelated, acting_tables=tables)
    path.write_bytes(b"SWTRJ999" + path.read_bytes()[8:])  # any other format: refused
    with pytest.raises(ValueError, match="SWTRJ003 required"):
        load_binary(path, tables)
