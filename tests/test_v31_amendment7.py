import copy
from pathlib import Path
import sys

import numpy as np
import torch

from test_efficient_model import fixture
from swccg_policy.efficient import EfficientPolicy, EfficientCritic
from swccg_policy.v4_checkpoint import load, save
from swccg_policy.v4_data import parse, stack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v31_probe_tasks import GATE_BUDGET, gate_budget, printed_example, printed_split


def test_balanced_init_changes_only_categorical_parameters_for_both_views():
    tables, cfg, _ = fixture()
    for constructor in (EfficientPolicy, EfficientCritic):
        cfg.embedding_init_std = 1.0
        torch.manual_seed(912)
        original = constructor(copy.deepcopy(cfg), tables)
        cfg.embedding_init_std = 0.02
        torch.manual_seed(912)
        balanced = constructor(copy.deepcopy(cfg), tables)
        before = original if constructor is EfficientPolicy else original.encoder
        after = balanced if constructor is EfficientPolicy else balanced.encoder
        categorical = {name + ".weight" for name in after.categorical_embeddings()}
        for name, value in before.state_dict().items():
            if name in categorical:
                assert not torch.equal(value, after.state_dict()[name])
                assert 0.01 < float(after.state_dict()[name].std()) < 0.03
            else:
                assert torch.equal(value, after.state_dict()[name]), name
        assert torch.count_nonzero(after.cards.delta.weight) == 0
        for name in ("context", "qtmpl") + (("otmpl",) if after.actor else ()):
            assert torch.equal(getattr(before, name).residual.weight, getattr(after, name).residual.weight)


def test_saved_initialization_setting_and_legacy_weights_survive_load(tmp_path):
    tables, cfg, record = fixture()
    cfg.embedding_init_std = 0.07
    actor = EfficientPolicy(cfg, tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    batch = stack([parse(record, tables)])
    path = tmp_path / "model.pt"
    save(path, actor, critic, tables)
    restored, restored_critic, _, payload = load(path)
    assert restored.cfg.embedding_init_std == 0.07
    assert restored_critic.cfg.embedding_init_std == 0.07
    assert torch.equal(restored(batch), actor(batch))
    assert torch.equal(restored_critic(batch), critic(batch))
    for key in ("cfg", "critic_cfg"):
        del payload[key]["embedding_init_std"]
    legacy, legacy_critic, _, _ = load(None, payload=payload)
    assert legacy.cfg.embedding_init_std == 1.0
    assert torch.equal(legacy(batch), actor(batch))
    assert torch.equal(legacy_critic(batch), critic(batch))


def test_only_exact_budget_qualifies():
    assert gate_budget(GATE_BUDGET)
    for key, value in {"steps": 2000, "lr": 1e-3, "batch": 20, "eval": 1000}.items():
        assert not gate_budget({**GATE_BUDGET, key: value})


def test_powered_split_includes_printed_zero_excludes_missing_and_disjoins_blueprints():
    from types import SimpleNamespace
    ids = ["<pad>", "<unk>", "-1_1", "-1_2", *[f"1_{i}" for i in range(12)]]
    features = np.zeros((16, 2), np.float32)
    features[4:14, 1] = 1
    features[4:14, 0] = np.arange(10) / 5
    cards = SimpleNamespace(ids=ids, cards={"features": features, "feature_names": ["num:power", "has:power"]})
    power, pools, missing, _ = printed_split(SimpleNamespace(cards=cards))
    assert set(pools[False]).isdisjoint(pools[True])
    assert set(pools[False]) | set(pools[True]) == set(range(4, 14))
    assert missing.tolist() == [14, 15]
    rng = np.random.default_rng(7)
    for heldout in (False, True):
        for _ in range(50):
            item, label = printed_example(rng, power, pools[heldout])
            pair = item["opt"][:, 3]
            assert set(pair).issubset(pools[heldout])
            assert power[pair[0]] != power[pair[1]]
            assert label == int(power[pair[1]] > power[pair[0]])
        item, _ = printed_example(rng, power, pools[heldout], missing)
        assert sum(index in missing for index in item["opt"][:, 3]) == 1
