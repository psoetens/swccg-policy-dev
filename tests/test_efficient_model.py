import copy

import numpy as np
import pytest
import torch

from swccg_policy.cards import SPECIAL
from swccg_policy.efficient import EfficientConfig, EfficientPolicy, EfficientCritic
from swccg_policy.soft_vocabulary import SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_contract import validate
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_tables import CardTable


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    ids = [*SPECIAL, "1_1", "1_2", "1_3"]
    cards = {"ids": np.array(ids), "titles": np.array(ids), "sides": np.array([""] * len(ids)),
             "features": np.random.default_rng(1).normal(size=(len(ids), 6)).astype(np.float32),
             "text_emb": np.random.default_rng(2).normal(size=(len(ids), 768)).astype(np.float32),
             "text_hashes": np.array([""] * len(ids)), "feature_names": np.array(list("abcdef"))}
    vocabulary = SoftVocabulary.build({"q": ["choose"], "o": ["act"], "context": ["battle.active"]},
                                      lambda texts: np.ones((len(texts), 768), np.float16), "seed")
    tables = TableBundle(CardTable(cards), vocabulary)
    cfg = EfficientConfig(len(ids), 6, 768, 513, 513, 513, [0, 1],
                          d=32, ff=64, layers=1, relation_layers=2, option_chunk=1)
    grid = np.zeros((1, 153), np.uint8)
    grid[0, :3] = [1, 2, 1]
    record = [4, "game", 1, "DS", "CARD_ACTION_CHOICE", "choose", [], [0] * 18, [
        [0, "1_1", 0, 1, -1, -1, 1, 0, 0, 0, 0, 0],
        [0, "1_1", 0, 1, -1, -1, 1, 0, 0, 0, 0, 0],
        [0, "1_2", 3, 1, -1, -1, 1, 0, 0, 0, 0, 0],
        [8, "<pad>", 0, 0, 2, 2, 1, 0, 1, 0, 0, 0],
        [9, "1_2", 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
        [9, "1_3", 0, 0, 1, -1, 1, 0, 0, 0, 0, 0],
        [10, "1_2", 0, 0, 0, -1, 1, 0, 2, 0, 0, 0],
        [2, "1_2", 0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
        [7, "<pad>", 0, 1, "battle.active", 0, -1, 0, -1, 1, -1, -1]],
        [["act", [], 0, "1_1", "a", None], ["act", [], 1, "1_1", "b", None]], [0], grid.tolist()]
    return tables, cfg, record


def test_grid_scope_bounds_exclusion_context_and_direction():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables).eval()
    original, _ = actor.encode(stack([parse(record, tables)]))
    for change in ("scope", "bound", "assertion", "index", "source", "direction"):
        altered = copy.deepcopy(record)
        if change == "scope":
            altered[11][0][:6] = [0, 0, 0, 1, 2, 1]
        elif change == "bound":
            altered[11][0][1] = 3
        elif change == "assertion":
            altered[11][0][2] = 0
        elif change == "index":
            altered[8][-1][8] = 2
        elif change == "source":
            altered[8][-1][5] = 1
        else:
            altered[8][3][5], altered[8][3][8] = 0, 3
        changed, _ = actor.encode(stack([parse(altered, tables)]))
        assert (changed - original).abs().max() > 1e-7, change


def test_unit_grid_indicators_are_gated_and_bounds_keep_their_scale():
    from swccg_policy.efficient import grid_features
    raw = torch.zeros(1, 153)
    raw[0, :15] = torch.tensor([1, 2, 1, 0, 0, 1, 3, 3, 1, 0, 0, 0, 1, 2, 0])
    scaled, combined = grid_features(raw)
    torch.testing.assert_close(scaled[0, :3], torch.tensor([1 / 60, 2 / 60, 1.0]))
    assert combined.shape == (1, 306)
    assert torch.equal(combined[0, 153:168],
                       torch.tensor([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]))
    assert torch.equal(combined[:, :153], scaled)
    assert torch.equal(grid_features(raw, False)[1], scaled)


def test_grid_validation_and_no_virtual_fallback():
    tables, _, record = fixture()
    for change in ("private", "overflow", "schema", "virtual", "rules"):
        altered = copy.deepcopy(record)
        if change == "private":
            altered[11][0][6:9] = [1, 1, 1]
        elif change == "overflow":
            altered[11][0][1] = 256
        elif change == "schema":
            altered[0] = 3
        elif change == "virtual":
            altered[8][0][1] = "200_1"
        else:
            altered[8][-1][4] = "rules.hidden"
        with pytest.raises(ValueError):
            parse(altered, tables)
    item = parse(record, tables)
    invalid = copy.deepcopy(item)
    invalid["tok"][7, 4] = 2
    with pytest.raises(ValueError, match="bound exactly once"):
        validate(invalid, tables.vocabulary, len(tables.cards.ids), actor=True)
    with pytest.raises(TypeError, match="already-validated"):
        validate(item, tables.vocabulary.payload, len(tables.cards.ids), actor=True)


def test_permutations_and_chunking():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables).eval()
    batch = stack([parse(record, tables)])
    expected = actor(batch)
    permutation = torch.tensor([6, 2, 0, 5, 3, 1, 4, 8, 7])
    inverse = torch.argsort(permutation)
    changed = {key: value.clone() for key, value in batch.items()}
    for key in ("tok", "tokf", "tmask"):
        changed[key] = changed[key][:, permutation]
    linked = changed["tok"][0, :, 5] > 0
    changed["tok"][0, linked, 5] = (inverse[changed["tok"][0, linked, 5].long() - 1] + 1).int()
    edge = changed["tok"][0, :, 0] == 8
    changed["tokf"][0, edge, 0] = (inverse[changed["tokf"][0, edge, 0].long() - 1] + 1).float()
    changed["opt"][0, :, 2] = (inverse[changed["opt"][0, :, 2].long() - 1] + 1).int()
    torch.testing.assert_close(expected, actor(changed), atol=1e-6, rtol=1e-5)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["opt"] = changed["opt"][:, [1, 0]]
    changed["tokf"][0, 6, 0] = 1
    torch.testing.assert_close(expected[:, [1, 0]], actor(changed), atol=1e-6, rtol=1e-5)
    actor.cfg.option_chunk = 100
    torch.testing.assert_close(expected, actor(batch), atol=1e-6, rtol=1e-5)


def test_bfloat16_gradients_and_independent_critic():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables)
    critic = EfficientCritic(copy.deepcopy(cfg), tables)
    batch = stack([parse(record, tables)])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = actor(batch).square().sum() + critic(batch).square().sum()
    assert torch.isfinite(loss)
    loss.backward()
    for model in (actor, critic):
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert critic.encoder.grid[0].weight.grad.abs().sum() > 0
    assert actor.context.projection.weight.grad.abs().sum() > 0


def test_model_growth_and_cached_tables_are_exact():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    batch = stack([parse(record, tables)])
    vocabulary = SoftVocabulary.build({"q": ["new"]},
                                      lambda texts: np.ones((len(texts), 768), np.float16), "growth", tables.vocabulary)
    grown = TableBundle(tables.cards, vocabulary, tables.sha, [tables.sha])
    with torch.no_grad():
        expected, value = actor(batch), critic(batch)
        actor.cache_card_table()
        critic.encoder.cache_card_table()
        assert torch.equal(actor(batch), expected)
        assert torch.equal(critic(batch), value)
        actor.grow(tables, grown)
        critic.encoder.grow(tables, grown)
        assert torch.equal(actor(batch), expected)
        assert torch.equal(critic(batch), value)


def test_checkpoint_owns_tables_and_accepts_only_prefix_growth(tmp_path):
    from swccg_policy.v4_checkpoint import save, load
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    path = tmp_path / "model.pt"
    save(path, actor, critic, tables)
    batch = stack([parse(record, tables)])
    vocabulary = SoftVocabulary.build({"q": ["new"]},
                                      lambda texts: np.ones((len(texts), 768), np.float16), "growth", tables.vocabulary)
    grown = TableBundle(tables.cards, vocabulary, tables.sha, [tables.sha])
    loaded, loaded_critic, loaded_tables, _ = load(path, tables=grown)
    assert loaded_tables.sha == grown.sha
    assert torch.equal(actor(batch), loaded(batch))
    assert torch.equal(critic(batch), loaded_critic(batch))
    unrelated = TableBundle(tables.cards, vocabulary)
    with pytest.raises(ValueError, match="Unrelated"):
        load(path, tables=unrelated)
    cfg.grid_indicators = False
    original_actor = EfficientPolicy(cfg, tables).eval()
    original_critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    old_path = tmp_path / "before-amendment5.pt"
    save(old_path, original_actor, original_critic, tables)
    payload = torch.load(old_path, weights_only=False)
    for key in ("cfg", "critic_cfg"):
        del payload[key]["grid_indicators"]
    torch.save(payload, old_path)
    restored, _, _, _ = load(old_path)
    assert restored.grid[0].in_features == 153
    assert torch.equal(original_actor(batch), restored(batch))


def test_torchscript_dynamic_grids_context_edges_and_weight_update(tmp_path):
    from swccg_policy.v4_native import export_ts, update_traced
    tables, cfg, record = fixture()
    actor = EfficientPolicy(cfg, tables).eval()
    traced = export_ts(actor, tmp_path / "actor.ts")
    batch = stack([parse(record, tables)])
    head = torch.cat([batch[key][:, None] for key in ("dtype", "qtmpl", "qbp", "side")] + [batch["glob"]], 1)
    arguments = [batch[key] for key in ("tok", "tokf", "tmask", "opt", "omask")] + [head, batch["kgrid"]]
    torch.testing.assert_close(actor(batch), traced(*arguments), atol=1e-6, rtol=1e-5)
    routes = []
    kind = batch["tok"][..., 0]
    for mask in ((kind == 8) | ((kind == 7) & (batch["tok"][..., 5] > 0)), kind == 9, kind == 10):
        routes.append(torch.cat((mask.flatten().nonzero().flatten(), torch.full((3,), -1, dtype=torch.long))))
    torch.testing.assert_close(actor(batch), traced.forward_packed(*arguments, *routes), atol=1e-6, rtol=1e-5)
    assert "aten::nonzero" not in str(traced.forward_packed.inlined_graph)
    addresses = {key: value.data_ptr() for key, value in traced.state_dict().items()}
    with torch.no_grad():
        actor.score[-1].bias.add_(0.1)
        actor.context.projection.weight.add_(0.01)
    update_traced(traced, actor)
    assert all(value.data_ptr() == addresses[key] for key, value in traced.state_dict().items())
    torch.testing.assert_close(actor(batch), traced(*arguments), atol=1e-6, rtol=1e-5)
