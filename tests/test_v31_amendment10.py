import copy

import torch

from test_efficient_model import fixture
from swccg_policy.efficient import COMPONENTS, OPTION_COMPONENTS, EfficientCritic, EfficientPolicy
from swccg_policy.v4_checkpoint import load, save
from swccg_policy.v4_data import parse, stack


def models(cfg, tables, component_norm, seed=5, scope="all"):
    cfg = copy.deepcopy(cfg)
    cfg.component_norm = component_norm
    cfg.component_norm_scope = scope
    torch.manual_seed(seed)
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    return actor, critic


def test_component_norm_off_has_no_extra_parameters_and_legacy_checkpoints_load(tmp_path):
    tables, cfg, record = fixture()
    actor, critic = models(cfg, tables, False)
    assert actor.component_norms is None
    assert not any(name.startswith("component_norms") for name, _ in actor.named_parameters())
    path = tmp_path / "legacy.pt"
    save(path, actor, critic, tables)
    payload = torch.load(path, weights_only=False)
    for config in (payload["cfg"], payload["critic_cfg"]):
        del config["component_norm"]
    loaded, loaded_critic, _, _ = load(None, payload=payload)
    batch = stack([parse(record, tables)])
    assert loaded.cfg.component_norm is False
    assert torch.equal(actor(batch), loaded(batch))
    assert torch.equal(critic(batch), loaded_critic(batch))


def test_component_norm_on_normalizes_every_component_and_round_trips(tmp_path):
    tables, cfg, record = fixture()
    actor, critic = models(cfg, tables, True)
    assert set(actor.component_norms) == set(COMPONENTS) | set(OPTION_COMPONENTS)
    assert set(critic.encoder.component_norms) == set(COMPONENTS)
    batch = stack([parse(record, tables)])
    norms = {}
    with torch.no_grad():
        table = actor.cards.table()
        ids = batch["tok"][..., 1].long()
        norms["card"] = actor.part("card", torch.nn.functional.embedding(ids, table))[ids > 0].norm(dim=-1)
        norms["kind"] = actor.part("kind", actor.kind(batch["tok"][..., 0].long())).norm(dim=-1)
        norms["question"] = actor.part("question_template", actor.qtmpl(batch["qtmpl"].long())).norm(dim=-1)
    expected = {"card": 1.0, "question": 1.0, "kind": actor.cfg.component_label_gain}
    for name, values in norms.items():
        target = expected[name] * actor.cfg.d ** 0.5
        assert torch.allclose(values, torch.full_like(values, target), rtol=0.03), name
    path = tmp_path / "balanced.pt"
    save(path, actor, critic, tables)
    loaded, loaded_critic, _, _ = load(path)
    assert loaded.cfg.component_norm is True
    assert torch.equal(actor(batch), loaded(batch))
    assert torch.equal(critic(batch), loaded_critic(batch))
    assert torch.isfinite(actor(batch)[batch["omask"]]).all()


def test_options_context_scope_normalizes_only_option_and_context_parts(tmp_path):
    tables, cfg, record = fixture()
    actor, critic = models(cfg, tables, True, scope="options_context")
    expected = set(OPTION_COMPONENTS) | {"context_index", "context_value", "sentinel", "context_role", "context_template"}
    assert set(actor.component_norms) == expected
    assert set(critic.encoder.component_norms) == expected - set(OPTION_COMPONENTS)
    batch = stack([parse(record, tables)])
    path = tmp_path / "scoped.pt"
    save(path, actor, critic, tables)
    loaded, loaded_critic, _, _ = load(path)
    assert loaded.cfg.component_norm_scope == "options_context"
    assert torch.equal(actor(batch), loaded(batch))
    assert torch.equal(critic(batch), loaded_critic(batch))
