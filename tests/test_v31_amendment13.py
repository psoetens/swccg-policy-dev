import copy

import torch

from test_efficient_model import fixture
from swccg_policy.efficient import EfficientCritic, EfficientPolicy
from swccg_policy.ppo_step import PPOConfig, policy_loss
from swccg_policy.v4_checkpoint import load, save
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_native import export_ts


def actor_pair(cap):
    tables, cfg, record = fixture()
    torch.manual_seed(5)
    plain = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    capped_cfg = copy.deepcopy(cfg)
    capped_cfg.logit_cap = cap
    capped = EfficientPolicy(capped_cfg, tables).eval()
    capped.load_state_dict(plain.state_dict())
    return tables, plain, capped, stack([parse(record, tables)])


def test_cap_off_is_the_default_and_old_checkpoints_load_uncapped(tmp_path):
    tables, cfg, record = fixture()
    assert cfg.logit_cap == 0.0
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    path = tmp_path / "old.pt"
    save(path, actor, critic, tables)
    payload = torch.load(path, weights_only=False)
    for config in (payload["cfg"], payload["critic_cfg"]):
        del config["logit_cap"]
    loaded, _, _, _ = load(None, payload=payload)
    batch = stack([parse(record, tables)])
    assert loaded.cfg.logit_cap == 0.0
    assert torch.equal(actor(batch), loaded(batch))


def test_soft_cap_bounds_valid_logits_keeps_masked_and_preserves_order():
    _, plain, capped, batch = actor_pair(0.05)
    with torch.no_grad():
        raw = plain(batch)
        out = capped(batch)
    valid = batch["omask"]
    assert torch.isinf(out[~valid]).all() and (out[~valid] < 0).all()
    assert (out[valid].abs() < 0.05).all()
    assert torch.allclose(out[valid], 0.05 * torch.tanh(raw[valid] / 0.05))
    assert torch.equal(out[valid].argsort(), raw[valid].argsort())
    assert torch.isclose(capped.raw_max_abs_logit, raw[valid].abs().max())


def test_large_cap_is_identity_within_tolerance():
    _, plain, capped, batch = actor_pair(1e6)
    with torch.no_grad():
        assert torch.allclose(plain(batch)[batch["omask"]], capped(batch)[batch["omask"]], atol=1e-5)


def test_traced_export_applies_the_cap(tmp_path):
    _, _, capped, batch = actor_pair(0.05)
    traced = export_ts(capped, tmp_path / "capped.ts")
    names = ("tok", "tokf", "tmask", "opt", "omask")
    head = torch.stack([batch["dtype"], batch["qtmpl"], batch["qbp"], batch["side"]], 1)
    head = torch.cat([head, batch["glob"]], 1).int()
    with torch.no_grad():
        out = traced(*(batch[name] for name in names), head, batch["kgrid"])
        eager = capped(batch)
    valid = batch["omask"]
    assert (out[valid].abs() < 0.05).all()
    assert torch.allclose(out[valid], eager[valid], atol=1e-5)


def test_entropy_coefficient_scales_the_bonus():
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    batch = {"omask": torch.ones(1, 3, dtype=torch.bool), "opt": torch.zeros(1, 3, 6, dtype=torch.int32)}
    sequences = torch.zeros(1, 1, dtype=torch.long)
    lengths = torch.ones(1, dtype=torch.long)
    excluded = torch.zeros(1, 3, dtype=torch.bool)
    old = torch.log_softmax(logits, 1)[:, 0]
    advantage = torch.zeros(1)
    losses = {coef: policy_loss(logits, batch, sequences, lengths, excluded, old, advantage,
                                PPOConfig(entropy=coef))[0] for coef in (0.003, 0.01)}
    _, stats = policy_loss(logits, batch, sequences, lengths, excluded, old, advantage, PPOConfig())
    assert PPOConfig().entropy == 0.003
    assert torch.isclose(losses[0.01] - losses[0.003], -(0.01 - 0.003) * stats["entropy"])
