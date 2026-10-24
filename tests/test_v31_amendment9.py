import copy

import numpy as np
import torch

from test_efficient_model import fixture
from swccg_policy.efficient import EfficientPolicy, EfficientCritic
from swccg_policy.soft_vocabulary import SoftVocabulary
from swccg_policy.text_priors import TextPriorEmbedding
from swccg_policy.train_step import TrainingConfig, TrainingStep, probe_config, probe_loss
from swccg_policy.v4_checkpoint import load, save
from swccg_policy.v4_data import parse, stack


def test_identity_is_keyed_reproducible_and_growth_preserves_trained_rows():
    embed = lambda texts: np.ones((len(texts), 768), np.float16)
    old = SoftVocabulary.build({"q": ["first", "second"]}, embed, "seed")
    new = SoftVocabulary.build({"q": ["third"]}, embed, "growth", old)

    def module(vocab, family="q", seed=71):
        return TextPriorEmbedding(vocab.priors["q"], 256, keys=vocab.payload["q"], family=family,
                                  seed=seed, prior_scale=33.0, identity_std=1.0)

    original = module(old)
    fresh = module(new)
    assert torch.equal(original.residual.weight, fresh.residual.weight[:514])
    assert not torch.equal(original.residual.weight, module(old, family="context").residual.weight)
    assert not torch.equal(original.residual.weight[512:], module(old, seed=72).residual.weight[512:])
    assert 0.9 < float(original.residual.weight.std().detach()) < 1.1
    with torch.no_grad():
        original.residual.weight.add_(0.123)
    before = original.table().detach().clone()
    original.grow(new.priors["q"], new.payload["q"])
    assert torch.equal(before, original.table()[:514])
    assert torch.equal(original.residual.weight[514:], fresh.residual.weight[514:])
    # Identical priors still have distinguishable question identities.
    assert not torch.equal(original.table()[512], original.table()[513])


def test_template_config_loads_saved_and_legacy_semantics(tmp_path):
    tables, cfg, record = fixture()
    for legacy in (False, True):
        cfg.template_prior_scale = 1.0 if legacy else 33.0
        cfg.template_identity_std = 0.0 if legacy else 1.0
        cfg.template_identity_seed = 77
        actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
        critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
        path = tmp_path / f"{legacy}.pt"
        save(path, actor, critic, tables)
        payload = torch.load(path, weights_only=False)
        if legacy:
            for config in (payload["cfg"], payload["critic_cfg"]):
                for key in ("template_prior_scale", "template_identity_std", "template_identity_seed"):
                    del config[key]
        loaded, loaded_critic, _, _ = load(None, payload=payload)
        batch = stack([parse(record, tables)])
        assert torch.equal(actor(batch), loaded(batch))
        assert torch.equal(critic(batch), loaded_critic(batch))
        assert loaded.cfg.template_prior_scale == cfg.template_prior_scale
        assert loaded.cfg.template_identity_std == cfg.template_identity_std


def test_shared_step_clips_matches_reference_and_probe_smoothing():
    torch.manual_seed(83)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    config = probe_config()
    training = TrainingStep(model, config)
    optimizer = torch.optim.Adam(reference.parameters(), lr=config.lr)
    batch = torch.randn(12, 3) * 100
    labels = torch.arange(12) % 2
    loss = torch.nn.functional.cross_entropy(reference(batch), labels, label_smoothing=0.05)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    optimizer.step()
    result = training(batch, probe_loss(labels))
    assert norm > 1.0
    assert result["grad_norm"] > 1.0
    assert torch.equal(result["loss"], loss.detach())
    for expected, actual in zip(reference.parameters(), model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert TrainingConfig().label_smoothing == 0.0
    assert config.label_smoothing == 0.05
