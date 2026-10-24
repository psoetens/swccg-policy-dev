import copy
from pathlib import Path
import sys

import pytest
import torch

from test_efficient_model import fixture
from swccg_policy.efficient import (COMPONENTS, CONTEXT_COMPONENTS, ENTITY_COMPONENTS, LABEL_COMPONENTS,
                                    OPTION_COMPONENTS, EfficientPolicy)
from swccg_policy.v4_data import parse, stack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from night_probe import HEAD_MODULES, freeze_to_head


def actor(scope):
    tables, cfg, record = fixture()
    cfg = copy.deepcopy(cfg)
    cfg.component_norm = True
    cfg.component_norm_scope = scope
    cfg.component_label_gain = 1.0
    return EfficientPolicy(cfg, tables).eval(), stack([parse(record, tables)])


@pytest.mark.parametrize("scope,excluded", [("all_but_labels", LABEL_COMPONENTS),
                                            ("all_but_entity", ENTITY_COMPONENTS),
                                            ("all_but_grid", ("grid", "degree"))])
def test_new_scopes_exclude_exactly_their_parts(scope, excluded):
    model, batch = actor(scope)
    expected = (set(COMPONENTS) | set(OPTION_COMPONENTS)) - set(excluded)
    assert set(model.component_norms) == expected
    assert torch.isfinite(model(batch)[batch["omask"]]).all()


def test_existing_scopes_are_unchanged():
    everything, _ = actor("all")
    assert set(everything.component_norms) == set(COMPONENTS) | set(OPTION_COMPONENTS)
    scoped, _ = actor("options_context")
    assert set(scoped.component_norms) == set(OPTION_COMPONENTS) | set(CONTEXT_COMPONENTS)
    with pytest.raises(ValueError):
        actor("nonsense")


def test_head_freezing_trains_only_the_option_read_out():
    model, batch = actor("all")
    freeze_to_head(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    logits = model(batch)
    logits[batch["omask"]].sum().backward()
    optimizer.step()
    changed = {name.split(".")[0] for name, parameter in model.named_parameters()
               if not torch.equal(parameter.detach(), before[name])}
    assert changed and changed <= set(HEAD_MODULES)


def test_options_context_question_scope_adds_only_question_parts():
    from swccg_policy.efficient import QUESTION_COMPONENTS
    model, batch = actor("options_context_question")
    assert set(model.component_norms) == set(OPTION_COMPONENTS) | set(CONTEXT_COMPONENTS) | set(QUESTION_COMPONENTS)
    assert torch.isfinite(model(batch)[batch["omask"]]).all()


def test_rl_cosine_schedule_endpoints_and_constant_default():
    from swccg_policy.v4_rl import scheduled_lr
    assert scheduled_lr(5e-5, "constant", 100, 200) == 5e-5
    assert scheduled_lr(5e-5, "cosine", 0, 200) == pytest.approx(5e-5)
    assert scheduled_lr(5e-5, "cosine", 100, 200) == pytest.approx(2.5e-5)
    assert scheduled_lr(5e-5, "cosine", 300, 200) == pytest.approx(0.0)
