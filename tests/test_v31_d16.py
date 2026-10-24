import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from test_efficient_model import fixture
from swccg_policy.efficient import (HIDDEN_ORDER_ZONES, HIDDEN_QUANTITIES, EfficientPolicy,
                                    mask_opponent_hidden)
from swccg_policy.v4_checkpoint import new_models
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_native import export_ts

ROOT = Path(__file__).resolve().parents[1]
# collect-test (new activation rule) with the tables of its start checkpoint; the A13 trajectories were pruned 2026-10-09.
TRAJECTORIES = ROOT / "runs/dojo/collect-test/trajectories"
CHECKPOINT = ROOT / "runs/dojo/focus-s2/final.pt"
A10C = {"component_norm": True, "component_norm_scope": "options_context", "component_label_gain": 0.1}


@pytest.fixture(scope="module")
def real():
    if not TRAJECTORIES.is_dir() or not CHECKPOINT.exists():
        pytest.skip("runs/dojo/collect-test or the focus-s2 checkpoint not present")
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import unpack_tables
    from swccg_policy.v4_trajectory import TrajectoryReader
    tables = unpack_tables(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["tables"])
    reader = TrajectoryReader(tables, [])
    items = [item for path in trajectory_files(TRAJECTORIES)[:1] for game in reader.load(path)
             for item in game["items"]]
    items = [item for item in items if ((item["tok"][:, 0] == 2) & (item["tok"][:, 3] == 0)).any()][:6]
    masked, critic = new_models(tables, seed=906101, knowledge_mask="opp_hidden", **A10C)
    plain, _ = new_models(tables, seed=906101, **A10C)
    return tables, items, masked.eval(), plain.eval(), critic


def perturb(items, rng, hidden=True):
    out = []
    for item in items:
        item = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in item.items()}
        tok = item["tok"]
        grid = item["kgrid"].reshape(len(item["kgrid"]), 17, 3, 3)
        quantities = list(HIDDEN_QUANTITIES) if hidden else [3]
        for index in np.nonzero((tok[:, 0] == 2) & (tok[:, 3] == 0))[0]:
            row = tok[index, 4] - 1
            grid[row, quantities] = rng.integers(0, 4, size=grid[row, quantities].shape, dtype=np.uint8)
        if hidden:
            order = (tok[:, 0] == 5) & (tok[:, 3] == 0) & np.isin(tok[:, 2], HIDDEN_ORDER_ZONES)
            tok[order, 1] = rng.integers(4, 900, size=int(order.sum()))
        item["kgrid"] = grid.reshape(len(grid), 153)
        out.append(item)
    return out


def dropped_identity_perturbed(items, rng):
    out = []
    for item in items:
        batch = stack([item], "cpu")
        drop = (batch["tmask"] & ~mask_opponent_hidden(batch)["tmask"])[0].numpy()[:len(item["tok"])]
        item = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in item.items()}
        item["tok"][drop & (item["tok"][:, 0] == 2), 1] = rng.integers(4, 900, size=int((drop & (item["tok"][:, 0] == 2)).sum()))
        out.append(item)
    return out


def test_removed_information_does_not_reach_the_actor(real):
    _, items, masked, plain, _ = real
    rng = np.random.default_rng(1)
    changed = perturb(items, rng)
    with torch.no_grad():
        assert torch.equal(masked(stack(items, "cpu")), masked(stack(changed, "cpu")))
        assert not torch.equal(plain(stack(items, "cpu")), plain(stack(changed, "cpu")))


def test_identity_of_emptied_knowledge_tokens_does_not_leak(real):
    _, items, masked, _, _ = real
    changed = dropped_identity_perturbed(items, np.random.default_rng(2))
    with torch.no_grad():
        assert torch.equal(masked(stack(items, "cpu")), masked(stack(changed, "cpu")))


def test_kept_lost_pile_knowledge_still_matters(real):
    _, items, masked, _, _ = real
    changed = perturb(items, np.random.default_rng(3), hidden=False)
    with torch.no_grad():
        assert not torch.equal(masked(stack(items, "cpu")), masked(stack(changed, "cpu")))


def test_critic_keeps_the_privileged_view_and_default_is_off(real):
    tables, _, masked, plain, critic = real
    assert masked.cfg.knowledge_mask == "opp_hidden"
    assert critic.cfg.knowledge_mask == "none" and critic.encoder.cfg.knowledge_mask == "none"
    assert plain.cfg.knowledge_mask == "none"
    with pytest.raises(ValueError):
        new_models(tables, seed=1, knowledge_mask="everything")


def test_traced_export_matches_eager_under_the_mask(tmp_path):
    tables, cfg, record = fixture()
    cfg = copy.deepcopy(cfg)
    cfg.knowledge_mask = "opp_hidden"
    model = EfficientPolicy(cfg, tables).eval()
    batch = stack([parse(record, tables)])
    traced = export_ts(model, tmp_path / "masked.ts")
    head = torch.cat([torch.stack([batch["dtype"], batch["qtmpl"], batch["qbp"], batch["side"]], 1), batch["glob"]], 1).int()
    with torch.no_grad():
        out = traced(batch["tok"], batch["tokf"], batch["tmask"], batch["opt"], batch["omask"], head, batch["kgrid"])
        eager = model(batch)
    valid = batch["omask"]
    assert torch.allclose(out[valid], eager[valid], atol=1e-5)


def test_mask_constants_are_buffers_and_zero_rows_work():
    tables, cfg, record = fixture()
    cfg = copy.deepcopy(cfg)
    cfg.knowledge_mask = "opp_hidden"
    model = EfficientPolicy(cfg, tables).eval()
    names = dict(model.named_buffers())
    assert "hidden_cells" in names and "hidden_zones" in names
    assert {"hidden_cells", "hidden_zones"} <= set(model.state_dict())
    batch = stack([parse(record, tables)])
    empty = {**batch, "kgrid": batch["kgrid"][:, :0],
             "tok": batch["tok"].masked_fill((batch["tok"][..., :1] == 2).expand_as(batch["tok"]), 0)}
    out = mask_opponent_hidden(empty, model.hidden_cells, model.hidden_zones)
    assert out["kgrid"].shape[1] == 0 and torch.equal(out["tmask"], empty["tmask"])


def test_in_place_traced_update_works_under_the_mask(tmp_path):
    from swccg_policy.v4_native import update_traced
    tables, cfg, record = fixture()
    cfg = copy.deepcopy(cfg)
    cfg.knowledge_mask = "opp_hidden"
    model = EfficientPolicy(cfg, tables).eval()
    traced = export_ts(model, tmp_path / "masked.ts")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
    update_traced(traced, model)
    batch = stack([parse(record, tables)])
    head = torch.cat([torch.stack([batch["dtype"], batch["qtmpl"], batch["qbp"], batch["side"]], 1), batch["glob"]], 1).int()
    with torch.no_grad():
        out = traced(batch["tok"], batch["tokf"], batch["tmask"], batch["opt"], batch["omask"], head, batch["kgrid"])
        eager = model(batch)
    assert torch.allclose(out[batch["omask"]], eager[batch["omask"]], atol=1e-5)
