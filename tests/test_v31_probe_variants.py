"""Lossless scope grouping and reference routing in the isolated D1c prototype."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v31_probe_variants import per_scope_probe, validate_per_scope


def observation():
    grid = np.zeros((1, 17, 3, 3), np.uint8)
    grid[0, 2, 0] = [0, 0, 1]
    grid[0, 2, 1] = [1, 1, 1]
    return {"tok": np.array([[2, 4, 0, 1, 1, 0, 0, 0], [0, 5, 0, 1, 0, 0, 1, 0],
                              [0, 6, 0, 1, 0, 0, 1, 0], [8, 0, 0, 1, 3, 2, 1, 0]], np.int32),
            "tokf": np.array([[0] * 4, [0] * 4, [0] * 4, [3, 0, 0, 0]], np.float32),
            "opt": np.array([[1, 0, 2, 5, 0, 0], [1, 0, 3, 6, 0, 0]], np.int32),
            "kgrid": grid.reshape(1, 153), "schema_version": 4}


def test_scope_split_preserves_zero_assertions_union_and_routes():
    item = observation()
    original = copy.deepcopy(item)
    split = per_scope_probe(item)
    assert split["tok"][:2, 7].tolist() == [1, 2]
    np.testing.assert_array_equal(split["kgrid"].sum(0), item["kgrid"][0])
    assert split["tok"][-1, 5] == 3
    assert split["tokf"][-1, 0] == 4
    assert split["opt"][:, 2].tolist() == [3, 4]
    for key in ("tok", "tokf", "opt", "kgrid"):
        np.testing.assert_array_equal(item[key], original[key])


def test_identical_scopes_merge_and_actor_rejects_private():
    item = observation()
    grid = item["kgrid"].reshape(1, 17, 3, 3)
    grid[:, :, 1] = grid[:, :, 0]
    merged = per_scope_probe(item)
    assert merged["tok"][0, 7] == 3
    np.testing.assert_array_equal(merged["kgrid"], item["kgrid"])
    grid[:, :, 2] = grid[:, :, 0]
    with pytest.raises(ValueError, match="actor-private"):
        per_scope_probe(item)
    item["privileged"] = True
    critic = per_scope_probe(item)
    assert critic["tok"][0, 7] == 7
    np.testing.assert_array_equal(critic["kgrid"], item["kgrid"])


def test_reject_unmasked_grid_and_overlapping_groups():
    split = per_scope_probe(observation())
    split["kgrid"][0].reshape(17, 3, 3)[2, 1] = [1, 1, 1]
    with pytest.raises(ValueError, match="entirely zero"):
        validate_per_scope(split)
    split = per_scope_probe(observation())
    split["tok"][1, 7] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        validate_per_scope(split)
