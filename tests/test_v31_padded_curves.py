from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_padded_curves import clear_cache, threshold_history
from v31_padded_control import decode_rows, grid_rows


def test_grid_control_roundtrip_and_scope_coalescing():
    defaults = (((1, 2, 0), (4, 0)), ((1, 2, 1), (4, 0)))
    grid = decode_rows([], [], 9, 1, defaults)
    grid[2, :2] = [1, 2, 1]
    for index, mask in enumerate((3, 5, 6, 7), 13):
        for scope in range(2):
            grid[index, scope, :2] = sum((grid[pile, scope, :2].astype(int)
                                         for pile in range(3) if mask & (1 << pile)), np.zeros(2, int))
    grid[16, :2] = [1, 1, 1]
    tokens, floats = grid_rows(9, 1, grid.tobytes(), defaults, False)
    np.testing.assert_array_equal(decode_rows(tokens, floats, 9, 1, defaults), grid)
    assert len(tokens) == 2
    assert tokens[:, 7].tolist() == [6, 6]
    assert tokens[:, 4].tolist() == [0, 8]


def test_unrecoverable_default_is_reported_without_fabricating_assertion():
    grid = np.zeros((17, 3, 3), np.uint8)
    grid[4, 1] = [0, 3, 0]
    tokens, reason = grid_rows(9, 1, grid.tobytes(), (), False)
    assert tokens is None
    assert "[4, 1, 1]" in reason


def test_threshold_history_records_regression_and_budget_limit():
    curve = [{"step": step, "accuracy": {"scope": value}}
             for step, value in zip([2000, 4000, 16000, 18000], [0.5, 0.995, 0.8, 0.99], strict=True)]
    result = threshold_history(curve, "scope")
    assert result["first_step_at_99"] == 4000
    assert result["stayed_at_99"] is False
    assert result["classification"] == "passed"
    for point in curve:
        point["accuracy"]["scope"] = 0.5
    assert "representation/attention" in threshold_history(curve, "scope")["classification"]
    for point, value in zip(curve, [0.5, 0.7, 0.8, 0.9], strict=True):
        point["accuracy"]["scope"] = value
    assert "budget-limited" in threshold_history(curve, "scope")["classification"]
    assert threshold_history(curve[:2], "scope")["classification"] == "pending_final_evaluation"


def test_eval_cache_is_cleared_before_training_continues():
    from types import SimpleNamespace
    model = SimpleNamespace(_card_table="detached evaluation table")
    clear_cache(model)
    assert model._card_table is None
