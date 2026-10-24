import numpy as np
import pytest

from swccg_policy.v4_parity import greedy_match


def test_only_python_top_two_qualify():
    reference = [1e-7, 0, -1e-7]
    assert greedy_match(reference, [0, 1, -1])["used_near_tie"]
    assert not greedy_match(reference, [0, -1, 1])["matched"]
    assert greedy_match(reference, [1, 0, -1])["strict"]


def test_threshold_is_strict_and_uses_fp32():
    threshold = np.float32(1e-6)
    # This FP32 number lies just below the mathematical threshold.
    assert greedy_match([threshold, 0], [0, 1])["matched"]
    above = np.nextafter(threshold, np.float32(np.inf))
    assert not greedy_match([above, 0], [0, 1])["matched"]
    assert not greedy_match([2e-6, 0], [0, 1])["matched"]


def test_multiple_ties_do_not_expand_the_exception():
    assert greedy_match([0, 0, 0], [0, 1, 0])["used_near_tie"]
    assert not greedy_match([0, 0, 0], [0, 0, 1])["matched"]
    assert greedy_match([0], [1])["strict"]
    with pytest.raises(ValueError):
        greedy_match([np.nan], [0])
