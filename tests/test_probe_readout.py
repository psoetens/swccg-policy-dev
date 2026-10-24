"""Plan probe-economics: decision kinds, the cross-fitted regret (a lower bound, no winner's curse), the intra-option
correlation, and the go / no-go arithmetic."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_readout as pr  # noqa: E402


def test_kinds():
    assert pr.kind_of('Choose Draw action or Pass', 'CARD_ACTION_CHOICE') == ('act-or-pass', 'draw')
    assert pr.kind_of('Choose weapons segment action to play or Pass', 'CARD_ACTION_CHOICE') == ('act-or-pass', 'weapons')
    where = "Choose where to deploy <div class='cardHint' value='1_2'>Luke</div>, or click 'Done' to cancel"
    assert pr.kind_of(where, 'CARD_SELECTION') == ('deploy-where', None)
    assert pr.kind_of("Choose where to move [card] using landspeed", 'CARD_SELECTION') == ('card-selection', None)
    assert pr.kind_of('Choose amount of Force to activate', 'INTEGER') == ('other', None)


def outcomes(rates, n=24):
    """Per option: n rollouts with wins exactly at the given rate on both halves."""
    out = []
    for rate in rates:
        wins = int(round(rate * n))
        half = [1] * wins + [0] * (n - wins)
        out.append([(2 * i, w) for i, w in enumerate(half)] + [(2 * i + 1, w) for i, w in enumerate(half)])
    return out


def test_folds_regret_and_top1():
    position = {'pi': [0.25, 0.75], 'outcomes': outcomes([0.75, 0.25]), 'pass': [0, 1]}
    units = pr.folds(position)
    assert len(units) == 2
    for u in units:
        assert u['gap'] == pytest.approx(0.5)
        assert u['regret'] == pytest.approx(0.75 - (0.25 * 0.75 + 0.75 * 0.25))
        assert u['top1'] == 0 and u['best_is_pass'] is False
    perfect = pr.folds({'pi': [0.9, 0.1], 'outcomes': outcomes([0.75, 0.25]), 'pass': [0, 1]})
    assert all(u['top1'] == 1 for u in perfect)


def test_icc_shared_and_independent_errors():
    rng = np.random.default_rng(0)
    shared = [np.full(24, rng.normal()) + rng.normal(0, 0.01, 24) for _ in range(200)]
    assert pr.icc(shared) > 0.95
    independent = [rng.normal(0, 1, 24) for _ in range(200)]
    assert abs(pr.icc(independent)) < 0.05


def test_snr_and_ppo_baseline():
    assert pr.snr(0.5) == pytest.approx(1 / 3)
    assert pr.snr(None) is None
    assert pr.snr(pr.PPO_R) * pr.PPO_DECISIONS == pytest.approx(0.9394, abs=1e-3)


def test_signal_recovers_perfect_estimates():
    truth = {0: {'options': [0, 1, 2], 'pi': {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 'a_star': {0: 0.1, 1: 0.0, 2: -0.1}},
             1: {'options': [0, 1], 'pi': {0: 0.5, 1: 0.5}, 'a_star': {0: -0.05, 1: 0.05}}}
    estimates = {0: {0: 5.1, 1: 5.0, 2: 4.9}, 1: {0: -2.05, 1: -1.95}}  # offsets per position cancel (pi-centred)
    assert pr.signal(estimates, truth, [0, 1], 1.0) == pytest.approx(1.0)
