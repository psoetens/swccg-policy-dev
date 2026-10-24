"""Plan critic-noise E1: the noise-scale estimator recovers a known B_noise = tr(S) / |G|^2 from synthetic per-decision
gradients, its bootstrap interval covers it, and the E1 verdict follows the pre-registered rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from pg_noise import E1_THRESHOLD, b_noise, e1_verdict, noise_scale  # noqa: E402

DIM = 256
TRUE_B = 2000.0  # tr(S) / |G|^2


def synthetic_gram(size, k, seed):
    """Gram matrix of K minibatch means of SIZE per-decision gradients g_i = G + noise (anisotropic S)."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=DIM)
    scales = np.linspace(0.5, 2.0, DIM)  # per-dimension noise sd before scaling
    scales *= np.sqrt(TRUE_B * signal @ signal / (scales ** 2).sum())  # tr(S) = TRUE_B |G|^2
    means = np.stack([(signal + rng.normal(size=(size, DIM)) * scales).mean(0) for _ in range(k)])
    return means @ means.T


@pytest.mark.parametrize('size', [256, 1024])
def test_noise_scale_recovers_a_known_b_noise_and_the_interval_covers_it(size):
    cell = noise_scale(synthetic_gram(size, 24, seed=size), size)
    assert cell['c'] == pytest.approx(1 / (1 + TRUE_B / size), rel=0.2)
    assert cell['b_noise'] == pytest.approx(TRUE_B, rel=0.25)
    assert cell['b_noise_dot'] == pytest.approx(TRUE_B, rel=0.25)
    assert cell['b_noise_lo'] <= TRUE_B <= cell['b_noise_hi']
    assert cell['c_lo'] <= cell['c'] <= cell['c_hi']


def test_interval_coverage_over_repeats():
    covered = [(cell := noise_scale(synthetic_gram(512, 24, seed), 512, resamples=500))['b_noise_lo'] <= TRUE_B
               <= cell['b_noise_hi'] for seed in range(40)]
    assert np.mean(covered) >= 0.8


def test_b_noise_without_signal_is_infinite():
    assert b_noise(0.0, 64) == float('inf') and b_noise(-0.01, 64) == float('inf')
    assert b_noise(0.5, 64) == pytest.approx(64)


def cell(b, lo):
    c = 1 / (1 + b / 4096)
    return {'b_noise': b, 'b_noise_lo': lo, 'c': c}


def test_e1_verdict_rules():
    holds = e1_verdict({1024: cell(20000, 9000), 4096: cell(16000, 13000)},
                       {1024: cell(30000, 1), 4096: cell(24000, 1)})
    assert holds['E1'] == 'holds' and holds['consistent_within_2x'] and not holds['holds_strict']
    assert holds['critic_gain'][4096] == pytest.approx(cell(16000, 0)['c'] / cell(24000, 0)['c'])
    low = e1_verdict({1024: cell(14000, 11000), 4096: cell(14000, 11900)})
    assert low['E1'] == 'fails' and not low['lower_bound_at_least_threshold']
    inconsistent = e1_verdict({1024: cell(60000, 40000), 4096: cell(20000, 15000)})
    assert inconsistent['E1'] == 'fails' and not inconsistent['consistent_within_2x']
    assert e1_verdict({1024: cell(60000, 40000), 4096: cell(40000, 30000)})['holds_strict']
    assert e1_verdict({256: cell(1, 1)})['E1'] == 'unmeasured'
    assert E1_THRESHOLD == 12000
