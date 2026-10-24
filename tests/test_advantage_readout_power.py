"""Plan critic-noise stage 1b (E2, BL-52): the powered readout's arithmetic on synthetic data with a known true
correlation between the critic advantage and the true advantage and known rollout (binomial) measurement noise."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import advantage_readout as ar  # noqa: E402


def synthetic(rho, positions=3000, rollouts=48, spread=0.1, seed=7):
    """Positions with 2-6 options: true win rates 0.5 + base + delta, A* the pi-weighted advantage, and a critic
    advantage with correlation rho to A* over all pairs; outcomes Bernoulli(true win rate)."""
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(positions):
        n = int(rng.integers(2, 7))
        pi = rng.dirichlet(np.ones(n))
        rate = np.clip(0.5 + rng.normal(0, 0.1) + rng.normal(0, spread, n), 0.02, 0.98)
        raw.append((pi, rate, rate - pi @ rate))
    sd = np.concatenate([a for _, _, a in raw]).std()
    out = []
    for pi, rate, a_star in raw:
        adv = rho * a_star / sd + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, len(rate))
        outcomes = [[(r, int(rng.random() < p)) for r in range(rollouts)] for p in rate]
        out.append({'pi': pi.tolist(), 'adv': (adv * 0.05).tolist(), 'outcomes': outcomes, 'true_rate': rate})
    return out, rollouts


def test_disattenuated_pearson_recovers_the_true_correlation():
    for rho in (0.3, 0.6):
        positions, _ = synthetic(rho)
        stats = ar.power_stats(positions, boot=200)
        assert 0.4 < stats['reliability'] < 0.8  # the binomial noise is substantial
        assert stats['pearson'] < rho - 0.04  # attenuated
        assert abs(stats['pearson_disattenuated'] - rho) < 0.05
        low, high = stats['pearson_disattenuated_ci95']
        assert low < stats['pearson_disattenuated'] < high


def test_reliability_matches_the_known_noise():
    positions, rollouts = synthetic(0.3, spread=0.15)
    stats = ar.power_stats(positions, boot=50)
    # expected: var(true A*) / (var(true A*) + mean binomial noise variance of the estimated A*)
    true_a, noise = [], []
    for p in positions:
        pi, rate = np.array(p['pi']), p['true_rate']
        v = rate * (1 - rate) / rollouts
        true_a.append(rate - pi @ rate)
        noise.append(v * (1 - pi) ** 2 + (pi ** 2 * v).sum() - pi ** 2 * v)
    var_true = np.concatenate(true_a).var()
    expected = var_true / (var_true + np.concatenate(noise).mean())
    assert abs(stats['reliability'] - expected) < 0.05
    assert stats['split_half_r'] < stats['reliability']  # Spearman-Brown raises the half-length reliability


def test_verdict_rule():
    assert ar.verdict(0.25, 0.40, 0.7) == 'holds'
    assert ar.verdict(0.25, 0.50, 0.7) == 'fails'  # upper bound above 0.45
    assert ar.verdict(0.35, 0.40, 0.7) == 'fails'
    assert ar.verdict(0.10, 0.20, 0.45) == 'unmeasured'  # reliability below 0.5
    assert ar.verdict(None, None, 0.9) == 'unmeasured'


def test_power_stats_verdicts_on_synthetic_critics():
    weak, _ = synthetic(0.0, positions=2000, seed=3)
    assert ar.power_stats(weak, boot=300)['verdict'] == 'holds'
    strong, _ = synthetic(0.8, positions=2000, seed=4)
    assert ar.power_stats(strong, boot=300)['verdict'] == 'fails'
    noisy, _ = synthetic(0.0, positions=2000, rollouts=4, spread=0.03, seed=5)
    assert ar.power_stats(noisy, boot=100)['verdict'] == 'unmeasured'


def test_true_advantages_and_missing_options():
    keep, a_star, rate = ar.true_advantages([0.5, 0.5, 0.0], [[1, 1], [0, 0], []])
    assert keep == [0, 1] and np.allclose(a_star, [0.5, -0.5]) and np.allclose(rate, [1, 0])
    # an option without a critic value or rollouts drops out; a position needs two usable options
    pairs = ar.power_pairs([{'pi': [0.5, 0.5], 'adv': [0.1, None], 'outcomes': [[(0, 1)], [(0, 0)]]}])
    assert pairs == []
    pairs = ar.power_pairs([{'pi': [0.2, 0.8], 'adv': [0.1, -0.1], 'outcomes': [[(0, 1), (1, 1)], [(0, 0), (1, 1)]]}])
    assert np.allclose(pairs[0]['a_star'], [1 - 0.6, 0.5 - 0.6])
    assert np.allclose(pairs[0]['h1'], [1 - 0.2, 0 - 0.2]) and np.allclose(pairs[0]['h2'], [0, 0])


def test_spearman_brown():
    assert ar.spearman_brown(0.5) == 2 * 0.5 / 1.5
    assert ar.spearman_brown(0.0) == 0.0
