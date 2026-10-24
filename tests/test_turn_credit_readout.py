"""Plan turn-credit stage A: the readout (scripts/turn_credit_readout.py) on synthetic data."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from advantage_readout import true_advantages  # noqa: E402
import turn_credit_readout as tcr  # noqa: E402


def synthetic(seed, positions=300, options=3, rollouts=24, truth_rollouts=48, noise=None):
    """E2-like truth (48 Bernoulli rollouts per option, pi-centred win rates) and stage-A samples: per rollout, scheme
    advantage = A_true(o) + Gaussian noise of the scheme's level (plus a per-position offset common to its options, as
    V(s_K) is). Returns truth, samples, the noise-free A_true per sample and per pair, and the per-scheme estimates."""
    rng = np.random.default_rng(seed)
    noise = noise or {name: 0.30 for name in tcr.NAMES}
    truth, samples, sample_true, pair_true, pair_means = {}, [], [], [], {name: [] for name in tcr.NAMES}
    per_sample = {name: [] for name in tcr.NAMES}
    for p in range(positions):
        rate = rng.uniform(0.2, 0.8, options)
        pi = rng.dirichlet(np.ones(options))
        a_true = rate - pi @ rate
        wins = [list((rng.random(truth_rollouts) < rate[o]).astype(int)) for o in range(options)]
        _, a_star, observed = true_advantages(list(pi), wins)
        halves = [dict(zip(range(options), true_advantages(list(pi), [w[parity::2] for w in wins])[1]))
                  for parity in (0, 1)]
        truth[p] = {'options': list(range(options)), 'pi': dict(enumerate(pi)), 'a_star': dict(enumerate(a_star)),
                    'rate': dict(enumerate(observed)), 'h1': halves[0], 'h2': halves[1]}
        offset = rng.normal(0, 0.1)
        for o in range(options):
            draws = {name: a_true[o] + offset + rng.normal(0, noise[name], rollouts) for name in tcr.NAMES}
            for r in range(rollouts):
                samples.append((p, o, {name: float(draws[name][r]) for name in tcr.NAMES}))
                sample_true.append(a_true[o])
                for name in tcr.NAMES:
                    per_sample[name].append(draws[name][r])
            pair_true.append(a_true[o])
            for name in tcr.NAMES:
                pair_means[name].append(draws[name].mean())
    return truth, samples, np.array(sample_true), np.array(pair_true), per_sample, pair_means


def test_step0_advantages_by_hand():
    values = [0.1, 0.2, -0.1, 0.3, 0.5]
    heads = np.zeros((5, 22), int)
    heads[:, 4] = [4, 4, 4, 5, 5]   # the seat's turn number
    heads[:, 7] = [1, 1, 0, 1, 1]   # my-turn flag: own turns start at steps 0 and 3
    out = tcr.step0_advantages(values, 1.0, heads)
    assert out['gae-1'] == pytest.approx(1.0 - 0.1)            # Monte Carlo: the result minus V(s_K)
    assert out['gae-0'] == pytest.approx(0.2 - 0.1)            # one-step TD
    assert out['tc-r1'] == pytest.approx(0.3 - 0.1)            # to the next own-turn start
    assert out['tc-r2'] == pytest.approx(1.0 - 0.1)            # two rounds reach the end here
    assert out['tc-r3'] == pytest.approx(1.0 - 0.1)
    assert out['tc-rl05'] == pytest.approx(0.2 + 0.5 * (1.0 - 0.3))
    assert out['tc-rl07'] == pytest.approx(0.2 + 0.7 * (1.0 - 0.3))
    flipped = tcr.step0_advantages(values, -1.0, heads)
    assert flipped['gae-1'] == pytest.approx(-1.0 - 0.1)       # a loss of the branching seat
    assert flipped['tc-r1'] == pytest.approx(out['tc-r1'])     # one round never reaches the result here


def test_seat_reward_is_the_branching_seats_view():
    assert tcr.seat_reward('finished', 'ls', 'ls') == 1.0
    assert tcr.seat_reward('finished', 'ds', 'ls') == -1.0
    assert tcr.seat_reward('finished', 'ds', 'ds') == 1.0
    assert tcr.seat_reward('finished', 'ls', 'ds') == -1.0
    for outcome in ('cap', 'timeout', 'loop'):
        assert tcr.seat_reward(outcome, '', 'ds') == 0.0
    assert tcr.seat_reward('finished', '', 'ls') == 0.0


def test_scheme_names_and_flags():
    assert tcr.TURN_VARIANTS == ['tc-r1', 'tc-r2', 'tc-r3', 'tc-rl05', 'tc-rl07']
    flags = {name: f for name, _, _, f in tcr.SCHEMES}
    assert flags['tc-r2'] == '--credit turn --rounds 2'
    assert flags['tc-rl05'] == '--credit turn --round-lambda 0.5'
    assert flags['tc-rl07'] == '--credit turn --round-lambda 0.7'
    assert [n for n in tcr.NAMES if n.startswith('gae')] == ['gae-0', 'gae-0.8', 'gae-0.9', 'gae-0.95', 'gae-0.985',
                                                             'gae-1']


def test_disattenuation_recovers_the_known_correlation():
    noise = {name: 0.30 for name in tcr.NAMES}
    truth, samples, sample_true, pair_true, per_sample, pair_means = synthetic(1, positions=400, noise=noise)
    result = tcr.readout(truth, samples, boot=200)
    row = result['schemes']['gae-0.95']
    target = float(np.corrcoef(per_sample['gae-0.95'], sample_true)[0, 1])       # vs the noise-free advantage
    assert 0 < result['truth_reliability'] < 0.95
    assert row['signal'] < target                                              # the noisy truth attenuates
    assert row['signal_dis'] == pytest.approx(target, abs=0.05)
    bias_target = float(np.corrcoef(pair_means['gae-0.95'], pair_true)[0, 1])
    assert row['bias_dis'] == pytest.approx(bias_target, abs=0.06)
    # Within-option sum of squares (around the option's mean of 24) = (n - 1) / n of the noise variance; total = noise
    # + the spread of the options' true advantages and the positions' offsets.
    values = np.array(per_sample['gae-0.95'])
    assert row['noise_share'] == pytest.approx(23 / 24 * 0.30 ** 2 / values.var(), abs=0.01)
    # The pi-centred variant removes the per-position offset, which is unrelated to A*: it carries more signal.
    assert row['signal_centred_dis'] > row['signal_dis']
    assert row['signal_dis_minus_reference'] == 0


def test_bootstrap_interval_covers_the_truth():
    covered = 0
    replications = 20
    for seed in range(replications):
        truth, samples, sample_true, _, per_sample, _ = synthetic(100 + seed, positions=80, rollouts=8)
        target = float(np.corrcoef(per_sample['tc-r1'], sample_true)[0, 1])
        ci = tcr.readout(truth, samples, boot=200, seed=seed)['schemes']['tc-r1']['signal_dis_ci95']
        covered += ci[0] <= target <= ci[1]
    assert covered >= 16  # nominal 95%: >= 16 of 20 (P(fewer) < 2% at 90% true coverage)


def test_choice_picks_the_two_best_turn_variants():
    noise = {name: 0.6 for name in tcr.NAMES}
    noise.update({'tc-r2': 0.2, 'tc-rl07': 0.3, 'gae-0': 0.1})  # a GAE scheme is best overall but is not a candidate
    truth, samples, *_ = synthetic(7, positions=200, noise=noise)
    result = tcr.readout(truth, samples, boot=50)
    assert [c['name'] for c in result['choice']] == ['tc-r2', 'tc-rl07']
    assert result['choice'][0]['flags'] == '--credit turn --rounds 2'
    assert result['choice'][1]['flags'] == '--credit turn --round-lambda 0.7'
    signal = {name: result['schemes'][name]['signal_dis'] for name in tcr.NAMES}
    assert signal['gae-0'] > signal['tc-r2'] > signal['tc-rl07'] > max(signal[n] for n in ('tc-r1', 'tc-r3', 'tc-rl05'))
    # The paired difference against GAE(0.95) has its interval.
    lower, upper = result['schemes']['tc-r2']['signal_dis_minus_reference_ci95']
    assert lower > 0 and upper > lower


def test_positions_need_two_scored_options():
    truth, samples, *_ = synthetic(3, positions=10, options=3, rollouts=4)
    kept = [s for s in samples if not (s[0] == 0 and s[1] > 0)]  # position 0 keeps one option only
    data = tcr.build_pairs(truth, kept)
    assert 0 not in data['positions'] and len(data['positions']) == 9
    assert len(data['sample_a_star']) == 9 * 3 * 4
    # pi-centring: per position, the pi-weighted option means of the centred scheme are 0.
    for index, p in enumerate(data['positions']):
        mask = data['pair_position'] == index
        pi = np.array([truth[p]['pi'][o] for o in range(3)])
        assert pi / pi.sum() @ data['pair|tc-r1|c'][mask] == pytest.approx(0, abs=1e-12)


def test_stage_reports(tmp_path):
    (tmp_path / 'report-tc.c0.txt').write_text(
        'job-0\tbranch\t607\t3\t0\t0\t2\n'
        'job-0\tend\t2082\tfinished\tls\texact\n'
        'job-1\tmismatch\t12\tls|X|a\tls|X|b\n'
        'job-1\tend\t900\tfinished\tds\tmismatch\n'
        'job-2\tend\t6000\tcap\t\texact\n')
    ends, lines = tcr.stage_reports(tmp_path)
    assert lines == 1
    assert ends[0] == ('exact', 'finished', 'ls')
    assert ends[1][0] == 'mismatch'
    assert ends[2] == ('exact', 'cap', '')
    assert math.isclose(tcr.seat_reward(*ends[2][1:], 'ds'), 0.0)
