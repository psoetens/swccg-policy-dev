"""Plan critic-ceiling: the offline critic fit (schedule, convergence test, scored checkpoint, outcome targets, a toy
whose values are recovered), the stage-B verdicts and statistics, the reshuffle check, the stage-C luck statistics,
the stage-D2 position selection and the decision-text template filter."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import critic_ceiling_readout as ccr  # noqa: E402
import critic_fit  # noqa: E402
from advantage_readout import decision_texts, template_of  # noqa: E402


def test_learning_rate_constant_then_linear_decay():
    budget, decay = 120 * 60, 20 * 60
    assert critic_fit.learning_rate(3e-4, 0, budget, decay) == 3e-4
    assert critic_fit.learning_rate(3e-4, 100 * 60, budget, decay) == 3e-4
    assert critic_fit.learning_rate(3e-4, 110 * 60, budget, decay) == pytest.approx(1.5e-4)
    assert critic_fit.learning_rate(3e-4, 120 * 60, budget, decay) == 0.0


def test_convergence_window_and_scored_checkpoint():
    history = [{'minute': 0, 'mse': 0.65}, {'minute': 70, 'mse': 0.600}, {'minute': 100, 'mse': 0.597},
               {'minute': 110, 'mse': 0.58}, {'minute': 120, 'mse': 0.59}]
    assert critic_fit.converged(history, 100, 70) is True  # 0.5% better: < 1%
    history[2]['mse'] = 0.590
    assert critic_fit.converged(history, 100, 70) is False  # 1.7% better
    assert critic_fit.converged(history[:2], 100, 70) is None
    assert critic_fit.scored(history)['minute'] == 110
    assert critic_fit.scored([{'minute': 10, 'mse': 0.5}, {'minute': 20, 'mse': 0.5}])['minute'] == 10


def test_outcome_target():
    assert critic_fit.outcome_target({'w': 'ls', 's': 'ls'}) == 1.0
    assert critic_fit.outcome_target({'w': 'ds', 's': 'ls'}) == -1.0
    assert critic_fit.outcome_target({'w': '', 's': 'ls'}) == 0.0


def test_mse_on_outcomes_recovers_the_win_rate():
    """The premise of the Monte-Carlo targets: MSE on +-1 results converges to 2 p - 1 per state (the schedule and
    optimiser of critic_fit)."""
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    states = rng.integers(0, len(p), 20000)
    targets = np.where(rng.random(len(states)) < p[states], 1.0, -1.0).astype(np.float32)
    torch.manual_seed(0)
    model = torch.nn.Embedding(len(p), 1)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
    steps = 600
    for step in range(steps):
        for group in optimizer.param_groups:
            group['lr'] = critic_fit.learning_rate(3e-2, step, steps, 150)
        index = rng.integers(0, len(states), 256)
        loss = torch.nn.functional.mse_loss(model(torch.as_tensor(states[index])).squeeze(-1),
                                            torch.as_tensor(targets[index]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    np.testing.assert_allclose(model.weight.detach().numpy().ravel(), 2 * p - 1, atol=0.06)


def test_verdicts_partition():
    assert ccr.verdict(0.30, [0.02, 0.2]) == 'not learned'
    assert ccr.verdict(0.25, [0.001, 0.2]) == 'not learned'
    assert ccr.verdict(0.20, [0.01, 0.1]) == 'partly learned'
    assert ccr.verdict(0.40, [-0.01, 0.3]) == 'ceiling holds'
    assert ccr.verdict(0.10, None) == 'ceiling holds'
    assert ccr.privileged([0.01, 0.2]) == 'helps'
    assert ccr.privileged([-0.2, -0.01]) == 'hurts'
    assert ccr.privileged([-0.1, 0.1]) == 'no difference'


def test_auc_matches_pairwise_count():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 5, 60).astype(float)
    labels = rng.integers(0, 2, 60)
    wins, losses = scores[labels == 1], scores[labels == 0]
    brute = np.mean([(w > l) + 0.5 * (w == l) for w in wins for l in losses])
    assert ccr.auc(scores, labels) == pytest.approx(brute)
    assert ccr.auc([1, 2], [1, 1]) is None


def test_outcome_statistic_turn_starts():
    games = []
    rng = np.random.default_rng(2)
    for g in range(40):
        target = 1.0 if g % 2 else -1.0
        turns = np.repeat(np.arange(1, 13), 3)
        anchor = np.tile([True, False, False], 12)
        values = np.where(anchor, target * 0.5, rng.normal(0, 0.5, len(turns)))  # turn starts are informative
        games.append({'values': values, 'target': target, 'turn': turns, 'own': np.ones(len(turns), bool),
                      'anchor': anchor})
    result = ccr.outcome_statistic(games)
    assert result['auc_start'] == 1.0
    assert result['auc_other'] < 0.8
    assert result['auc_start_minus_other'] > 0
    assert result['auc_first'] == 1.0


def test_reshuffle_report_check():
    good = ('self:RESERVE_DECK:30:1:0:0 self:FORCE_PILE:5:1:0:1 self:USED_PILE:3:1:1:0 self:LOST_PILE:2:1:1:0 '
            'self:HAND:7:1:1:0 opp:RESERVE_DECK:28:1:1:0 opp:FORCE_PILE:4:1:1:0 opp:USED_PILE:0:1:1:0 '
            'opp:LOST_PILE:1:1:1:0 opp:HAND:8:1:1:0')
    lines = {0: {'reshuffle': ccr.parse_reshuffle('self', good)}}
    check = ccr.reshuffle_check(lines)
    assert check['ok'] and check['violations'] == 0
    assert check['named_piles_order_changed_share'] == 1.0
    assert check['jobs_seat_knew_a_position_share'] == 1.0
    bad = good.replace('opp:RESERVE_DECK:28:1:1:0', 'opp:RESERVE_DECK:28:1:0:0')  # 'self' must not touch the opponent
    assert not ccr.reshuffle_check({0: {'reshuffle': ccr.parse_reshuffle('self', bad)}})['ok']
    assert ccr.reshuffle_check({0: {'reshuffle': ccr.parse_reshuffle('both', bad)}})['ok']
    lost = good.replace('self:FORCE_PILE:5:1:0:1', 'self:FORCE_PILE:5:0:0:1')  # a card left the pile
    assert not ccr.reshuffle_check({0: {'reshuffle': ccr.parse_reshuffle('self', lost)}})['ok']


def test_report_lines_and_exactness(tmp_path):
    (tmp_path / 'report-gt.c0.txt').write_text(
        'job-0\treshuffle\tself\tself:RESERVE_DECK:3:1:0:0\njob-0\tbranch\t5\t2\t0\tx\t0\njob-0\tend\t90\tfinished\tls\texact\n'
        'job-1\tmismatch\t3\ta\tb\njob-1\tend\t40\tfinished\tds\tmismatch\n')
    lines = ccr.report_lines(str(tmp_path))
    assert lines[0]['end'] == ('finished', 'ls') and not lines[0]['mismatch']
    assert lines[1]['mismatch']
    assert ccr.exactness(lines, 4) == {'planned': 4, 'ended': 2, 'exact': 1, 'exact_share': 0.25}


def test_preferences_are_pi_centred():
    pi = np.array([0.7, 0.2, 0.1])
    pref = ccr.preferences(pi)
    assert float(pi @ pref) == pytest.approx(0.0, abs=1e-12)
    assert pref[0] > pref[1] > pref[2]


def test_luck_statistic_identical_truths_and_perfect_signal():
    rng = np.random.default_rng(3)
    units = []
    for _ in range(60):
        a_star = rng.normal(0, 1, 3)
        a_star -= a_star.mean()
        noise = [rng.normal(0, 0.3, 3) for _ in range(2)]
        units.append({'a_star_a': a_star, 'a_star_b': a_star.copy(), 'h1_a': a_star + noise[0],
                      'h2_a': a_star + noise[1], 'h1_b': a_star + noise[0], 'h2_b': a_star + noise[1],
                      'signals': {'S': a_star.copy()}, 'options': [0, 1, 2]})
    result = ccr.luck_statistic(units, ['S'])
    assert result['corr_a_b'] == pytest.approx(1.0)
    assert result['mean_abs_a_minus_b'] == 0.0
    assert result['S_a_minus_b'] == pytest.approx(0.0)
    assert result['S_a'] == pytest.approx(1 / math.sqrt(result['reliability_a']))


def test_turn_start_candidates_and_spread_pick():
    import random
    rows = []
    for game in range(3):
        index = 0
        for turn in range(1, 8):
            for _ in range(3):
                rows.append({'id': f'rs|{game}|{index}', 'game': game, 'index': index, 'side': 'ds', 'turn': turn,
                             'own': True})
                index += 1
            rows.append({'id': f'rs|{game}|{index}', 'game': game, 'index': index, 'side': 'ds', 'turn': turn,
                         'own': False})
            index += 1
    starts, mids = ccr.turn_start_candidates(rows, (3, 12))
    assert [r['turn'] for r in starts[0]] == [3, 4, 5, 6, 7]
    assert all(r['index'] % 4 == 0 for r in starts[0])  # the first own decision of each turn
    assert len(mids[0]) == 10 and all(r['index'] % 4 in (1, 2) for r in mids[0])
    chosen = ccr.spread_pick(starts, [0, 1, 2], 5, 2, random.Random(0))
    assert len(chosen) == 5
    per_game = {}
    for r in chosen:
        per_game.setdefault(r['game'], set()).add(r['turn'])
    assert all(len(turns) == len([r for r in chosen if r['game'] == g]) for g, turns in per_game.items())
    assert max(len([r for r in chosen if r['game'] == g]) for g in per_game) <= 2


def test_template_filter(tmp_path):
    record = tmp_path / 'g-0.tsv'
    record.write_text('0\tp\tINTEGER\tChoose amount of Force to activate \t3\n'
                      '1\tp\tCARD_ACTION_CHOICE\tChoose Draw action or Pass\t0\n'
                      '2\tp\tCARD_ACTION_CHOICE\tChoose Deploy action or Pass\t0\n')
    texts = decision_texts(record)
    templates = ['Choose amount of Force to activate', 'Choose Draw action or Pass']
    assert template_of(texts[0], templates) == templates[0]
    assert template_of(texts[1], templates) == templates[1]
    assert template_of(texts[2], templates) is None


def test_option_units_align_positions():
    row = {'id': 'rs|1|2', 'position': 0, 'pi': [0.5, 0.5], 'adv': [0.1, -0.1],
           'outcomes': [[[0, 1], [1, 1], [2, 0], [3, 1]], [[0, 0], [1, 0], [2, 1], [3, 0]]]}
    other = {**row, 'adv': [0.2, 0.0]}
    units = ccr.option_units({'X': [row], 'Y': [other, {**row, 'id': 'rs|9|9'}]})
    assert len(units) == 1 and units[0]['id'] == 'rs|1|2'
    np.testing.assert_allclose(units[0]['signals']['Y'], [0.2, 0.0])
    assert json.dumps({k: v for k, v in units[0].items() if k == 'id'})


def test_critic_ceiling_2_verdicts():
    assert not ccr.gate_passed([-0.01, 0.05]) and not ccr.gate_passed(None)
    assert ccr.gate_passed([0.001, 0.05])
    assert ccr.verdict2(False, [0.1, 0.2]) == 'no better critic produced'
    assert ccr.verdict2(True, [0.01, 0.2]) == 'critic quality transfers'
    assert ccr.verdict2(True, [-0.1, 0.09]) == 'critic quality does not transfer to options'
    assert ccr.verdict2(True, [-0.1, 0.12]) == 'inconclusive'
    assert ccr.replicates(0.14) and not ccr.replicates(0.30) and not ccr.replicates(None)


def test_pooled_rows_prefix_ids_and_preferences(tmp_path):
    row = {'id': 'rs|1|2', 'position': 0, 'pi': [0.8, 0.2], 'adv': [0.1, -0.1], 'outcomes': [[[0, 1]], [[0, 0]]]}
    (tmp_path / 'a.json').write_text(json.dumps({'D': [row]}))
    rows = ccr.pooled_rows([f'e2={tmp_path / "a.json"}', f'e3={tmp_path / "a.json"}'], pref='D')
    assert [r['id'] for r in rows['D']] == ['e2:rs|1|2', 'e3:rs|1|2']
    np.testing.assert_allclose(rows['D-pref'][0]['adv'], ccr.preferences([0.8, 0.2]))
