"""Plan objective-dojo end-goal A-6: Dojo rewards in the learner, the coverage gate, accounting and BL-20(a) unpacking."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from dojo_check import held_value  # noqa: E402
import dojo_coverage  # noqa: E402
from swccg_policy.efficient import unpack_location_icons  # noqa: E402
from swccg_policy.v4_rl import advantages  # noqa: E402

REWARDS = json.loads((ROOT / 'data/dojo/rewards.json').read_text())


def returns(games, values):
    return advantages(games, values)[1]


def test_signed_and_stacked_dojo_rewards_land_on_their_steps():
    values = np.zeros(5, np.float32)
    base = [{'items': [0] * 5, 'reward': 1.0}]
    dojo = [{'items': [0] * 5, 'reward': 1.0,
             'dojo_rewards': [[1, 0.25, '8_167', 'endor_control_2'], [1, 0.125, '8_167', 'ominous_rumors_on_table'],
                              [3, -0.25, '8_167', 'endor_control_2']]}]
    gain = returns(dojo, values) - returns(base, values)
    # gamma = 1, lambda = 0.95: step 1 gets +0.375 now and -0.25 two steps later; steps after 3 see nothing.
    expected = np.array([0.375 * 0.95 - 0.25 * 0.95 ** 3, 0.375 - 0.25 * 0.95 ** 2, -0.25 * 0.95, -0.25, 0.0])
    assert np.allclose(gain, expected, atol=1e-6)


def test_dojo_rewards_add_like_the_older_keys():
    values = np.zeros(4, np.float32)
    old = [{'items': [0] * 4, 'reward': 0.0, 'step_rewards': [[2, 1.0]]}]
    new = [{'items': [0] * 4, 'reward': 0.0, 'dojo_rewards': [[2, 1.0, '110_6', 'success']]}]
    assert np.allclose(returns(old, values), returns(new, values))


def test_standard_gameplay_without_dojo_rewards_is_terminal_only():
    values = np.zeros(3, np.float32)
    # Zero values: GAE(lambda=0.95) returns are the lambda-weighted terminal reward.
    assert np.allclose(returns([{'items': [0] * 3, 'reward': -1.0}], values), [-0.95 ** 2, -0.95, -1.0])


def test_held_value_replays_gain_loss_regain_and_flip_back():
    events = [
        {'item': 'x', 'm': 'a', 'ev': 'gain', 'v': 0.25}, {'item': 'x', 'm': 'a', 'ev': 'loss', 'v': -0.25},
        {'item': 'x', 'm': 'a', 'ev': 'gain', 'v': 0.2}, {'item': 'x', 'm': 'success', 'ev': 'success', 'v': 1.0},
        {'item': 'x', 'm': 'success', 'ev': 'flip_back', 'v': -1.0}, {'item': 'y', 'm': 'b', 'ev': 'gain', 'v': 0.05},
    ]
    assert held_value(events) == pytest.approx(0.25)
    assert sum(event['v'] for event in events) == pytest.approx(held_value(events))


def test_held_value_adds_repeating_events_and_refunds_flipped_states():
    """Plan dojo-three-objectives: a repeating event gains several times and keeps all of it; a "when": "flipped" state
    milestone is refunded on the flip-back together with the success."""
    events = [
        {'item': '13_73', 'm': 'target_lost', 'ev': 'gain', 'v': 0.1}, {'item': '13_73', 'm': 'target_lost', 'ev': 'gain', 'v': 0.08},
        {'item': '13_73', 'm': 'success', 'ev': 'success', 'v': 0.5},
        {'item': '13_73', 'm': 'saber_combat_won', 'ev': 'gain', 'v': 0.15},
        {'item': 'x', 'm': 'flipped_state', 'ev': 'gain', 'v': 0.2},
        {'item': '13_73', 'm': 'success', 'ev': 'flip_back', 'v': -0.5}, {'item': 'x', 'm': 'flipped_state', 'ev': 'loss', 'v': -0.2},
    ]
    assert held_value(events) == pytest.approx(0.33)
    assert sum(event['v'] for event in events) == pytest.approx(held_value(events))


def test_new_milestone_fields_are_well_formed():
    """"when": "flipped" only on objective milestones that are not necessary; "repeat" only on events."""
    for key, item in REWARDS['items'].items():
        for milestone in item['milestones']:
            if milestone.get('when', 'front') != 'front':
                assert milestone['when'] == 'flipped' and item['kind'] == 'objective' and not milestone['necessary'], key
            if milestone.get('repeat', 1) != 1:
                assert milestone['kind'] == 'event' and milestone['repeat'] > 1, key


def test_coverage_gate_passes_our_pools_and_refuses_uncovered_decks(tmp_path):
    assert dojo_coverage.missing([ROOT / 'data/schedules/eval_decipher_heldout.tsv']) == {}
    v_pool = [line for line in (ROOT / 'data/schedules/rl_pool_v4.txt').read_text().splitlines() if '\t' in line]
    schedule = tmp_path / 'v.txt'
    schedule.write_text('\n'.join(v_pool) + '\n')
    gaps = dojo_coverage.missing([schedule])
    assert gaps and all(category in dojo_coverage.CATEGORIES for _, category, _ in gaps.values())
    with pytest.raises(SystemExit, match='supply the Dojo reward for'):
        dojo_coverage.require([schedule])


def test_reward_table_shape():
    items = REWARDS['items']
    assert len({key.split('~')[0] for key in items}) == 50
    for key, item in items.items():
        total = sum(milestone['w'] for milestone in item['milestones'])
        if any(milestone.get('when') == 'flipped' for milestone in item['milestones']):
            # Plan dojo-three-objectives: where the flip only stages the payoff, it pays less and the reward goes on after it.
            assert key in ('13_73', '13_46', '9_151', '9_61') and item['kind'] == 'objective' and item['success'] < 1.0, key
            assert sum(m['w'] for m in item['milestones'] if m.get('when') != 'flipped') <= 0.5 + 1e-5, key
        elif item['kind'] == 'objective':
            # User 2026-10-13: a ladder may grow beyond 0.5 when the flip pays less by the same amount; flip + ladder stays
            # 1.5 (Carbon Chamber Testing, Set Your Course For Alderaan, Mind What You Have Learned).
            if not item['milestones']:
                assert item['success'] == 1.0, key
            else:
                assert item['success'] <= 1.0 and total >= 0.5 - 1e-5, key
                assert item['success'] + total == pytest.approx(1.5, abs=1e-5), key
        elif item['kind'] == 'epic_event':
            # User 2026-10-13: Commence Primary Ignition's added Death Star sites milestone (0.0625) is not compensated.
            if key in ('13_24', '13_65', '11_80'):
                # Plan night-pool-v3 (user 2026-10-13): their set-up milestone comes out of the success, so the item keeps 0.5.
                assert item['success'] + total == pytest.approx(0.5, abs=1e-5) and item['milestones'], key
                continue
            ladder = 0.3125 if key == '2_130' else 0.25
            assert item['success'] == 0.5 and (total == pytest.approx(ladder, abs=1e-5) or not item['milestones']), key
        else:
            assert item['success'] in (0.2, 0.5) and total == pytest.approx(0.05, abs=1e-5), key


def test_location_icons_unpack_base_16():
    packed = torch.tensor([2 + 16 * 1 + 256 * 3 + 4096 * 2, 0.0, 5.0])
    digits = unpack_location_icons(packed, torch.tensor([True, True, False])) * 60
    assert torch.equal(digits.round(), torch.tensor([[2., 1., 3., 2.], [0., 0., 0., 0.], [0., 0., 0., 0.]]))


def test_location_state_unpack_pairs_and_drains():
    from swccg_policy.efficient import unpack_location_state
    half = lambda value: round(2 * value)  # noqa: E731 - engine packing h(x) = round(2x)
    packed = torch.tensor([[half(7.5) + 512 * half(3), half(4) + 512 * half(0), 2.0, 0.0], [99.0, 99.0, 9.0, 9.0]])
    state = unpack_location_state(packed, torch.tensor([True, False])) * 60
    assert torch.allclose(state, torch.tensor([[7.5, 3.0, 4.0, 0.0, 2.0, 0.0], [0.0] * 6]))


def test_float_rows_keep_abi4_width_and_refuse_mixed_requests():
    from swccg_policy.v4_data import float_rows
    assert float_rows([[1, 2, 3, 4]]).shape == (1, 4)
    assert float_rows([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]]).shape == (2, 8)
    assert float_rows([[1, 2, 3, 4]], width=8).tolist() == [[1, 2, 3, 4, 0, 0, 0, 0]]
    with pytest.raises(ValueError):
        float_rows([[1, 2, 3, 4, 5, 6, 7, 8]], width=4)
