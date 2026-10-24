"""Plan turn-credit: turn anchors from the decision heads, the turn-anchored advantages (swccg_policy.credit) and the
learner's credit='turn' path (value targets unchanged)."""
import numpy as np
import pytest

from swccg_policy.credit import gae, step_rewards, turn_anchors, turn_credit
from swccg_policy.v4_rl import advantages


def heads(turns):
    """Decision heads from (my turn number, my-turn flag) pairs (other columns zero)."""
    out = []
    for number, mine in turns:
        head = np.zeros(22, np.int32)
        head[4], head[7] = number, mine
        out.append(head)
    return out


# One seat's decisions: setup (turn 0, not mine), own turn 1 (3 decisions), opponent turn (2 responses),
# own turn 2 (2 decisions), opponent turn (1), own turn 3 (1).
TURNS = [(0, 0), (1, 1), (1, 1), (1, 1), (1, 0), (1, 0), (2, 1), (2, 1), (2, 0), (3, 1)]


def test_anchors_are_the_first_decisions_of_own_turns():
    assert turn_anchors(heads(TURNS)) == [1, 6, 9]
    # an own turn without a new number (e.g. a second activation prompt) is not a new anchor
    assert turn_anchors(heads([(1, 1), (1, 0), (1, 1), (2, 1)])) == [0, 3]


def toy(seed=3):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(TURNS))
    rewards = np.zeros(len(TURNS))
    rewards[4], rewards[-1] = 0.25, 1.0  # a Dojo milestone and the result
    return values, rewards, turn_anchors(heads(TURNS))


def test_full_horizon_is_monte_carlo():
    values, rewards, anchors = toy()
    monte_carlo = np.cumsum(rewards[::-1])[::-1] - values
    np.testing.assert_allclose(turn_credit(values, rewards, anchors, round_lambda=1.0), monte_carlo)
    np.testing.assert_allclose(turn_credit(values, rewards, anchors, rounds=10), monte_carlo)
    np.testing.assert_allclose(gae(values, rewards, 1.0), monte_carlo)


def test_one_round_is_round_lambda_zero_and_ends_at_the_next_anchor():
    values, rewards, anchors = toy()
    one = turn_credit(values, rewards, anchors, rounds=1)
    np.testing.assert_allclose(turn_credit(values, rewards, anchors, round_lambda=0.0), one)
    # decision 2 (inside own turn 1): rewards 2..5 + V(anchor 6) - V(2); the values in between cancel
    assert one[2] == pytest.approx(rewards[2:6].sum() + values[6] - values[2])
    # an anchor's own round ends at the next anchor
    assert one[6] == pytest.approx(rewards[6:9].sum() + values[9] - values[6])
    # after the last anchor: the rest of the game
    assert one[9] == pytest.approx(rewards[9] - values[9])
    # before the first anchor: up to it
    assert one[0] == pytest.approx(rewards[0] + values[1] - values[0])


def test_two_rounds_and_round_lambda_weights():
    values, rewards, anchors = toy()
    two = turn_credit(values, rewards, anchors, rounds=2)
    assert two[2] == pytest.approx(rewards[2:9].sum() + values[9] - values[2])
    half = turn_credit(values, rewards, anchors, round_lambda=0.5)
    first = rewards[2:6].sum() + values[6] - values[2]
    second = rewards[6:9].sum() + values[9] - values[6]
    third = rewards[9:].sum() - values[9]
    assert half[2] == pytest.approx(first + 0.5 * second + 0.25 * third)


def test_intermediate_values_do_not_enter_as_targets():
    """Changing the critic's values between a decision and its next anchor leaves its one-round advantage unchanged."""
    values, rewards, anchors = toy()
    before = turn_credit(values, rewards, anchors, rounds=1)
    changed = values.copy()
    changed[3:6] += 5.0
    after = turn_credit(changed, rewards, anchors, rounds=1)
    assert after[2] == pytest.approx(before[2])


def test_unbiased_with_the_true_value_function():
    """A chain where each step adds +-1 luck and the true value is the expected result: every scheme's mean advantage
    is the true advantage (0 here) when V is the true value."""
    rng = np.random.default_rng(7)
    count, games = len(TURNS), 20000
    anchors = turn_anchors(heads(TURNS))
    totals = {name: 0.0 for name in ('gae95', 'r1', 'r2', 'rl05')}
    for _ in range(games):
        luck = rng.choice([-1.0, 1.0], size=count) * 0.1
        rewards = np.zeros(count)
        rewards[-1] = luck.sum()
        values = np.concatenate([[0.0], np.cumsum(luck)[:-1]])  # E[result | luck so far]
        totals['gae95'] += gae(values, rewards, 0.95)[2]
        totals['r1'] += turn_credit(values, rewards, anchors, rounds=1)[2]
        totals['r2'] += turn_credit(values, rewards, anchors, rounds=2)[2]
        totals['rl05'] += turn_credit(values, rewards, anchors, round_lambda=0.5)[2]
    for name, total in totals.items():
        assert abs(total / games) < 0.01, name


def test_learner_turn_credit_keeps_the_value_targets():
    values, rewards, _ = toy()
    items = [{'head': head} for head in heads(TURNS)]
    game = {'items': items, 'reward': 1.0, 'dojo_rewards': [[4, 0.25, 'x', 'y']]}
    values = values.astype(np.float32)
    decision_adv, decision_returns = advantages([game], values, lam=0.95)
    turn_adv, turn_returns = advantages([game], values, lam=0.95, credit='turn', rounds=1)
    np.testing.assert_allclose(turn_returns, decision_returns)
    expected = turn_credit(values, step_rewards(game, len(TURNS)), turn_anchors(heads(TURNS)), rounds=1)
    np.testing.assert_allclose(turn_adv, (expected - expected.mean()) / (expected.std() + 1e-8), rtol=1e-5, atol=1e-6)
    # the default path is unchanged: GAE over the same rewards
    raw = gae(values, step_rewards(game, len(TURNS)), 0.95)
    np.testing.assert_allclose(decision_adv, (raw - raw.mean()) / (raw.std() + 1e-8), rtol=1e-5, atol=1e-6)


def test_learner_rejects_bad_settings():
    game = {'items': [{'head': head} for head in heads(TURNS)], 'reward': 1.0}
    values = np.zeros(len(TURNS), np.float32)
    with pytest.raises(ValueError):
        advantages([game], values, credit='turn', gamma=0.999, rounds=1)
    with pytest.raises(ValueError):
        advantages([game], values, credit='turn')
    with pytest.raises(ValueError):
        advantages([game], values, credit='rounds')
