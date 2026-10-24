"""Plan objective-learnability Amendment 1: per-step rewards in v4_rl.advantages."""
import numpy as np

from swccg_policy.v4_rl import advantages


def raw(games, values, gamma=1.0, lam=0.95):
    """Unnormalized GAE reference."""
    out, position = [], 0
    for game in games:
        count = len(game['items'])
        rewards = np.zeros(count)
        rewards[-1] = game['reward']
        for index, value in game.get('step_rewards', ()):
            rewards[index] += value
        v = values[position:position + count]
        last, adv = 0.0, np.zeros(count)
        for index in range(count - 1, -1, -1):
            following = v[index + 1] if index + 1 < count else 0.0
            last = rewards[index] + gamma * following - v[index] + gamma * lam * last
            adv[index] = last
        out.append(adv)
        position += count
    return np.concatenate(out)


def test_without_step_rewards_matches_terminal_only():
    games = [{'items': [0] * 4, 'reward': 1.0}, {'items': [0] * 3, 'reward': -1.0}]
    values = np.array([0.1, 0.2, 0.3, 0.4, -0.1, 0.0, 0.2], np.float32)
    normalized, returns = advantages(games, values)
    reference = raw(games, values)
    assert np.allclose(returns, reference + values, atol=1e-5)
    assert np.allclose(normalized, (reference - reference.mean()) / (reference.std() + 1e-8), atol=1e-5)


def test_step_reward_lands_on_its_step():
    values = np.zeros(5, np.float32)
    base = [{'items': [0] * 5, 'reward': 0.0}]
    bonus = [{'items': [0] * 5, 'reward': 0.0, 'step_rewards': [[2, 1.0]]}]
    _, r0 = advantages(base, values)
    _, r1 = advantages(bonus, values)
    # gamma = 1: every step up to and including 2 gains lambda-weighted credit; steps after it gain none.
    gain = r1 - r0
    assert np.allclose(gain[3:], 0.0)
    assert np.allclose(gain[:3], [0.95 ** 2, 0.95, 1.0], atol=1e-6)


def test_milestone_rewards_add_like_step_rewards():
    values = np.zeros(5, np.float32)
    step = [{'items': [0] * 5, 'reward': 0.0, 'step_rewards': [[1, 0.5]]}]
    milestone = [{'items': [0] * 5, 'reward': 0.0, 'milestone_rewards': [[1, 0.5]]}]
    assert np.allclose(advantages(step, values)[1], advantages(milestone, values)[1])
