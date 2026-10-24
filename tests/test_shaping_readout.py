"""Plan probe-economics stage 6: the shaping arithmetic (GAE linearity, the potential form telescopes, round changes)
and the potentials' definitions on the globals."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shaping_readout as sr  # noqa: E402
from swccg_policy.credit import gae  # noqa: E402


def test_shaping_term_is_gae_of_the_shaped_rewards():
    rng = np.random.default_rng(0)
    values, phi = rng.normal(size=12), rng.integers(-5, 6, 12).astype(float)
    rewards = np.zeros(12)
    rewards[-1] = 1.0
    for lam in (0.8, 0.95):
        plain = rewards.copy()
        plain[:-1] += np.diff(phi) * 0.02
        assert gae(values, plain, lam)[0] == pytest.approx(gae(values, rewards, lam)[0]
                                                           + 0.02 * sr.shaping_term(phi, lam, 'plain'))
        potential = plain.copy()
        potential[-1] += -phi[-1] * 0.02
        assert gae(values, potential, lam)[0] == pytest.approx(gae(values, rewards, lam)[0]
                                                               + 0.02 * sr.shaping_term(phi, lam, 'potential'))


def test_potential_form_telescopes_at_lambda_one():
    phi = [3.0, 5.0, 4.0, 9.0]
    assert sr.shaping_term(phi, 1.0, 'potential') == pytest.approx(-phi[0])
    assert sr.shaping_term(phi, 1.0, 'plain') == pytest.approx(phi[-1] - phi[0])


def test_potentials_and_round_change():
    g = [0] * 18
    g[sr.LIFE], g[sr.LIFE_OPP], g[sr.LOST], g[sr.LOST_OPP], g[sr.HAND], g[sr.HAND_OPP] = 40, 35, 6, 9, 5, 7
    assert sr.POTENTIALS['damage'](g) == 3
    assert sr.POTENTIALS['life'](g) == 5
    assert sr.POTENTIALS['life-hand'](g) == 3
    turns = [(3, 1), (3, 1), (3, 0), (4, 1), (4, 1), (5, 1)]  # (own turn number, my-turn flag) per step
    globals_ = [[t, 0, 0, flag] + [0] * 14 for t, flag in turns]
    phi = [0, 1, 2, 4, 6, 9]
    assert sr.round_change(globals_, phi, 1) == 4
    assert sr.round_change(globals_, phi, 2) == 9
    assert sr.round_change(globals_, phi, 3) == 9  # the game ended first: the last step


def _games(rng, counts=(7, 5)):
    games = []
    for count in counts:
        items = []
        for _ in range(count):
            head = np.zeros(22, np.int32)
            head[16], head[17] = rng.integers(0, 20), rng.integers(0, 20)
            items.append({'head': head})
        games.append({'items': items, 'reward': float(rng.choice([-1, 1]))})
    return games


def test_learner_shaping_matches_the_readout_and_leaves_the_critic():
    from swccg_policy import v4_rl
    rng = np.random.default_rng(1)
    games = _games(rng)
    values = rng.normal(size=12).astype(np.float32)
    base_adv, base_ret = v4_rl.advantages(games, values, lam=0.985, value_lam=0.95)
    adv, ret = v4_rl.advantages(games, values, lam=0.985, value_lam=0.95, shaping='damage', shaping_weight=0.2,
                                shaping_target='actor')
    np.testing.assert_allclose(ret, base_ret)  # the critic's targets do not see the shaping
    raw = []
    position = 0
    for game in games:  # unnormalised: GAE(0.985) + 0.2 * the readout's term, step by step
        count = len(game['items'])
        rewards = np.zeros(count)
        rewards[-1] = game['reward']
        v = values[position:position + count]
        phi = [float(it['head'][17]) - float(it['head'][16]) for it in game['items']]
        for t in range(count):
            raw.append(gae(v, rewards, 0.985)[t] + 0.2 * sr.shaping_term(phi[t:], 0.985, 'potential'))
        position += count
    raw = np.array(raw)
    np.testing.assert_allclose(adv, (raw - raw.mean()) / (raw.std() + 1e-8), rtol=1e-5, atol=1e-6)


def test_learner_default_path_unchanged():
    from swccg_policy import v4_rl
    rng = np.random.default_rng(2)
    games = _games(rng)
    values = rng.normal(size=12).astype(np.float32)
    a, r = v4_rl.advantages(games, values)
    b, s = v4_rl.advantages(games, values, shaping=None, shaping_weight=0.5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(r, s)
    with pytest.raises(ValueError):
        v4_rl.advantages(games, values, shaping='life')
