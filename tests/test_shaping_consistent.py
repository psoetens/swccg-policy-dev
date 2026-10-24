"""Plan shaping-consistent stage 1: damage shaping on the per-step rewards (--shaping-target both). The value targets
are the lambda-returns of the shaped rewards, the advantages GAE on them; with the shaped truth V - beta Phi as values
the shaped advantages equal the unshaped ones step by step; 'actor' keeps 6c'; the stage passes the target through."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swccg_policy import v4_rl

ROOT = Path(__file__).resolve().parents[1]


def _games(rng, lengths=(5, 7)):
    games = []
    for count in lengths:
        items = []
        for _ in range(count):
            head = np.zeros(22, np.int32)
            head[16], head[17] = rng.integers(0, 20), rng.integers(0, 20)
            items.append({'head': head})
        games.append({'items': items, 'reward': float(rng.choice([-1, 1]))})
    return games


def _phi(game):
    return np.array([float(it['head'][17]) - float(it['head'][16]) for it in game['items']])


def _raw(games, values, lam, value_lam, beta, form='potential'):
    """Reference: unnormalised GAE(lam) and lambda(value_lam)-returns of the shaped rewards, written out per step."""
    adv, ret = [], []
    position = 0
    for game in games:
        count = len(game['items'])
        phi = _phi(game)
        rewards = np.zeros(count)
        rewards[-1] = game['reward']
        for t in range(count):
            nxt = phi[t + 1] if t + 1 < count else (0.0 if form == 'potential' else phi[t])
            rewards[t] += beta * (nxt - phi[t])
        v = np.asarray(values[position:position + count], np.float64)
        for lam_, out in ((lam, adv), (value_lam, ret)):
            g, last = np.zeros(count), 0.0
            for t in range(count - 1, -1, -1):
                delta = rewards[t] + (v[t + 1] if t + 1 < count else 0.0) - v[t]
                last = delta + lam_ * last
                g[t] = last
            out.extend(g if out is adv else g + v)
        position += count
    return np.array(adv), np.array(ret)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def test_default_path_unchanged():
    rng = np.random.default_rng(3)
    games = _games(rng)
    values = rng.normal(size=12).astype(np.float32)
    a, r = v4_rl.advantages(games, values)
    for target in ('actor', 'both'):
        b, s = v4_rl.advantages(games, values, shaping=None, shaping_weight=0.5, shaping_target=target)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(r, s)
    with pytest.raises(ValueError):
        v4_rl.advantages(games, values, shaping='damage', shaping_weight=0.1, shaping_target='critic')


@pytest.mark.parametrize('form', ['potential', 'plain'])
def test_both_shapes_the_returns_and_the_advantages(form):
    rng = np.random.default_rng(4)
    games = _games(rng)
    values = rng.normal(size=12).astype(np.float32)
    adv, ret = v4_rl.advantages(games, values, lam=0.95, value_lam=0.9, shaping='damage', shaping_weight=0.05,
                                shaping_form=form)
    raw_adv, raw_ret = _raw(games, values, 0.95, 0.9, 0.05, form)
    np.testing.assert_allclose(ret, raw_ret, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(adv, _norm(raw_adv), rtol=1e-5, atol=1e-6)
    _, base_ret = v4_rl.advantages(games, values, lam=0.95, value_lam=0.9)
    assert not np.allclose(ret, base_ret)


def test_both_and_actor_give_the_same_advantages():
    # GAE is linear in the rewards: shaping the rewards adds the same lambda-weighted term 6c' added to the advantages.
    rng = np.random.default_rng(5)
    games = _games(rng)
    values = rng.normal(size=12).astype(np.float32)
    a_both, _ = v4_rl.advantages(games, values, lam=0.95, shaping='damage', shaping_weight=0.05)
    a_actor, _ = v4_rl.advantages(games, values, lam=0.95, shaping='damage', shaping_weight=0.05, shaping_target='actor')
    np.testing.assert_allclose(a_both, a_actor, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('lam', [0.95, 0.985, 1.0])
def test_neutral_with_the_shaped_truth(lam):
    # values V - beta Phi (Phi = 0 at the end): the shaped TD errors equal the unshaped ones, so do the advantages.
    rng = np.random.default_rng(6)
    games = _games(rng, lengths=(6, 4, 9))
    values = rng.normal(size=19)
    beta = 0.05
    shaped_values = values - beta * np.concatenate([_phi(g) for g in games])
    plain, plain_ret = v4_rl.advantages(games, values, lam=lam, value_lam=0.95)
    shaped, shaped_ret = v4_rl.advantages(games, shaped_values, lam=lam, value_lam=0.95, shaping='damage',
                                          shaping_weight=beta)
    np.testing.assert_allclose(shaped, plain, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(shaped_ret, plain_ret - (values - shaped_values), rtol=1e-6, atol=1e-8)


def test_stage_and_learner_pass_the_target():
    sys.path.insert(0, str(ROOT / 'scripts'))
    source = (ROOT / 'scripts/v31_stage.py').read_text()
    assert "'--shaping-target', getattr(args, 'shaping_target', 'both')" in source
    parser_source = (ROOT / 'src/swccg_policy/v4_rl.py').read_text()
    assert "'--shaping-target', choices=['actor', 'both'], default='both'" in parser_source
    import v31_stage  # noqa: F401  (the stage parses)
