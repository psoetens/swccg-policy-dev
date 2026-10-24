"""Plan critic-replay: the replay cache's selection, targets, labels, FIFO window, step count and a real training step."""
from pathlib import Path

import numpy as np
import pytest
import torch

from swccg_policy.critic_replay import CriticReplay, mc_returns, selected_steps
from swccg_policy.ppo_step import PPOConfig
from swccg_policy.v4_rl import advantages

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
WARM = ROOT / 'runs/dojo/critic-v2-warm/warm.pt'


def toy_games():
    games = []
    for count, reward in ((10, 1.0), (5, -1.0), (7, 0.0)):
        tok = lambda i: {'tok': np.full((3, 8), i, np.int32)}  # noqa: E731
        games.append({'items': [None] * count, 'critic_items': [tok(i) for i in range(count)], 'reward': reward,
                      'dojo_rewards': [[2, 0.25, 'x', 'y']] if count > 5 else []})
    return games


def test_mc_returns_match_lambda_one():
    games = toy_games()
    expected = advantages(games, np.zeros(22, np.float32), lam=1.0)[1]
    np.testing.assert_allclose(mc_returns(games), expected, atol=1e-6)


def test_selection_targets_and_labels():
    games = toy_games()
    replay = CriticReplay(updates=3, stride=4)
    replay.add(games)
    items, returns, labels, first = replay.flat()
    assert selected_steps(10, 4).tolist() == [0, 4, 8]
    # 10 steps -> 0, 4, 8; 5 -> 0, 4; 7 -> 0, 4
    assert [int(item['tok'][0, 0]) for item in items] == [0, 4, 8, 0, 4, 0, 4]
    assert first.tolist() == [True, False, False, True, False, True, False]
    assert labels[:3].tolist() == [1.0] * 3 and labels[3:5].tolist() == [0.0] * 2 and np.isnan(labels[5:]).all()
    mc = mc_returns(games)
    np.testing.assert_allclose(returns, mc[[0, 4, 8, 10, 14, 15, 19]])


def test_fifo_window_and_steps():
    replay = CriticReplay(updates=2, stride=4)
    for _ in range(3):
        replay.add(toy_games())
    assert len(replay.batches) == 2 and len(replay) == 14
    assert replay.steps(12000, 1.0, 256) == 47 and replay.steps(12000, 0.5, 256) == 23
    with pytest.raises(ValueError):
        CriticReplay(updates=0, stride=4)


@pytest.mark.skipif(not torch.cuda.is_available() or not DATA.is_dir() or not WARM.exists(),
                    reason='needs a GPU, runs/dojo/collect-test and the critic-v2 warm-start')
def test_replay_trains_only_the_critic():
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_rl import critic_groups
    actor, critic, tables, _ = load(WARM, 'cuda')
    from swccg_policy.v4_trajectory import TrajectoryReader
    games = TrajectoryReader(tables, []).load(sorted(trajectory_files(DATA))[0])
    before_actor = {k: v.clone() for k, v in actor.state_dict().items()}
    optimizer = torch.optim.AdamW(critic_groups(critic, 0.01), lr=1e-4)
    for mode in ('uniform', 'once'):
        replay = CriticReplay(updates=1, stride=8, seed=1, mode=mode)  # once with W = 1: everything is due now
        replay.add(games)
        before_critic = {k: v.clone() for k, v in critic.named_parameters()}
        stats = replay.train(critic, optimizer, PPOConfig(bf16=False, win_coef=0.5), stack, steps=2, size=16)
        assert set(stats) == {'replay_vloss', 'replay_win_loss', 'replay_rows'}
        assert all(np.isfinite(v) for v in stats.values())
        assert stats['replay_rows'] == (32 if mode == 'uniform' else sum(len(range(0, len(g['items']), 8)) for g in games))
        assert any(not torch.equal(before_critic[k], v) for k, v in critic.named_parameters())
    assert all(torch.equal(before_actor[k], v) for k, v in actor.state_dict().items())


def test_once_trains_every_entry_exactly_once_within_the_window():
    replay = CriticReplay(updates=5, stride=1, seed=3, mode='once')
    seen = []
    stored = 0
    for update in range(30):
        games = toy_games()
        for game in games:  # tag every item with (update, game, step) so each training row is identifiable
            for step, item in enumerate(game['critic_items']):
                item['tag'] = (update, id(game), step)
        replay.add(games)
        stored += sum(len(game['items']) for game in games)
        items, _, _ = replay.due()
        assert all(update - item['tag'][0] < 5 for item in items)  # trained within W updates of being stored
        seen += [item['tag'] for item in items]
    pending = len(replay)
    assert len(seen) == len(set(seen))  # never twice
    assert len(seen) + pending == stored  # nothing lost: trained or still waiting
    assert pending < 5 * 22  # only the last W updates' entries can still wait


def test_once_rows_per_update_match_the_intake_on_average():
    replay = CriticReplay(updates=4, stride=2, seed=5, mode='once')
    rows = []
    for _ in range(200):
        replay.add(toy_games())
        rows.append(len(replay.due()[0]))
    intake = len(selected_steps(10, 2)) + len(selected_steps(5, 2)) + len(selected_steps(7, 2))
    assert abs(np.mean(rows[10:]) - intake) < 1.0


def test_uniform_mode_unchanged_by_default():
    assert CriticReplay(updates=2, stride=4).mode == 'uniform'
    with pytest.raises(ValueError):
        CriticReplay(updates=2, stride=4, mode='other')
