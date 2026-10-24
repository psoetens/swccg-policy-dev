"""Plan links-round2: the discount per policy step (advantages(gamma=...)) and its stage plumbing."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swccg_policy.v4_rl import advantages

ROOT = Path(__file__).resolve().parents[1]


def test_gamma_one_is_the_previous_behaviour():
    games = [{'items': [0] * 4, 'reward': 1.0}, {'items': [0] * 3, 'reward': -1.0}]
    values = np.array([0.1, 0.2, 0.3, 0.4, -0.1, 0.0, 0.2], np.float32)
    a1, r1 = advantages(games, values)
    a2, r2 = advantages(games, values, gamma=1.0)
    assert np.array_equal(a1, a2) and np.array_equal(r1, r2)


def test_gamma_discounts_the_terminal_reward_per_step():
    games = [{'items': [0] * 5, 'reward': 1.0}]
    values = np.zeros(5, np.float32)
    _, returns = advantages(games, values, gamma=0.9, lam=1.0)
    assert np.allclose(returns, [0.9 ** 4, 0.9 ** 3, 0.9 ** 2, 0.9, 1.0])
    _, returns = advantages(games, values, gamma=0.9995, lam=1.0)
    assert returns[0] == pytest.approx(0.9995 ** 4)


def test_stage_passes_gamma_and_pair_weights(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import v31_stage
    source = (ROOT / 'scripts/v31_stage.py').read_text()
    assert "'--gamma', str(args.gamma)" in source and '-Dtraingen.pairWeights=' in source
    weights = tmp_path / 'w.tsv'
    weights.write_text('a.xml\tx.xml\t0.25\n')
    args = SimpleNamespace(seed=1, pool='p.txt', opponent_pool=None, threads=2, opponents='advanced', agent_side='alternate',
                           tag='t', rollout_games=4, pair_weights=str(weights), deck_weights=str(weights))
    work = tmp_path / 'work'
    (work / 'bundle').mkdir(parents=True)
    (work / 'bundle' / 'vocab.json').write_text('{}')
    command = v31_stage.rollout_command(args, 4, work, tmp_path, 0)
    assert f'-Dtraingen.pairWeights={weights.resolve()}' in command
    assert f'-Dtraingen.deckWeights={weights.resolve()}' in command
    assert command.index(f'-Dtraingen.pairWeights={weights.resolve()}') < command.index('com.gempukku.swccgo.bench.TrainGen')
