"""Plan critic-diagnostics: the matchup oracle, the CSV <-> trajectory join, critic modes and the gradient split."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DATA = ROOT / 'runs/dojo/collect-test'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
GPU_DATA = pytest.mark.skipif(not torch.cuda.is_available() or not DATA.is_dir() or not CHECKPOINT.exists(),
                              reason='needs a GPU, runs/dojo/collect-test and the focus-s2 checkpoint')


def test_oracle_recovers_planted_strengths():
    import matchup_oracle
    rng = np.random.default_rng(1)
    ds, ls = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
    ai = {'policy:learner': 0.0, 'random': -2.0}
    rows = []
    for game in range(6000):
        d, l_ = rng.integers(20), rng.integers(20)
        ds_ai, ls_ai = rng.choice(list(ai)), rng.choice(list(ai))
        p = 1 / (1 + np.exp(-(ds[d] - ls[l_] + ai[ds_ai] - ai[ls_ai])))
        rows.append({'g': str(game), 'ds_deck': f'd{d}.xml', 'ls_deck': f'l{l_}.xml', 'ds_ai': ds_ai, 'ls_ai': ls_ai,
                     'ds_won': int(rng.random() < p)})
    strengths = matchup_oracle.Oracle(1.0).fit(rows).strengths()
    assert np.corrcoef([strengths[f'ds:d{i}.xml'] for i in range(20)], ds)[0, 1] > 0.9
    assert np.corrcoef([-strengths[f'ls:l{i}.xml'] for i in range(20)], ls)[0, 1] > 0.9
    assert strengths['dsai:random'] < strengths['dsai:policy:learner']


def test_learner_batches_cover_every_index_once():
    import critic_diagnostics
    lengths = np.random.default_rng(2).integers(10, 500, 1000)
    batches = critic_diagnostics.learner_batches(lengths, 64, np.random.default_rng(3))
    assert sorted(np.concatenate(batches).tolist()) == list(range(1000))
    assert all(len(b) <= 64 for b in batches)


@pytest.mark.skipif(not DATA.is_dir(), reason='needs runs/dojo/collect-test')
def test_csv_trajectory_join():
    import critic_diagnostics
    from swccg_policy.trajectory import trajectory_files
    rows = critic_diagnostics.rollout_rows(DATA / 'rollouts')
    metas = [m for path in sorted(trajectory_files(DATA / 'trajectories')) for m in critic_diagnostics.scan_meta(path)]
    assert metas and all(m['g'] in rows for m in metas)
    for meta in metas:
        row = rows[meta['g']]
        # the recorded seat is a learner seat, and the CSV winner agrees with the trajectory's
        assert row[f"{meta['s']}_ai"] == 'policy:learner'
        assert (row['winner'] or None) == (meta['w'] if meta['w'] in ('ds', 'ls') else None)


@GPU_DATA
def test_critic_modes_and_gradient_groups():
    import critic_diagnostics as cd
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_rl import predict_values
    from swccg_policy.v4_trajectory import TrajectoryReader
    cd.setup()
    _, critic, tables, _ = load(CHECKPOINT, 'cuda')
    critic.encoder.clear_tables()
    reader = TrajectoryReader(tables, [])
    game = reader.load(sorted(trajectory_files(DATA / 'trajectories'))[0])[0]
    items = game['critic_items'][:4]
    with torch.no_grad():
        fp32 = cd.critic_value(critic, stack(items, 'cuda'), 'fp32').cpu().numpy()
        noisy = cd.critic_value(critic, stack(items, 'cuda'), 'fp32-noise').cpu().numpy()
    np.testing.assert_allclose(fp32, predict_values(critic, items, cd.FP32, 96), rtol=0, atol=1e-5)
    assert 0 < np.abs(noisy - fp32).max() < 0.05
    grads = cd.critic_gradient(critic, items, np.zeros(4, np.float32), np.arange(4), 'fp32')
    groups = {}
    for name, g in grads.items():
        groups.setdefault(cd.group_of(name), []).append(g)
    total = sum(float(torch.cat(v).norm() ** 2) for v in groups.values())
    assert total == pytest.approx(float(cd.flat(grads).norm() ** 2), rel=1e-5)
    assert {'head', 'block0', 'block5'} <= set(groups)
