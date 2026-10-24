"""Plan nvf-b-local: checkpoints and snapshots are pruned by iteration number, so learner-10003 is newer than 9999."""
from pathlib import Path

from swccg_policy import v4_rl


def test_prune_keeps_the_newest_past_ten_thousand(tmp_path, monkeypatch):
    monkeypatch.setattr(v4_rl, 'ROOT', tmp_path / 'elsewhere')
    for iteration in (9997, 9998, 9999, 10000, 10003):
        (tmp_path / f'learner-{iteration:04d}.pt').write_text('x')
        (tmp_path / f'learner-{iteration:04d}.ts').write_text('x')
    v4_rl.prune(tmp_path.glob('learner-*.pt'), 2, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('learner-*.pt')) == ['learner-10000.pt', 'learner-10003.pt']
    assert sorted(p.name for p in tmp_path.glob('learner-*.ts')) == ['learner-10000.ts', 'learner-10003.ts']


def test_iteration_order():
    names = ['snap-10003.ts', 'snap-9999.ts', 'snap-0005.ts', 'snap-10000.ts']
    assert [n for n in sorted(names, key=lambda n: v4_rl.iteration_order(Path(n)))] == \
        ['snap-0005.ts', 'snap-9999.ts', 'snap-10000.ts', 'snap-10003.ts']
