from pathlib import Path

from swccg_policy import v4_rl


def test_native_snapshots_are_complete_when_visible_and_retention_keeps_recent_actors(tmp_path, monkeypatch):
    for name in ('ckpt', 'snaps'):
        (tmp_path / name).mkdir()

    class Traced:
        def save(self, path):
            Path(path).write_bytes(b'complete actor export')

    def save_checkpoint(path, *unused):
        path.write_bytes(b'complete resumable checkpoint')

    def export_actor(actor, path):
        traced = Traced()
        traced.save(path)
        return traced

    def interrupted_copy(source, destination):
        destination.write_bytes(b'partial')
        # PolicyEngine discovers *.ts. Even mid-copy every visible opponent must be complete.
        for visible in (tmp_path / 'snaps').glob('*.ts'):
            assert visible.read_bytes() == b'complete actor export'
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(v4_rl, 'save', save_checkpoint)
    monkeypatch.setattr(v4_rl, 'export_ts', export_actor)
    monkeypatch.setattr(v4_rl, 'update_traced', lambda *unused: None)
    monkeypatch.setattr(v4_rl.shutil, 'copyfile', interrupted_copy)
    traced = None
    for iteration in (5, 10, 15):
        traced, _ = v4_rl.publish(tmp_path, None, None, None, {'iteration': iteration}, [], traced, 5)
        v4_rl.prune((tmp_path / 'snaps').glob('snap-*.ts'), 2, tmp_path)
    assert sorted(path.name for path in (tmp_path / 'snaps').iterdir()) == ['snap-0010.ts', 'snap-0015.ts']
    assert (tmp_path / 'ckpt/latest.pt').read_bytes() == b'complete resumable checkpoint'
