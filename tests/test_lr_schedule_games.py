"""Plan nvf-b-local: a cosine schedule over learner games (the same decay per game on a machine of another speed)."""
import argparse
import sys

import pytest

from swccg_policy import v4_rl


def args(games=116000, offset=0, minutes=1800):
    return argparse.Namespace(lr_schedule='cosine', minutes=minutes, lr_schedule_minutes=0, lr_schedule_offset_minutes=0,
                              lr_schedule_games=games, lr_schedule_offset_games=offset)


def test_cosine_over_games_ignores_the_clock():
    a = args()
    assert v4_rl.arm_lr(1.0, a, 10 * 3600, 0) == pytest.approx(1.0)
    assert v4_rl.arm_lr(1.0, a, 0, 58000) == pytest.approx(0.5)
    assert v4_rl.arm_lr(1.0, a, 0, 116000) == pytest.approx(0.0, abs=1e-12)
    assert v4_rl.arm_lr(1.0, a, 0, 130000) == pytest.approx(0.0, abs=1e-12)


def test_restart_continues_over_games():
    whole, restarted = args(), args(offset=40000)
    for games in (0, 1, 18000, 75999):
        assert v4_rl.arm_lr(5e-5, restarted, 0, games) == pytest.approx(v4_rl.arm_lr(5e-5, whole, 0, 40000 + games))


def test_wall_clock_unchanged_without_games():
    a = args(games=0, minutes=600)
    assert v4_rl.arm_lr(1.0, a, 300 * 60, 99999) == pytest.approx(0.5)


def test_learner_and_stage_flags():
    captured = {}
    argv, original = sys.argv, v4_rl.learn
    try:
        sys.argv = ['v4_rl', '--run', '/tmp/x', '--initial', '/tmp/y.pt', '--lr-schedule', 'cosine',
                    '--lr-schedule-games', '116000', '--lr-schedule-offset-games', '4000']
        v4_rl.learn = lambda parsed: captured.setdefault('args', parsed)
        v4_rl.main()
    finally:
        sys.argv, v4_rl.learn = argv, original
    assert (captured['args'].lr_schedule_games, captured['args'].lr_schedule_offset_games) == (116000, 4000)
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/v31_stage.py').read_text()
    assert "'--lr-schedule-games', str(args.lr_schedule_games)" in source
    assert "'--lr-schedule-offset-games', str(args.lr_schedule_offset_games)" in source


def wsd(offset=0, warmup=8000, games=116254, save=(), trunk=0):
    return argparse.Namespace(lr_schedule='wsd', minutes=1800, lr_schedule_minutes=0, lr_schedule_offset_minutes=0,
                              lr_schedule_games=games, lr_schedule_offset_games=offset, lr_warmup_games=warmup,
                              lr_decay_fraction=0.1, lr_floor=0.05, save_at_games=list(save), trunk_games=trunk)


def test_wsd_shape():
    """Plan nvf-b-local A1: warm-up 8k games, stable to 90%, cosine to 5% over the last 10%."""
    a, start = wsd(), 116254 * 0.9
    assert v4_rl.arm_lr(1.0, a, 0, 0) == pytest.approx(0.0)
    assert v4_rl.arm_lr(1.0, a, 0, 4000) == pytest.approx(0.5)
    assert v4_rl.arm_lr(1.0, a, 0, 8000) == pytest.approx(1.0)
    assert v4_rl.arm_lr(1.0, a, 0, 60000) == pytest.approx(1.0)
    assert v4_rl.arm_lr(1.0, a, 0, int(start)) == pytest.approx(1.0, abs=1e-6)
    assert v4_rl.arm_lr(1.0, a, 0, (start + 116254) / 2) == pytest.approx(0.525)
    assert v4_rl.arm_lr(1.0, a, 0, 116254) == pytest.approx(0.05)
    assert v4_rl.arm_lr(1.0, a, 0, 130000) == pytest.approx(0.05)


def test_wsd_restart_continues():
    whole, restarted = wsd(), wsd(offset=50000)
    for games in (0, 3000, 54000, 66000):
        assert v4_rl.arm_lr(5e-5, restarted, 0, games) == pytest.approx(v4_rl.arm_lr(5e-5, whole, 0, 50000 + games))
    # Plan B1: a restart from the 4k milestone with the warm-up ending there starts at the (halved) base rate.
    assert v4_rl.arm_lr(2.5e-5, wsd(offset=4000, warmup=4000), 0, 0) == pytest.approx(2.5e-5)


def test_milestones_once_per_crossing():
    a, saved = wsd(save=(4000, 8000, 16000), trunk=104629), set()
    assert v4_rl.milestones_due(a, 3999, saved) == []
    assert v4_rl.milestones_due(a, 4100, saved) == [('games-4000', 4000)]
    saved.add('games-4000')
    assert v4_rl.milestones_due(a, 9000, saved) == [('games-8000', 8000)]
    # A restarted segment does not re-save marks that an earlier segment crossed.
    assert v4_rl.milestones_due(wsd(offset=20000, save=(4000, 8000, 16000), trunk=104629), 90000, set()) == [('trunk', 104629)]


def test_stage_passes_wsd_options():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/v31_stage.py').read_text()
    for option in ("'--lr-warmup-games'", "'--save-at-games'", "'--trunk-games'", "'--save-dir'"):
        assert option in source


def test_learner_parses_wsd_flags():
    captured = {}
    argv, original = sys.argv, v4_rl.learn
    try:
        sys.argv = ['v4_rl', '--run', '/tmp/x', '--initial', '/tmp/y.pt', '--lr-schedule', 'wsd', '--lr-schedule-games',
                    '116254', '--lr-warmup-games', '8000', '--lr-decay-fraction', '0.1', '--lr-floor', '0.05',
                    '--save-at-games', '4000,8000,16000', '--trunk-games', '104629', '--save-dir', '/tmp/m']
        v4_rl.learn = lambda parsed: captured.setdefault('args', parsed)
        v4_rl.main()
    finally:
        sys.argv, v4_rl.learn = argv, original
    a = captured['args']
    assert (a.lr_schedule, a.lr_warmup_games, a.save_at_games, a.trunk_games) == ('wsd', 8000, [4000, 8000, 16000], 104629)
