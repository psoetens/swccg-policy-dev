"""Plan night-lr-decay: a cosine schedule that spans the whole arm and survives crash restarts."""
import argparse
import math
import sys

import pytest

from swccg_policy import v4_rl


def args(minutes=690, total=0, offset=0, schedule='cosine'):
    return argparse.Namespace(lr_schedule=schedule, minutes=minutes, lr_schedule_minutes=total,
                              lr_schedule_offset_minutes=offset)


def test_cosine_over_one_segment():
    a = args(minutes=600)
    assert v4_rl.arm_lr(1.0, a, 0) == pytest.approx(1.0)
    assert v4_rl.arm_lr(1.0, a, 300 * 60) == pytest.approx(0.5)
    assert v4_rl.arm_lr(1.0, a, 600 * 60) == pytest.approx(0.0, abs=1e-12)
    assert v4_rl.arm_lr(1.0, a, 700 * 60) == pytest.approx(0.0, abs=1e-12)


def test_restart_continues_the_schedule():
    """A crash after 200 of 690 minutes: the next segment (490 minutes left) continues where the first stopped."""
    whole = args(minutes=690)
    restarted = args(minutes=490, total=690, offset=200)
    for minute in (0, 1, 100, 245, 489):
        assert v4_rl.arm_lr(5e-5, restarted, minute * 60) == pytest.approx(v4_rl.arm_lr(5e-5, whole, (200 + minute) * 60))
    # Without the offset, the old per-segment behaviour would jump back to the full rate.
    assert v4_rl.arm_lr(5e-5, args(minutes=490), 0) == pytest.approx(5e-5)
    assert v4_rl.arm_lr(5e-5, restarted, 0) == pytest.approx(5e-5 * 0.5 * (1 + math.cos(math.pi * 200 / 690)))


def test_constant_ignores_the_schedule():
    assert v4_rl.arm_lr(3e-4, args(schedule='constant', offset=300), 1000) == 3e-4


def test_learner_and_stage_flags():
    captured = {}
    argv, original = sys.argv, v4_rl.learn
    try:
        sys.argv = ['v4_rl', '--run', '/tmp/x', '--initial', '/tmp/y.pt', '--lr-schedule', 'cosine',
                    '--lr-schedule-minutes', '690', '--lr-schedule-offset-minutes', '200']
        v4_rl.learn = lambda parsed: captured.setdefault('args', parsed)
        v4_rl.main()
    finally:
        sys.argv, v4_rl.learn = argv, original
    parsed = captured['args']
    assert (parsed.lr_schedule_minutes, parsed.lr_schedule_offset_minutes) == (690, 200)
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/v31_stage.py').read_text()
    assert "'--lr-schedule-minutes', str(args.lr_schedule_minutes)" in source
    assert "'--lr-schedule-offset-minutes', str(args.lr_schedule_offset_minutes)" in source
