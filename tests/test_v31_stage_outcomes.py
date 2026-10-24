"""Plan nvf-b-local: a loop-guarded game ('loop') is a valid rollout outcome; real failures still stop the stage."""
import v31_stage


def rows(*outcomes):
    return [{'outcome': outcome} for outcome in outcomes]


def test_loop_is_accepted():
    assert v31_stage.rollout_ok(0, rows('finished', 'loop', 'cap', 'timeout'), 4, 'ok')


def test_failures_stop_the_stage():
    assert not v31_stage.rollout_ok(0, rows('finished', 'exception'), 2, 'ok')
    assert not v31_stage.rollout_ok(0, rows('finished'), 2, 'ok')
    assert not v31_stage.rollout_ok(1, rows('finished', 'finished'), 2, 'ok')
    assert not v31_stage.rollout_ok(0, rows('finished', 'finished'), 2, 'native batch failed: x')


def test_rando_invalid_loop_is_tolerated():
    # Plan scratch-24h-ab amendment 1: only in a game with a Rando seat.
    rando = [{'outcome': 'finished'}, {'outcome': 'invalid_loop', 'ds_ai': 'rando', 'ls_ai': 'policy:learner'}]
    assert v31_stage.rollout_ok(0, rando, 2, 'ok')
    other = [{'outcome': 'finished'}, {'outcome': 'invalid_loop', 'ds_ai': 'policy:snap', 'ls_ai': 'policy:learner'}]
    assert not v31_stage.rollout_ok(0, other, 2, 'ok')
