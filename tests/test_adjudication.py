"""Plan decision-cap-fixes F3: unfinished games are adjudicated by the life-force difference; official numbers unchanged."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from evaluate_v31 import statistics  # noqa: E402

AGENT = 'policy:agent'


def row(outcome, winner='', agent_ds=True, lf_agent=10, lf_opponent=10):
    stats = {'ds': {'life_force': lf_agent if agent_ds else lf_opponent},
             'ls': {'life_force': lf_opponent if agent_ds else lf_agent}}
    return {'outcome': outcome, 'winner': winner, 'ds_ai': AGENT if agent_ds else 'advanced',
            'ls_ai': 'advanced' if agent_ds else AGENT, 'stats': json.dumps(stats), 'turns': '10', 'decisions': '100'}


def test_unfinished_games_are_scored_by_life_force_and_official_numbers_stay():
    rows = [row('finished', 'ds'), row('finished', 'ds', agent_ds=False),          # a win and a loss
            row('cap', lf_agent=20, lf_opponent=5), row('loop', agent_ds=False, lf_agent=3, lf_opponent=9),
            row('timeout', lf_agent=7, lf_opponent=7), {**row('cap'), 'stats': ''}]
    result = statistics(rows, AGENT)
    assert result['wins'] == 1 and result['win_rate_all_games'] == pytest.approx(1 / 6)
    adjudicated = result['adjudicated']
    assert (adjudicated['unfinished_won'], adjudicated['unfinished_lost'], adjudicated['unfinished_tied'],
            adjudicated['unfinished_unknown']) == (1, 1, 1, 1)
    assert adjudicated['unfinished'] == 4
    assert adjudicated['win_rate'] == pytest.approx((1 + 1 + 0.5) / 6)
