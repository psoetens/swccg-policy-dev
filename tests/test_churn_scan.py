import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import churn_scan  # noqa: E402

LOG = """Start of ls's turn #1
~ls initiates Force drain of 1 at •Endor: Back Door
~ls cancels Force drain at •Endor: Back Door
ls transfers •Anakin's Lightsaber from •Young Skywalker to •Anakin Skywalker, Padawan Learner
ls transfers •Anakin's Lightsaber from •Anakin Skywalker, Padawan Learner to •Young Skywalker
ls transfers •Anakin's Lightsaber from •Young Skywalker to •Anakin Skywalker, Padawan Learner
ls transfers •Anakin's Lightsaber from •Anakin Skywalker, Padawan Learner to •Young Skywalker
ls draws a card into hand from Force Pile
ls draws a card into hand from Force Pile
ls draws a card into hand from Force Pile
ls ship-docks •Home One and •Liberty
ls ship-docks •Home One and •Liberty
ls ship-docks •Home One and •Liberty
ls moves Rebel Snowspeeder from ◇Tatooine: Spaceport Docking Bay to •Hoth: Echo Base using landspeed
ls moves Rebel Snowspeeder from •Hoth: Echo Base to ◇Tatooine: Spaceport Docking Bay using landspeed
ls moves Rebel Snowspeeder from ◇Tatooine: Spaceport Docking Bay to •Hoth: Echo Base using landspeed
Start of ds's turn #2
ds moves •Irol from pilot capacity slot to passenger capacity slot of •Tempest 1
ds moves •Irol from pilot capacity slot to passenger capacity slot of •Tempest 1
"""


def test_scan(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'run-advanced-0.log').write_text(LOG)
    csv_path = tmp_path / 'run.csv'
    csv_path.write_text('game,ds_ai,ls_ai\nrun-advanced-0,advanced,policy:ck:x\n')
    result = churn_scan.scan(logs, churn_scan.policy_sides([csv_path]), threshold=2, repeats_over=2)
    assert result['games'] == 1
    # the 4 transfers: two exact lines x2 (not over 2), one card between the same two places x4 (over 2);
    # the 3 ship-docks of one pair: an exact repeat x3 and a back-and-forth x3
    assert [r['pattern'] for r in result['suspicious']] == ['ls ship-docks <card> and <card>']
    # and the non-unique Rebel Snowspeeder round trip (3 moves between the same two places)
    by_verb = {r['pattern'].split()[1]: r for r in result['back_and_forth']}
    assert set(by_verb) == {'transfers', 'ship-docks', 'moves'}
    assert by_verb['transfers']['actor'] == 'policy' and by_verb['transfers']['max'] == 4
    assert by_verb['ship-docks']['max'] == 3
    assert by_verb['moves']['max'] == 3 and 'Rebel Snowspeeder' in by_verb['moves']['example']
    # drawing cards one by one is benign; the two slot moves stay under the threshold
    assert [b['pattern'] for b in result['benign']] == ['ls draws a card into hand from Force Pile']
    assert result['drains_initiated'] == {'policy': 1}
    assert result['self_cancelled_drains'] == {'policy': 1}
