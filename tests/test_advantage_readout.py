"""Plan critic-followups BL-49: the readout's ground-truth parsing and ranking arithmetic (synthetic run folder)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import advantage_readout as ar  # noqa: E402


def test_ground_truth_counts_wins_of_the_branching_seat_and_drops_mismatches(tmp_path):
    meta = [{'position': 0, 'option': o, 'rollout': r} for o in (0, 1) for r in range(3)]
    (tmp_path / 'meta-gt.json').write_text(json.dumps(meta))
    # games 0-2: option 0, 3-5: option 1; seat ds. Option 1 wins all, option 0 wins one; game 2 mismatched.
    report = ['t-0\tend\t9\tfinished\tds\texact', 't-1\tend\t9\tfinished\tls\texact', 't-2\tmismatch\t5\ta\tb',
              't-2\tend\t9\tfinished\tds\tmismatch', 't-3\tend\t9\tfinished\tds\texact', 't-4\tend\t9\tfinished\tds\texact',
              't-5\tend\t9\tcap\t\texact']
    (tmp_path / 'report-gt.txt').write_text('\n'.join(report) + '\n')
    (tmp_path / 'states').mkdir()
    (tmp_path / 'states' / 'states.jsonl').write_text('\n'.join(json.dumps({'id': f'rb|{g}', 'game': g, 'index': 10,
                                                                            'side': 'ds'}) for g in range(6)) + '\n')
    _, table = ar.ground_truth(tmp_path)
    assert table[0][0] == [1, 2]  # game 2 dropped (mismatch)
    assert table[0][1] == [2, 3]  # the capped game counts as a non-win


def test_spearman_ranks():
    assert ar.spearman(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.5, 0.9])) == 1.0
    assert ar.spearman(np.array([0.3, 0.2, 0.1]), np.array([0.2, 0.5, 0.9])) == -1.0
    assert ar.spearman(np.array([0.1, 0.1]), np.array([0.2, 0.5])) is None
