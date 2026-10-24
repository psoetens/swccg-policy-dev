"""Plan precision-capacity-check: smoke test of scripts/v31_precision_capacity.py (skipped without GPU or data)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'


@pytest.mark.skipif(not torch.cuda.is_available() or not DATA.is_dir() or not CHECKPOINT.exists(),
                    reason='needs a GPU, runs/dojo/collect-test and the focus-s2 checkpoint')
def test_bf16_smoke(tmp_path):
    subprocess.run([sys.executable, str(ROOT / 'scripts/v31_precision_capacity.py'), 'bf16', '--data', str(DATA),
                    '--checkpoint', str(CHECKPOINT), '--n', '64', '--minibatches', '1', '--out', str(tmp_path)],
                   check=True, cwd=ROOT, capture_output=True)
    result = json.loads((tmp_path / 'bf16.json').read_text())
    assert result['decisions'] > 0
    # fp32 repeated is the noise floor: identical choices.
    assert result['all']['fp32-repeat:agree'] == 1.0
    assert 0.9 < result['gradients']['all:actor:cos_fp32_fp32']['median'] <= 1.0 + 1e-6
    # Plan night-integer-fp32: unpacked bf16 gradients beside the packed ones.
    assert 'all:critic:cos_bf16_fp32' in result['gradients']
    assert 'all:critic:cos_bf16-packed_fp32' in result['gradients']


OPTION_INPUTS = ROOT / 'runs/dojo/oi-treat'


@pytest.mark.skipif(not torch.cuda.is_available() or not (OPTION_INPUTS / 'trajectories').is_dir(),
                    reason='needs a GPU and runs/dojo/oi-treat (option inputs v2) with its kept trajectories')
def test_bf16_families_and_integer_branch(tmp_path):
    subprocess.run([sys.executable, str(ROOT / 'scripts/v31_precision_capacity.py'), 'bf16',
                    '--data', str(OPTION_INPUTS / 'trajectories'), '--checkpoint', str(OPTION_INPUTS / 'final.pt'),
                    '--n', '400', '--minibatches', '2', '--out', str(tmp_path)], check=True, cwd=ROOT, capture_output=True)
    gradients = json.loads((tmp_path / 'bf16.json').read_text())['gradients']
    for family in ('all', 'integer', 'card'):
        assert f'{family}:actor:cos_bf16_fp32' in gradients
    assert 'integer:actor_integer_branch:cos_bf16-packed_fp32' in gradients
