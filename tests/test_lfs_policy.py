import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / '.git').exists(), reason='not a git checkout')
def test_large_and_binary_files_are_in_lfs():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_lfs_policy.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lfs_objects_are_pulled():
    """A clone without `git lfs pull` has pointer files; the models and tables must be real."""
    pointer = b'version https://git-lfs.github.com/spec/v1'
    for path in ['models/D/policy.pt', 'runs/knowledge-v3.1/tables-decipher/cards.npz', 'native/build-v3.1/libswccgpolicy.so']:
        with open(ROOT / path, 'rb') as stream:
            assert not stream.read(len(pointer)) == pointer, f'{path} is an LFS pointer: run git lfs pull'
