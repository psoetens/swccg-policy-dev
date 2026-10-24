"""Trajectory file helpers shared by V1 (JSON `part-*.jsonl.gz`) and V3.1 (SWTRJ003 `part-*.traj`, read by
`v4_trajectory.py`, which uses this module's frame header and decision types).

Plan remove-old-generations: the SWTRJ001 / SWTRJ002 binary readers and writers (V2 / V3) were removed.
"""
import hashlib
import struct
from pathlib import Path

FRAME = struct.Struct('>IQQI')
TYPES = ('EMPTY', 'INTEGER', 'MULTIPLE_CHOICE', 'ARBITRARY_CARDS',
         'CARD_ACTION_CHOICE', 'ACTION_CHOICE', 'CARD_SELECTION')


def trajectory_files(folder):
    return sorted(p for p in Path(folder).glob('part-*')
                  if p.name.endswith(('.jsonl.gz', '.traj')))


def card_fingerprint(cards):
    return hashlib.sha256(''.join(f'{k}\t{v}\n' for k, v in sorted(cards.items())).encode()).digest()


def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise ValueError('truncated binary trajectory')
    return b
