"""Plan objective-dojo: shared readers for trajectory metadata and the per-game stats column."""
import csv
import json
from pathlib import Path

import orjson

from swccg_policy.trajectory import FRAME, trajectory_files


def metas(path):
    """Per-game-side metadata of one SWTRJ003 file."""
    with open(path, 'rb') as stream:
        stream.read(8 + 1 + 32)  # magic, codec, acting-table SHA
        while header := stream.read(FRAME.size):
            metadata_size, _, stored_size, _ = FRAME.unpack(header)
            yield orjson.loads(stream.read(metadata_size))
            stream.seek(stored_size, 1)


def folder_metas(folder):
    return [meta for path in trajectory_files(folder) for meta in metas(path)]


def stats_rows(csv_paths):
    """Rollout/eval CSV rows with the parsed stats JSON under 'stats'."""
    rows = []
    for path in csv_paths:
        with open(path) as stream:
            for row in csv.DictReader(stream):
                row['stats'] = json.loads(row['stats']) if row.get('stats') else None
                rows.append(row)
    return rows
