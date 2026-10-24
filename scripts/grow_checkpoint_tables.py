"""Plan night-integer-fp32, Amendment 1: a checkpoint grown into a child table bundle, without optimizer state.

Loads CHECKPOINT with --tables (v4_checkpoint.load grows actor and critic; anything but a prefix extension of the
checkpoint's own tables is refused), and saves it with rl_meta kept and no optimizer state: growth adds embedding rows,
so the AdamW state cannot be kept. Without --tables the checkpoint is saved unchanged minus its optimizer state (the
control arm's reset).

    PYTHONPATH=scripts:src .venv/bin/python scripts/grow_checkpoint_tables.py runs/dojo/oi-treat/final.pt OUT.pt \\
        --tables runs/knowledge-v3.1/tables-v7b
"""
import argparse
import json
from pathlib import Path

import torch

from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tables', type=Path, help='child bundle to grow into (omit: only reset the optimizer state)')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    tables = TableBundle.load(args.tables) if args.tables else None
    if tables is not None and tables.sha == payload['table_sha']:
        raise ValueError('--tables is the checkpoint\'s own bundle: nothing to grow')
    actor, critic, grown, _ = load(args.checkpoint, tables=tables, payload=payload)
    metadata = {**payload.get('rl_meta', {}), 'optimizer_reset_from': str(args.checkpoint)}
    if tables is not None:
        metadata['table_growth_parent_sha'] = payload['table_sha']
    save(args.out, actor, critic, grown, metadata)
    print(json.dumps({'checkpoint': str(args.checkpoint), 'out': str(args.out), 'parent_sha': payload['table_sha'],
                      'sha': grown.sha, 'grown': tables is not None,
                      'rows': {family: len(grown.vocabulary.payload[family]) for family in ('q', 'o', 'context')},
                      'cards': len(grown.cards.ids), 'normalizer': grown.vocabulary.payload.get('normalizer_version'),
                      'games': metadata.get('games'), 'iteration': metadata.get('iteration')}))


if __name__ == '__main__':
    main()
