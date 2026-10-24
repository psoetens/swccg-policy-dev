"""Write the TorchScript actor next to a V3.1 checkpoint (X.pt -> X.ts), as a fixed training opponent needs it
(`policy:ck:X.pt` in --opponents; v31_stage.check_fixed_opponents). Same export as v31_stage / evaluate_v31.

    PYTHONPATH=src:scripts .venv/bin/python scripts/export_torchscript.py models/D/policy.pt [more.pt ...] [--force]
"""
import argparse
from pathlib import Path

import torch

from swccg_policy.native import export_ts
from swccg_policy.v4_checkpoint import load


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--force', action='store_true', help='overwrite an existing .ts')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for checkpoint in args.checkpoints:
        out = checkpoint.with_suffix('.ts')
        if out.exists() and not args.force:
            print(f'{out} exists')
            continue
        actor, critic, tables, _ = load(checkpoint, args.device)
        export_ts(actor, out)
        print(f'{out} written (table sha {tables.sha[:16]})')
        del actor, critic, tables


if __name__ == '__main__':
    main()
