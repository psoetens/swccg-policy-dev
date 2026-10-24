"""Plan critic-replay rev. 2: score many critic checkpoints on fixed held-out probe sets, built once.

    critic_track.py --out JSON --data DIR [DIR ...] CHECKPOINT [CHECKPOINT ...]

For every data folder (trajectories; the held-out split = every 5th file, as critic_diagnostics.split_files) it builds
the probe once: ~20k decisions spread over the held-out games (held-out MC EV) and the first decision of every decided
held-out seat (first-decision AUC of the value and the win head). Every checkpoint's critic is then scored on each
probe (fp32, unpacked attention, as train-mc's probe), with its iteration / games from rl_meta.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import critic_diagnostics as cd  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--eval-decisions', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    cd.setup()
    _, critic, tables, _ = load(args.checkpoints[0], 'cuda')
    reader = TrajectoryReader(tables, [])
    pool = cd.decode_pool(reader, args.workers)
    probes = {}
    for folder in args.data:
        started = time.monotonic()
        _, held = cd.split_files(folder)
        probes[str(folder)] = (cd.make_probe(reader, held, args.eval_decisions, critic, pool),
                               cd.first_decisions(reader, held, pool))
        print(json.dumps({'probe': str(folder), 'decisions': len(probes[str(folder)][0]['items']),
                          'first_decisions': len(probes[str(folder)][1][0]), 'seconds': time.monotonic() - started}),
              flush=True)
    pool.terminate()
    del critic
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for path in args.checkpoints:
        if str(path) in results:
            continue
        _, critic, _, checkpoint = load(path, 'cuda')
        critic.encoder.clear_tables()
        meta = checkpoint['rl_meta']
        row = {'iteration': meta.get('iteration'), 'games': meta.get('games'), 'steps': meta.get('steps')}
        for folder, (probe, firsts) in probes.items():
            _, metrics = cd.probe_metrics(critic, probe, firsts)
            row[folder] = {k: v for k, v in metrics.items() if k != 'heldout_gae_mse'}
        results[str(path)] = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + '\n')
        print(json.dumps({'checkpoint': str(path), **row}), flush=True)
        del critic, checkpoint
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
