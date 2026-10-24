"""Head-to-head round-robin between V3.1 checkpoints on the frozen held-out schedule (D17).

Each pairing reuses evaluate_v31.py with only the reference opponent: 128 paired games, both seats.
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True, help='NAME=CHECKPOINT.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--tag-prefix', default='rr')
    parser.add_argument('--only-with', help='Play only pairings that include this model name')
    args = parser.parse_args()
    models = dict(item.split('=', 1) for item in args.models)
    report = json.loads(args.out.read_text()) if args.out.exists() else {'games': args.games, 'pairings': {}}
    for a, b in itertools.combinations(models, 2):
        if args.only_with and args.only_with not in (a, b):
            continue
        key = f'{a}__vs__{b}'
        if key in report['pairings']:
            continue
        tag = f'{args.tag_prefix}-{a}-vs-{b}'
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluate_v31.py'), '--checkpoint', models[a],
                        '--reference', models[b], '--tag', tag, '--games', str(args.games), '--threads', str(args.threads),
                        '--opponents', 'initial'],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        result = json.loads((ROOT / 'runs/knowledge-v3.1/evaluations' / tag / 'results.json').read_text())['results']['initial']
        report['pairings'][key] = {'agent': a, 'opponent': b, 'wins': result['wins'], 'games': result['games'],
                                   'finished': result['finished'], 'win_rate': result['win_rate_all_games'],
                                   'ci95': result['ci95_all_games'], 'win_rate_finished': result['win_rate_finished'],
                                   'evaluation': tag}
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: report['pairings'][key]['win_rate']}), flush=True)


if __name__ == '__main__':
    main()
