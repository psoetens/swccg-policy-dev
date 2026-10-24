"""Plan objective-dojo A-3: record learner trajectories with a fixed checkpoint and no learner (checks only).

Plays `--games` games with the checkpoint as `policy:learner` (so trajectories and their Dojo rewards are written)
against `--opponent`, under the current rules (activate once, no visible cancel, no free return). Nothing trains;
the run folder keeps the rollout CSV (with the stats column), the logs and the trajectory metadata.

    python scripts/dojo_smoke.py --tag absorb-endor --pool data/schedules/objective/endor-ops-clean.txt \
        --agent-side ds --games 64 [--jvm -Dtraingen.flipLog=DIR ...]
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from dojo_coverage import require
from evaluate_v31 import prepare_model
from dojo_lib import folder_metas

ROOT = Path(__file__).resolve().parents[1]
N1 = ROOT / 'runs/knowledge-v3.1/v31-train8h-n1/final.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--checkpoint', type=Path, default=N1)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=909001)
    parser.add_argument('--opponent', default='policy:learner',
                        help='TrainGen opponent spec, e.g. policy:learner, advanced, random')
    parser.add_argument('--agent-side', choices=['alternate', 'ds', 'ls'], default='alternate')
    parser.add_argument('--jvm', action='append', default=[], help='Extra -D property (checks only)')
    args = parser.parse_args()
    if bool(args.pool) == bool(args.schedule):
        raise ValueError('Exactly one of --pool and --schedule')
    require([args.pool or args.schedule])
    folder = ROOT / 'runs/dojo/checks' / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('ckpt', 'snaps', 'traj'):
        (folder / name).mkdir()
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'SWCCG_GRAPH_T': '1536', 'SWCCG_V31_CLASSES': str(folder / 'classes'),
                   'CUDA_MPS_PIPE_DIRECTORY': str(folder / 'no-mps')}
    subprocess.run(['bash', 'scripts/efficient_java_check.sh', 'CorpusRandomCheck'],
                   env=environment, check=True, stdout=subprocess.DEVNULL)
    prepare_model(args.checkpoint, folder / 'ckpt/latest.ts')
    shutil.move(str(folder / 'ckpt/latest.ts.bundle'), folder / 'bundle')
    (folder / 'ckpt/latest.ts.json').write_text(json.dumps({'iteration': 0}) + '\n')
    native = {'knowledge': 'true', 'schema': 4, 'captureCritic': 'true', 'binaryTrajectory': 'true',
              'trajectoryCompression': 1, 'corpusThreadSeeds': 'true',
              'titleTable': ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv',
              'native.run': folder, 'native.bundle': folder / 'bundle', 'native.bc': folder / 'ckpt/latest.ts',
              'native.lib': ROOT / 'native/build-v3.1/libswccgpolicy.so', 'native.tf32': 'false',
              'native.gamesPerFile': 2, 'native.waitMs': 0.5, 'native.seed': args.seed,
              'native.maxQueueSteps': 10 ** 8, 'native.maxQueueBytes': 2 ** 40,}
    decks = ['--pool', str(args.pool)] if args.pool else ['--schedule', str(args.schedule)]
    command = ['bash', 'scripts/v31_engine.sh', *[f'-Dpolicy.{key}={value}' for key, value in native.items()],
               *args.jvm, 'com.gempukku.swccgo.bench.TrainGen', '--decks', str(ROOT), *decks,
               '--games', str(args.games), '--threads', str(args.threads), '--warmup', '0',
               '--agent', 'policy:learner', '--opponent', args.opponent, '--agentSide', args.agent_side,
               '--cap', '6000', '--maxGameMs', '120000', '--seed', str(args.seed), '--pairSeed', str(args.seed),
               '--tag', args.tag, '--stackLog', str(folder / 'errors.log'), '--out', str(folder / 'rollout.csv')]
    (folder / 'command.json').write_text(json.dumps(command, indent=1) + '\n')
    with (folder / 'rollout.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = list(csv.DictReader((folder / 'rollout.csv').open()))
    metas = folder_metas(folder / 'traj')
    (folder / 'trajectory-meta.jsonl').write_text(''.join(json.dumps(meta) + '\n' for meta in metas))
    print(json.dumps({'games': len(rows), 'outcomes': sorted({row['outcome'] for row in rows}),
                      'trajectories': len(metas), 'folder': str(folder)}))


if __name__ == '__main__':
    main()
