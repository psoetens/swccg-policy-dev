"""Audit sampled behavior probabilities using an actual published trained TorchScript actor."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from swccg_policy.native import write_bundle


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--traced', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=905310)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=False)
    for name in ('ckpt', 'snaps', 'traj'):
        (args.run / name).mkdir()
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'SWCCG_GRAPH_T': '1536', 'SWCCG_V31_CLASSES': str(args.run / 'classes'),
                   'CUDA_MPS_PIPE_DIRECTORY': str(args.run / 'no-mps')}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    iteration = int(checkpoint.get('rl_meta', {}).get('iteration', 0))
    del checkpoint
    inputs = {'checkpoint': str(args.checkpoint), 'published_export': str(args.traced),
              'iteration': iteration, 'seed': args.seed,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'export_sha256': hashlib.sha256(args.traced.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.run / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    write_bundle(args.checkpoint, args.run / 'bundle', 'owned')
    # Preserve the actual published cached export, not a fresh trace from checkpoint weights.
    shutil.copyfile(args.traced, args.run / 'ckpt/latest.ts')
    (args.run / 'ckpt/latest.ts.json').write_text(json.dumps({'iteration': iteration}) + '\n')
    subprocess.run(['bash', 'scripts/efficient_java_check.sh', 'CorpusRandomCheck'],
                   env=environment, check=True, stdout=subprocess.DEVNULL)
    properties = {'knowledge': 'true', 'schema': 4, 'corpusThreadSeeds': 'true', 'captureCritic': 'true',
                  'binaryTrajectory': 'true', 'trajectoryCompression': 1,
                  'titleTable': ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv',
                  'native.run': args.run, 'native.bundle': args.run / 'bundle',
                  'native.bc': args.run / 'ckpt/latest.ts', 'native.tf32': 'false',
                  'native.lib': ROOT / 'native/build-v3.1/libswccgpolicy.so',
                  'native.seed': args.seed, 'native.gamesPerFile': 2, 'native.waitMs': .5,
                  'native.maxQueueSteps': 1000000, 'native.maxQueueBytes': 1073741824}
    command = ['bash', 'scripts/v31_engine.sh', *[f'-Dpolicy.{key}={value}' for key, value in properties.items()],
               'com.gempukku.swccgo.bench.TrainGen', '--decks', '/', '--pool', 'data/schedules/rl_pool_decipher.txt',
               '--games', '4', '--threads', '2', '--warmup', '0', '--seed', str(args.seed),
               '--pairSeed', str(args.seed), '--agent', 'policy:learner', '--opponent', 'random',
               '--agentSide', 'alternate', '--cap', '2000', '--maxGameMs', '120000',
               '--tag', args.run.name, '--out', str(args.run / 'games.csv'),
               '--stackLog', str(args.run / 'errors.log')]
    (args.run / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (args.run / 'engine.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (args.run / 'games.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4 or any(row['outcome'] not in ('finished', 'cap', 'timeout') for row in rows):
        raise RuntimeError('Trained-policy audit rollout failed')
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/check_v31_ppo.py',
                    '--checkpoint', str(args.checkpoint), '--trajectories', str(args.run / 'traj'),
                    '--out', str(args.out), '--logp-only'], env=environment, check=True)
    report = json.loads(args.out.read_text())
    report.update({**inputs, 'source_sha256': {**report['source_sha256'],
                                              str(Path(__file__)): inputs['script_sha256']}})
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
