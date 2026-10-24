"""Plan precision-capacity-check P-0: collect trajectories from a FIXED V3.1 policy (no learner, no weight changes).

    v31_collect.py --start runs/dojo/focus-s2/final.pt --tag collect-s2 --steps 250000 \
        --pool runs/dojo/focus-pool-v2.txt --opponent-pool runs/dojo/opponent-pool-v2.txt \
        --opponents "policy:learner=0.6,policy:snap=0.25,advanced=0.10,random=0.05" \
        --seed 910001

Reuses v31_stage.prepare() (model export, bundle, snapshots from START's run) and rollout_command() (the training
rollouts). The collector is the queue's consumer: it moves every finished trajectory file (with its sidecar) from the
/tmp spool to runs/dojo/<tag>/trajectories/ and stops once STEPS decisions are stored. 'policy:learner' keeps the start
weights for the whole collection because nothing publishes a new latest.ts.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dojo_coverage  # noqa: E402
import v31_stage  # noqa: E402
from stage_utils import resources, stop  # noqa: E402
from swccg_policy.trajectory import trajectory_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--opponent-pool')
    parser.add_argument('--opponents', required=True)
    parser.add_argument('--seed', type=int, default=910001)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--agent-side', choices=['alternate', 'ds', 'ls'], default='alternate')
    parser.add_argument('--max-minutes', type=float, default=90)
    args = parser.parse_args()
    if not args.tag.replace('-', '').isalnum():
        raise ValueError('Alphanumeric/hyphen tag required')
    dojo_coverage.require([args.pool, *([args.opponent_pool] if args.opponent_pool else [])])
    evidence = ROOT / 'runs/dojo' / args.tag
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / 'rollouts').mkdir()
    out = evidence / 'trajectories'
    out.mkdir()
    work = Path('/tmp') / ('swccg-v31-' + args.tag)
    work.mkdir(exist_ok=False)
    for name in ('ckpt', 'snaps', 'traj', 'traj-consumed'):
        (work / name).mkdir()
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'SWCCG_GRAPH_T': '1536', 'SWCCG_V31_CLASSES': str(evidence / 'classes'),
                   'CUDA_MPS_PIPE_DIRECTORY': str(work / 'no-mps')}
    subprocess.run(['bash', 'scripts/efficient_java_check.sh', 'CorpusRandomCheck'], env=environment, check=True,
                   stdout=subprocess.DEVNULL, cwd=ROOT)
    # prepare() wants the stage's argument object: the training defaults plus what the collection sets.
    stage_args = argparse.Namespace(configuration='V31-c6', variant='a10c', logit_cap=30.0, entropy_coef=0.01,
                                    lr_schedule='constant', knowledge_mask='none', start=args.start, seed=args.seed,
                                    purpose='collection', critic_fp32=False, batch_steps=12000, keep_consumed=0, epochs=1, max_games=0,
                                    minutes=args.max_minutes, pool=args.pool, opponent_pool=args.opponent_pool,
                                    opponents=args.opponents, agent_side=args.agent_side, threads=args.threads,
                                    tag=args.tag)
    v31_stage.prepare(stage_args, evidence, work, environment)
    (evidence / 'configuration.json').write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()}, indent=2) + '\n')
    stored, chunk, rollout, log, reason = 0, 0, None, None, 'steps_complete'
    started = time.time()
    deadline = time.monotonic() + args.max_minutes * 60
    try:
        while stored < args.steps:
            if time.monotonic() > deadline:
                reason = 'time_cap'
                break
            if (evidence / 'STOP').exists():
                reason = 'user_stop'
                break
            for path in trajectory_files(work / 'traj'):
                sidecar = path.with_name(path.name + '.meta.json')
                if not sidecar.exists():
                    continue
                stored += int(json.loads(sidecar.read_text())['steps'])
                shutil.move(path, out / path.name)
                shutil.move(sidecar, out / sidecar.name)
            if rollout is None or rollout.poll() is not None:
                if rollout is not None:
                    log.close()
                    rows = list(csv.DictReader((evidence / 'rollouts' / f'{chunk:04d}.csv').open()))
                    if rollout.returncode or len(rows) != 64:
                        reason = 'rollout_failure'
                        break
                if stored >= args.steps:
                    break
                chunk += 1
                command = v31_stage.rollout_command(stage_args, 4, work, evidence, chunk)
                log = (evidence / 'rollouts' / f'{chunk:04d}.log').open('w')
                rollout = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=ROOT)
            sample = resources()
            if sample['memory_available'] < 2 * 2 ** 30 or sample['disk_free'] < 20 * 2 ** 30:
                reason = 'resource_guard'
                break
            time.sleep(2)
    finally:
        stop(rollout)
        if log:
            log.close()
        for path in trajectory_files(work / 'traj'):  # files finished while stopping
            sidecar = path.with_name(path.name + '.meta.json')
            if sidecar.exists():
                stored += int(json.loads(sidecar.read_text())['steps'])
                shutil.move(path, out / path.name)
                shutil.move(sidecar, out / sidecar.name)
        result = {'reason': reason, 'steps_stored': stored, 'chunks': chunk, 'elapsed_seconds': time.time() - started,
                  'files': len(trajectory_files(out))}
        (evidence / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        if work.parent == Path('/tmp') and work.name == 'swccg-v31-' + args.tag:
            shutil.rmtree(work)
    if reason != 'steps_complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
