"""Bounded native self-play supervisor with preserved evidence and a disposable /tmp spool."""

import argparse
import csv
import hashlib
import functools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import torch

import dojo_coverage
from stage_utils import queue_stats, resources, rss_tree, stop
from swccg_policy.native import export_ts, write_bundle
from swccg_policy.table_bundle import TableBundle
from swccg_policy.trajectory import trajectory_files
from swccg_policy.v4_checkpoint import load, new_models, save


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
# Amendment 12: input-normalization variant of a fresh V3.1 model; a10c is the EfficientConfig default.
VARIANTS = {
    'a10c': {'component_norm': True, 'component_norm_scope': 'options_context', 'component_label_gain': 0.1},
    'a10': {'component_norm': True, 'component_norm_scope': 'all', 'component_label_gain': 1.0},
    # D15 H1 candidates (synthetic screening first; used for RL only if they pass the probe gate).
    'a10nl': {'component_norm': True, 'component_norm_scope': 'all_but_labels', 'component_label_gain': 1.0},
    'a10ne': {'component_norm': True, 'component_norm_scope': 'all_but_entity', 'component_label_gain': 1.0},
    'a10ng': {'component_norm': True, 'component_norm_scope': 'all_but_grid', 'component_label_gain': 1.0},
    # D15 step 2c: separate the two differences between a10ne and a10c.
    'a10cg1': {'component_norm': True, 'component_norm_scope': 'options_context', 'component_label_gain': 1.0},
    'a10cq': {'component_norm': True, 'component_norm_scope': 'options_context_question', 'component_label_gain': 0.1},
}


def check_variant(cfg, variant):
    wrong = {key: getattr(cfg, key) for key, value in VARIANTS[variant].items() if getattr(cfg, key) != value}
    if wrong:
        raise ValueError(f'Model does not match variant {variant}: {wrong}')


def fresh_inputs(args):
    """Plan actor-v6-inputs: input options of a fresh model, for actor and critic (BL-25 scaling, BL-26 location)."""
    inputs = {'input_scaling': args.input_scaling} if getattr(args, 'input_scaling', None) else {}
    if inputs.get('input_scaling') == 'v2' and context_version(args.tables) == 5:
        # Context version 5's own scales (efficient.py, plan actor-v6-inputs).
        inputs.update(context_index_scale=18.0, context_value_scale=8.0)
    if getattr(args, 'location_inputs', False):
        inputs.update(location_icons=True, location_state=True, tensor_abi=5)
    return inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def engine_source():
    tree = Path(os.environ.get('SWCCG_ENGINE_TREE', ROOT / '.cache/gemp-knowledge-v5'))
    stamp = tree / 'GEMP_SOURCE'
    return stamp.read_text().strip() if stamp.exists() else f'unstamped tree {tree}'


def prepare(args, evidence, work, environment):
    schema = {'V31-c6': 4, 'V31-c4': 4}[args.configuration]  # plan remove-old-generations: V3.1 only
    initial = work / 'initial.pt'
    if schema == 4:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.start:
            actor, critic, tables, _ = load(args.start, 'cuda')
            check_variant(actor.cfg, args.variant)
            if actor.cfg.knowledge_mask != args.knowledge_mask:
                raise ValueError(f'Start checkpoint knowledge mask {actor.cfg.knowledge_mask} != {args.knowledge_mask}')
            if actor.cfg.logit_cap != args.logit_cap:
                raise ValueError(f'Start checkpoint logit cap {actor.cfg.logit_cap} != requested {args.logit_cap}')
            if fresh_inputs(args):
                raise ValueError('--input-scaling / --location-inputs apply to a fresh model only')
            shutil.copyfile(args.start, initial)
        else:
            tables = TableBundle.load(args.tables)
            critic_layers = 4 if args.configuration.endswith('c4') else 6
            actor, critic = new_models(tables, seed=args.seed, critic_layers=critic_layers,
                                       logit_cap=args.logit_cap, knowledge_mask=args.knowledge_mask,
                                       **VARIANTS[args.variant], **fresh_inputs(args))
            check_variant(actor.cfg, args.variant)
            save(initial, actor, critic, tables, {'iteration': 0, 'fresh_seed': args.seed})
            actor = actor.cuda()
        export_ts(actor, work / 'initial.ts')
        write_bundle(initial, work / 'bundle', 'owned')
        del actor, critic
        torch.cuda.empty_cache()
    shutil.copyfile(initial, work / 'ckpt/latest.pt')
    shutil.copyfile(work / 'initial.ts', work / 'ckpt/latest.ts')
    checkpoint = torch.load(initial, map_location='cpu', weights_only=False)
    iteration = int(checkpoint.get('rl_meta', {}).get('iteration', 0))
    write(work / 'ckpt/latest.ts.json', {'iteration': iteration})
    if args.purpose != 'preflight':
        # A stage started from a checkpoint stores it as a hard link (BL-17): same bytes, no second copy.
        try:
            if not args.start:
                raise OSError('fresh model: nothing to link')
            os.link(args.start, evidence / 'initial.pt')
        except OSError:
            shutil.copyfile(initial, evidence / 'initial.pt')
        shutil.copyfile(work / 'initial.ts', evidence / 'initial.ts')
        if args.start and (args.start.parent / 'snapshots').is_dir():
            for path in (args.start.parent / 'snapshots').iterdir():
                # Plan night-integer-fp32: a snapshot may carry its own table bundle (snap-N.ts.bundle), with which the
                # rollout engine encodes for it (an opponent on other tables than the run's).
                if path.is_dir():
                    shutil.copytree(path, work / 'snaps' / path.name)
                else:
                    shutil.copyfile(path, work / 'snaps' / path.name)
    base = [str(PYTHON), '-m', 'swccg_policy.v4_rl']
    if schema == 4:
        command = [*base, '--run', str(work), '--initial', str(initial), '--minutes', str(args.minutes),
                   '--batch-steps', str(args.batch_steps), '--minibatch', str(getattr(args, 'minibatch', 64)),
                   '--infer-batch', str(getattr(args, 'infer_batch', 96)),
                   # Plan actor-v6-inputs: the token caps grow with the minibatch (512 per row), one microbatch each.
                   '--actor-tokens', str(512 * getattr(args, 'minibatch', 64)), '--critic-tokens', str(512 * getattr(args, 'minibatch', 64)),
                   '--seed', str(args.seed), '--lr', str(getattr(args, 'lr', 5e-5)), '--value-lr', str(getattr(args, 'value_lr', 3e-4)),
                   '--lam', str(getattr(args, 'lam', 0.95)), '--value-warmup', '2', '--packed-attention',
                   '--entropy-coef', str(args.entropy_coef), '--lr-schedule', args.lr_schedule,
                   # Plan night-lr-decay: a schedule spanning the whole arm, continued by a restarted segment.
                   *(['--lr-schedule-minutes', str(args.lr_schedule_minutes)] if getattr(args, 'lr_schedule_minutes', 0) else []),
                   *(['--lr-schedule-offset-minutes', str(args.lr_schedule_offset_minutes)]
                     if getattr(args, 'lr_schedule_offset_minutes', 0) else []),
                   # Plan nvf-b-local: a schedule over learner games instead of wall clock.
                   *(['--lr-schedule-games', str(args.lr_schedule_games)] if getattr(args, 'lr_schedule_games', 0) else []),
                   *(['--lr-schedule-offset-games', str(args.lr_schedule_offset_games)]
                     if getattr(args, 'lr_schedule_offset_games', 0) else []),
                   # Plan nvf-b-local A1/A2: warm-up/stable/decay shape; checkpoints at arm-game marks and the trunk.
                   *(['--lr-warmup-games', str(args.lr_warmup_games), '--lr-decay-fraction', str(args.lr_decay_fraction),
                      '--lr-floor', str(args.lr_floor)] if args.lr_schedule == 'wsd' else []),
                   *(['--save-at-games', args.save_at_games] if getattr(args, 'save_at_games', '') else []),
                   *(['--trunk-games', str(args.trunk_games)] if getattr(args, 'trunk_games', 0) else []),
                   *(['--save-dir', str(evidence / 'milestones')]
                     if getattr(args, 'save_at_games', '') or getattr(args, 'trunk_games', 0) else []),
                   '--keep-consumed', str(args.keep_consumed), '--epochs', str(args.epochs),
                   *(['--max-games', str(args.max_games)] if args.max_games else []),
                   *(['--critic-fp32'] if args.critic_fp32 else []),
                   # Plan night-integer-fp32: an fp32 actor; per-parameter learning-rate factors.
                   *(['--actor-fp32'] if getattr(args, 'actor_fp32', False) else []),
                   *[option for value in getattr(args, 'param_lr_scale', []) for option in ('--param-lr-scale', value)],
                   # Plan critic-v2: critic options, passed only when set.
                   *(['--value-lambda', str(getattr(args, 'value_lambda', None))] if getattr(args, 'value_lambda', None) is not None else []),
                   # Plan links-round2: the discount per policy step, passed only when set.
                   *(['--gamma', str(args.gamma)] if getattr(args, 'gamma', 1.0) != 1.0 else []),
                   *(['--win-coef', str(getattr(args, 'win_coef', 0.0))] if getattr(args, 'win_coef', 0.0) else []),
                   *(['--value-weight-decay', str(getattr(args, 'value_weight_decay', 0.0))] if getattr(args, 'value_weight_decay', 0.0) else []),
                   *(['--critic-accumulate', str(getattr(args, 'critic_accumulate', 1))] if getattr(args, 'critic_accumulate', 1) > 1 else []),
                   # Plan turn-credit: turn-anchored actor advantages, passed only when set.
                   *(['--credit', 'turn'] if getattr(args, 'credit', 'decision') == 'turn' else []),
                   *(['--round-lambda', str(args.round_lambda)] if getattr(args, 'round_lambda', None) is not None else []),
                   *(['--rounds', str(args.rounds)] if getattr(args, 'rounds', None) is not None else []),
                   # Plan critic-noise Amendment 2: actor gradient accumulation, passed only when set.
                   *(['--actor-accumulate', str(getattr(args, 'actor_accumulate', 1))] if getattr(args, 'actor_accumulate', 1) > 1 else []),
                   *(['--critic-warmup', str(getattr(args, 'critic_warmup', 0))] if getattr(args, 'critic_warmup', 0) else []),
                   *(['--load-workers', str(getattr(args, 'load_workers', 0))] if getattr(args, 'load_workers', 0) else []),
                   *(['--infer-tokens', str(getattr(args, 'infer_tokens', 0))] if getattr(args, 'infer_tokens', 0) else []),
                   *(['--critic-replay-updates', str(getattr(args, 'critic_replay_updates', 0)), '--critic-replay-stride',
                      str(getattr(args, 'critic_replay_stride', 8)), '--critic-replay-ratio', str(getattr(args, 'critic_replay_ratio', 1.0))]
                     if getattr(args, 'critic_replay_updates', 0) else []),
                   *(['--archive', str(evidence / 'archive'), '--archive-minutes', str(getattr(args, 'archive_minutes', 0))]
                     if getattr(args, 'archive_minutes', 0) else []),
                   *(['--critic-replay-mode', getattr(args, 'critic_replay_mode', 'uniform')]
                     if getattr(args, 'critic_replay_updates', 0) else []),
                   *(['--critic-replay-tokens', str(getattr(args, 'critic_replay_tokens', 0))]
                     if getattr(args, 'critic_replay_tokens', 0) else []),
                   # Plan learner-speed: the learner speed options, passed only when set.
                   *(['--critic-tf32'] if getattr(args, 'critic_tf32', False) else []),
                   *(['--compile', args.compile] if getattr(args, 'compile', 'off') != 'off' else []),
                   *(['--critic-stream'] if getattr(args, 'critic_stream', False) else []),
                   *(['--prefetch'] if getattr(args, 'prefetch', False) else []),
                   *(['--single-tables'] if getattr(args, 'single_tables', False) else []),
                   # Plan q-critic stage 2: the option-value advantage, passed only when set.
                   *(['--advantage', 'q'] if getattr(args, 'advantage', 'gae') == 'q' else []),
                   *(['--qnet', str(args.qnet)] if getattr(args, 'qnet', None) else []),
                   *(['--qnet-lr', str(args.qnet_lr)] if getattr(args, 'qnet_lr', None) is not None else []),
                   # Plan probe-economics 6c': damage shaping of the actor's advantages, passed only when set.
                   *(['--shaping', args.shaping, '--shaping-weight', str(args.shaping_weight),
                      '--shaping-form', args.shaping_form,
                      '--shaping-target', getattr(args, 'shaping_target', 'both')] if getattr(args, 'shaping', None) else [])]
    return schema, command, iteration


def stage_gpus(args):
    """The GPUs this stage uses (nvidia-smi indices), or None for today's single default GPU."""
    if args.learner_device is None and args.rollout_gpus is None:
        return None
    rollout = ([int(index) for index in args.rollout_gpus.split(',')] if args.rollout_gpus is not None
               else [args.learner_device])
    learner = [args.learner_device] if args.learner_device is not None else [0]
    return sorted(set(learner + rollout))


def rollout_environment(environment, args):
    if args.rollout_gpus is None:
        return environment
    return {**environment, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': args.rollout_gpus}


def gpu_over_limit(sample, limit_gib=28):
    """Per GPU (M-3): LIMIT GiB, or 4 GiB below a smaller card's total. Plan critic-noise Amendment 2: only called with
    --gpu-guard-gib (off by default)."""
    if 'gpus' not in sample:
        return sample['gpu_used_mib'] >= limit_gib * 1024
    return any(gpu['used_mib'] >= min(limit_gib * 1024, gpu['total_mib'] - 4 * 1024) for gpu in sample['gpus'])


def context_version(bundle):
    """Plan context-overhaul: the context version of a table bundle (vocab.json context_version, default 4)."""
    return json.loads((Path(bundle) / 'vocab.json').read_text()).get('context_version', 4)


@functools.lru_cache(maxsize=None)
def checkpoint_table_sha(path):
    return torch.load(path, map_location='cpu', weights_only=False)['table_sha']


def check_fixed_opponents(opponents, bundle):
    """A fixed checkpoint opponent (`policy:ck:PATH`) plays with this run's bundle: in context version 4 any V3.1
    checkpoint fits (plan context-overhaul); otherwise its tables must be the run's own (plan scratch-24h-ab: D in a
    fresh run with D's tables) and its TorchScript export must exist."""
    paths = [key.split('=')[0][len('policy:ck:'):].removesuffix('@greedy')
             for key in opponents.split(',') if key.startswith('policy:ck:')]
    if not paths or context_version(bundle) == 4:
        return
    run_sha = json.loads((Path(bundle) / 'bundle.json').read_text())['table_sha']
    for path in paths:
        if not Path(path).with_suffix('.ts').exists():
            raise ValueError(f'Fixed checkpoint opponent {path} has no TorchScript export {Path(path).with_suffix(".ts")}')
        if checkpoint_table_sha(path) != run_sha:
            raise ValueError(f'Fixed checkpoint opponent {path} has other tables than this run (context version '
                             f'{context_version(bundle)} needs the same tables)')


def rollout_command(args, schema, work, evidence, chunk):
    native = {'knowledge': 'true', 'schema': schema, 'context': context_version(work / 'bundle'), 'captureCritic': 'true', 'binaryTrajectory': 'true',
              'trajectoryCompression': 1, 'corpusThreadSeeds': 'true',
              # Plan dojo-v-cards: a V-card checkpoint encodes with its own (grown) title table.
              'titleTable': getattr(args, 'title_table', None) or ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv',
              'unknownKeysDir': ROOT / 'runs/unknown-keys',  # plan actor-v6-inputs: unknown texts
              'native.run': work, 'native.bundle': work / 'bundle', 'native.bc': work / 'initial.ts',
              'native.lib': ROOT / 'native/build-v3.1/libswccgpolicy.so',
              'native.tf32': 'false', 'native.gamesPerFile': 2, 'native.waitMs': 0.5,
              'native.seed': args.seed + chunk,
              'native.maxQueueSteps': 36000, 'native.maxQueueBytes': 1073741824}
    check_fixed_opponents(args.opponents, work / 'bundle')
    # Plan links-round2: balanced matchups (lopsided pairings drawn at their listed weight).
    weights = [f'-Dtraingen.pairWeights={Path(args.pair_weights).resolve()}'] if getattr(args, 'pair_weights', None) else []
    if getattr(args, 'deck_weights', None):
        weights.append(f'-Dtraingen.deckWeights={Path(args.deck_weights).resolve()}')
    command = ['bash', 'scripts/v31_engine.sh', *[f'-Dpolicy.{key}={value}' for key, value in native.items()], *weights,
               'com.gempukku.swccgo.bench.TrainGen', '--decks', str(ROOT), '--pool', args.pool,
               *(['--opponentPool', args.opponent_pool] if args.opponent_pool else []),
               '--games', str(getattr(args, 'rollout_games', 64)), '--threads', str(args.threads), '--warmup', '0', '--agent', 'policy:learner',
               '--opponent', args.opponents, '--agentSide', args.agent_side,
               '--cap', '6000', '--maxGameMs', '120000', '--seed', str(args.seed + chunk),
               '--pairSeed', str(args.seed + chunk), '--tag', f'{args.tag}-{chunk}',
               '--stackLog', str(evidence / 'rollouts' / f'{chunk:04d}-errors.log'),
               '--out', str(evidence / 'rollouts' / f'{chunk:04d}.csv')]
    return command


def summarize(evidence, started, result, args):
    metrics_path = evidence / 'learner.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    lag_path = evidence / 'load-policy-lag.jsonl'
    lags = ({row['iteration']: row for row in map(json.loads, lag_path.read_text().splitlines())}
            if lag_path.exists() else {})
    previous = started
    for row in metrics:
        interval = row['time'] - previous
        previous = row['time']
        row.setdefault('queue_wait_seconds', max(0.0, interval - sum(row.get('phases', {}).values())))
        for key in ('policy_lag_mean', 'policy_lag_max'):
            if key in lags.get(row['iteration'], {}):
                row[key] = lags[row['iteration']][key]
    steady = [row for index, row in enumerate(metrics) if index >= 2 and row['time'] - started >= 180]
    records = []
    previous = started
    for row in metrics:
        interval = row['time'] - previous
        previous = row['time']
        if row in steady:
            records.append({'accepted_decisions_per_second': row['steps'] / interval,
                            'seconds_per_10k': {name: seconds * 10000 / row['steps']
                                               for name, seconds in row['phases'].items()}, **row})
    def spread(values):
        return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
                'q75': float(np.quantile(values, .75))} if values else None
    rates = [row['accepted_decisions_per_second'] for row in records]
    report = {'configuration': args.configuration, 'purpose': args.purpose, 'result': result,
              'minutes_requested': args.minutes, 'discard_first_seconds': 180, 'discard_first_updates': 2,
              'updates': metrics, 'steady_updates': records, 'accepted_decisions_per_second': spread(rates),
              'learner_seconds_per_10k': {name: spread([row['seconds_per_10k'][name] for row in records])
                                         for name in ('load', 'values_gae', 'ppo', 'checkpoint')},
              'measured': result['elapsed_seconds'] >= 1200 and len(records) >= 3
                          and result['reason'] in ('duration_complete', 'games_complete'),
              'weights_disposable': args.purpose == 'preflight'}
    samples = json.loads((evidence / 'resources.json').read_text())
    if samples:
        report['resources'] = {
            'peak_gpu_mib': max(row['gpu_used_mib'] for row in samples),
            'peak_learner_rss': max(row['learner_rss'] for row in samples),
            'peak_rollout_rss': max(row['rollout_rss'] for row in samples),
            'minimum_available_ram': min(row['memory_available'] for row in samples),
            'minimum_free_disk': min(row['disk_free'] for row in samples),
            'maximum_queue_bytes': max(row['queue']['bytes'] for row in samples),
            'maximum_queue_steps': max(row['queue']['steps'] for row in samples)}
    rollout_rates = []
    for path in sorted((evidence / 'rollouts').glob('*.log')):
        text = path.read_text()
        requests = re.search(r'native policy: (\d+) requests', text)
        duration = re.search(r'games on \d+ threads in ([\d.]+) s', text)
        started_path = path.with_name(path.stem + '-started.json')
        chunk_started = json.loads(started_path.read_text())['time'] if started_path.exists() else started
        if requests and duration and chunk_started - started >= 180:
            rollout_rates.append(int(requests[1]) / float(duration[1]))
    report['rollout_policy_decisions_per_second'] = spread(rollout_rates)
    report['rollout_rate_note'] = ('All native policy requests, including opponents; '
                                   'completed chunks starting after warmup.')
    report['queue_wait_seconds'] = spread([row.get('queue_wait_seconds', 0) for row in steady])
    report['policy_lag'] = {key: spread([row[key] for row in steady if key in row])
                           for key in ('policy_lag_mean', 'policy_lag_max')}
    write(evidence / 'report.json', report)
    summary = {key: value for key, value in report.items() if key not in ('updates', 'steady_updates')}
    print(json.dumps(summary), flush=True)


# Plan nvf-b-local: the loop guard (plan decision-cap-fixes F2) ends a looping game as 'loop', a valid outcome like
# 'cap'; an engine exception, a short chunk, a failed process or a failed native batch still stops the stage.
ROLLOUT_OUTCOMES = ('finished', 'cap', 'timeout', 'loop')


def rollout_ok(returncode, rows, games, text):
    # Plan scratch-24h-ab amendment 1: Rando_Cal sometimes answers invalidly 20+ times in a row (its own bug, no
    # decisions exported); in a game with a Rando seat that 'invalid_loop' is tolerated, as evaluate_v31.py does.
    def accepted(row):
        return row['outcome'] in ROLLOUT_OUTCOMES or (
            row['outcome'] == 'invalid_loop' and 'rando' in (row.get('ds_ai'), row.get('ls_ai')))
    return (not returncode and len(rows) == games and 'native batch failed' not in text
            and all(accepted(row) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', choices=['V31-c6', 'V31-c4'], required=True)
    parser.add_argument('--purpose', choices=['preflight', 'pilot', 'training'], default='preflight')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--runs', type=Path, default=Path('runs/knowledge-v3.1'),
                        help='Folder for the run evidence (plan objective-dojo: runs/dojo)')
    parser.add_argument('--minutes', type=float, default=22)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-steps', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=883040)
    parser.add_argument('--start', type=Path)
    parser.add_argument('--tables', type=Path, default=ROOT / 'runs/knowledge-v3.1/tables-decipher',
                        help='fresh model only: the table bundle (plan context-overhaul: tables-v5 = context version 5)')
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='a10c',
                        help='Input-normalization variant of a V3.1 model (Amendment 12)')
    # Amendment 13 logit-control options; the defaults reproduce earlier stages exactly.
    parser.add_argument('--logit-cap', type=float, default=0.0, help='Actor soft cap for a fresh V3.1 model')
    # Plan actor-v6-inputs (defaults = the previous behaviour).
    parser.add_argument('--input-scaling', choices=['legacy', 'v2'], help='fresh model: input scaling of actor and critic')
    parser.add_argument('--location-inputs', action='store_true',
                        help='fresh model: location Force icons and location state (tensor ABI 5)')
    parser.add_argument('--minibatch', type=int, default=64, help='PPO minibatch (token caps 512 per row)')
    parser.add_argument('--lr', type=float, default=5e-5, help='actor learning rate')
    parser.add_argument('--value-lr', type=float, default=3e-4, help='critic learning rate')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda (advantages and, by default, value targets)')
    parser.add_argument('--entropy-coef', type=float, default=0.003)
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine', 'wsd'], default='constant')
    parser.add_argument('--lr-warmup-games', type=int, default=0)
    parser.add_argument('--lr-decay-fraction', type=float, default=0.1)
    parser.add_argument('--lr-floor', type=float, default=0.05)
    parser.add_argument('--save-at-games', default='', help='Plan nvf-b-local A2: comma-separated arm games')
    parser.add_argument('--trunk-games', type=int, default=0)
    parser.add_argument('--lr-schedule-minutes', type=float, default=0)
    parser.add_argument('--lr-schedule-offset-minutes', type=float, default=0)
    parser.add_argument('--lr-schedule-games', type=int, default=0)
    parser.add_argument('--lr-schedule-offset-games', type=int, default=0)
    parser.add_argument('--knowledge-mask', choices=['none', 'opp_hidden'], default='none',
                        help='D16 B-run: remove opponent hidden-pile tracking from the actor input')
    parser.add_argument('--opponents', default='policy:learner=0.6,policy:snap=0.2,random=0.1,advanced=0.1')
    parser.add_argument('--title-table', type=Path, default=None,
                        help='title table the rollouts encode with (default: the frozen Decipher table; plan dojo-v-cards: '
                             'runs/v-cards/titles-virtual-display.tsv for V-card checkpoints)')
    # Plan objective-learnability: target-deck pools and a fixed learner seat. Plan objective-dojo: the Dojo rewards and
    # the per-game stats column are always on; there is no switch for them.
    parser.add_argument('--pool', default='data/schedules/rl_pool_decipher.txt')
    parser.add_argument('--agent-side', choices=['alternate', 'ds', 'ls'], default='alternate')
    # Plan dojo-focus: the learner seat draws from --pool, the other seat from --opponent-pool; a games budget.
    parser.add_argument('--opponent-pool')
    # Plan critic-fp32: the critic in fp32, the actor unchanged.
    parser.add_argument('--critic-fp32', action='store_true')
    # Plan night-integer-fp32: the actor in fp32 (critic unchanged); NAME_PREFIX=FACTOR actor learning-rate factors.
    parser.add_argument('--actor-fp32', action='store_true')
    parser.add_argument('--param-lr-scale', action='append', default=[], metavar='NAME_PREFIX=FACTOR')
    # Plan critic-v2: the critic package (see swccg_policy.v4_rl --help); the critic's architecture comes from the
    # start checkpoint (scripts/reinit_critic.py).
    parser.add_argument('--value-lambda', type=float, default=None)
    # Plan links-round2: the discount per policy step; a pairing-weights file for the rollouts (see TrainGen).
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--pair-weights')
    parser.add_argument('--deck-weights')
    parser.add_argument('--win-coef', type=float, default=0.0)
    parser.add_argument('--value-weight-decay', type=float, default=0.0)
    parser.add_argument('--critic-accumulate', type=int, default=1)
    parser.add_argument('--actor-accumulate', type=int, default=1)
    parser.add_argument('--credit', choices=['decision', 'turn'], default='decision')
    parser.add_argument('--round-lambda', type=float)
    parser.add_argument('--rounds', type=int)
    # Plan critic-noise Amendment 2 (Peter: "let them crash if really full"): the GPU memory guard is off unless set
    # (GiB; for runs on a GPU that also drives the display). The RAM and disk guards always apply.
    parser.add_argument('--gpu-guard-gib', type=float, default=0)
    parser.add_argument('--critic-warmup', type=int, default=0)
    # Plan critic-v2 speed-up (user 2026-10-08): bigger rollout chunks (less JVM start-up and tail idle), parallel
    # batch decoding and a larger value-pass batch in the learner. Defaults are the previous behaviour.
    parser.add_argument('--rollout-games', type=int, default=64)
    parser.add_argument('--load-workers', type=int, default=0)
    parser.add_argument('--infer-batch', type=int, default=96)
    parser.add_argument('--infer-tokens', type=int, default=0)
    # Plan critic-replay: the critic replay cache (see swccg_policy.v4_rl --help).
    parser.add_argument('--critic-replay-updates', type=int, default=0)
    parser.add_argument('--critic-replay-stride', type=int, default=8)
    parser.add_argument('--critic-replay-ratio', type=float, default=1.0)
    parser.add_argument('--critic-replay-tokens', type=int, default=0)
    parser.add_argument('--critic-replay-mode', choices=['uniform', 'once'], default='uniform')
    # Plan learner-speed (see swccg_policy.v4_rl --help).
    parser.add_argument('--critic-tf32', action='store_true')
    parser.add_argument('--compile', choices=['off', 'tables'], default='off')
    parser.add_argument('--critic-stream', action='store_true')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--single-tables', action='store_true')
    # Plan q-critic stage 2 (see swccg_policy.v4_rl --help).
    parser.add_argument('--advantage', choices=['gae', 'q'], default='gae')
    parser.add_argument('--qnet', type=Path)
    parser.add_argument('--qnet-lr', type=float)
    # Plan probe-economics 6c' (see swccg_policy.v4_rl --help).
    parser.add_argument('--shaping', choices=['damage'], default=None)
    parser.add_argument('--shaping-weight', type=float, default=0.0)
    parser.add_argument('--shaping-form', choices=['potential', 'plain'], default='potential')
    parser.add_argument('--shaping-target', choices=['actor', 'both'], default='both',
                        help="plan shaping-consistent: 'both' shapes the rewards (critic and actor), 'actor' is 6c'")
    parser.add_argument('--archive-minutes', type=float, default=0,
                        help='plan critic-replay rev. 2: copy the full checkpoint to <run>/archive every N minutes')
    parser.add_argument('--max-games', type=int, default=0,
                        help='Training stops after this many games (learner count); --minutes is then the safety cap')
    # Plan permanent-play-rules: one activation per turn, no all-visible "Done to cancel" and no free return are
    # engine rules for every game (no options).
    parser.add_argument('--epochs', type=int, default=1, help='PPO passes per batch (user 2026-10-03: one pass by default)')
    parser.add_argument('--keep-consumed', type=int, default=4, help='Consumed trajectory files preserved as evidence')
    parser.add_argument('--cpu-cores', help='Comma-separated fixed CPU allocation for every matched configuration')
    # Plan host-migration M-3: GPUs by nvidia-smi index; unset = today's behaviour (the default CUDA device).
    parser.add_argument('--learner-device', type=int, help='GPU index (nvidia-smi numbering) for the learner')
    parser.add_argument('--rollout-gpus', help='CUDA_VISIBLE_DEVICES for the rollouts (nvidia-smi numbering); '
                        'default: the learner GPU')
    args = parser.parse_args()
    gpus = stage_gpus(args)
    if gpus:
        # Before any CUDA call in this process: the learner (and the stage's own checkpoint load) use this GPU.
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if args.learner_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.learner_device)
    if args.cpu_cores:
        cores = {int(core) for core in args.cpu_cores.split(',')}
        if not cores or not cores <= os.sched_getaffinity(0):
            raise ValueError('Requested CPU cores are unavailable')
        os.sched_setaffinity(0, cores)
    if not args.tag.replace('-', '').isalnum():
        raise ValueError('Alphanumeric/hyphen tag required')
    evidence = ROOT / args.runs / args.tag
    if args.purpose == 'preflight':
        if not args.tag.startswith('pipeline-') or not 20 <= args.minutes <= 24:
            raise ValueError('Preflights require pipeline-* names and 20..24 minutes')
        attempted = list((ROOT / 'runs/knowledge-v3.1').glob('pipeline-*/configuration.json'))
        if len(attempted) >= 8:
            raise ValueError('Eight-preflight authorization exhausted')
    else:
        gates = json.loads((ROOT / 'runs/knowledge-v3.1/gates.json').read_text())
        if not all(gates.get(name, {}).get('passed') is True for name in ('A', 'B', 'C', 'D', 'E')):
            raise ValueError('Implementation gates must pass before an RL stage')
        if args.purpose == 'pilot' and args.minutes != 30:
            raise ValueError('Authorized pilot is 30 minutes')
        # Plan objective-learnability: four-hour specialist stages, only on a target-deck pool. Plan critic-replay
        # rev. 3 (user 2026-10-09): 3-hour arms; at most four hours (a crash restart runs the remaining minutes).
        objective_stage = args.pool != 'data/schedules/rl_pool_decipher.txt' and 0 < args.minutes <= 240
        if args.lr_schedule == 'wsd' and not args.lr_schedule_games:
            raise ValueError('--lr-schedule wsd runs over --lr-schedule-games')
        # Plan nvf-b-local: a games budget on a slower machine may need up to 30 h (the time is then the safety cap).
        if args.max_games and not (args.purpose == 'training' and args.minutes <= 1800 and args.opponent_pool):
            raise ValueError('A games budget is authorized for dojo-focus training stages (opponent pool, <= 30 h cap)')
        if args.purpose == 'training' and args.minutes != 480 and not objective_stage and not args.max_games:
            raise ValueError('Authorized training stage is eight hours (four hours for an objective pool)')
    # Plan objective-dojo A2: every objective, Epic Event and Jedi Test in the pool needs a Dojo reward (fatal).
    dojo_coverage.require([args.pool, *([args.opponent_pool] if args.opponent_pool else [])])
    evidence.mkdir(exist_ok=False)
    (evidence / 'rollouts').mkdir()
    work = Path('/tmp') / ('swccg-v31-' + args.tag)
    work.mkdir(exist_ok=False)
    for name in ('ckpt', 'snaps', 'traj', 'traj-consumed'):
        (work / name).mkdir()
    for name in ('learner.jsonl', 'stale-discards.jsonl', 'load-policy-lag.jsonl'):
        (work / name).symlink_to(evidence / name)
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'SWCCG_GRAPH_T': '1536', 'SWCCG_V31_CLASSES': str(evidence / 'classes'),
                   'CUDA_MPS_PIPE_DIRECTORY': str(work / 'no-mps')}
    # Every comparison uses the same explicit MPS-off setting and the same current seeded harness.
    subprocess.run(['bash', 'scripts/efficient_java_check.sh', 'CorpusRandomCheck'],
                   env=environment, check=True, stdout=subprocess.DEVNULL)
    schema, learner_command, initial_iteration = prepare(args, evidence, work, environment)
    files = [*ROOT.glob('src/swccg_policy/*.py'), *ROOT.glob('engine/**/*.java'), Path(__file__),
             ROOT / 'scripts/v31_engine.sh', ROOT / 'scripts/prepare_efficient_overlay.py']
    write(evidence / 'configuration.json', {
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'learner_command': learner_command, 'initial_iteration': initial_iteration, 'mps': False,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'environment': {key: value for key, value in environment.items()
                        if key.startswith(('SWCCG_', 'CUDA_MPS', 'CUDA_VISIBLE', 'CUDA_DEVICE_ORDER'))},
        'frozen_python_commit': None, 'work_directory': str(work),
        # Plan engine-on-feat-brainiac: the Gemp source of the engine tree (repo, branch, commit).
        'gemp_source': engine_source()})
    learner = None
    rollout = None
    chunk = 0
    reason = 'duration_complete'
    started = time.time()
    write(evidence / 'started.json', {'time': started})
    deadline = time.monotonic() + args.minutes * 60
    samples = []
    logs = []
    try:
        log = (evidence / 'learner.log').open('w')
        logs.append(log)
        # Plan actor-v6-inputs: larger minibatches fragment the caching allocator (PPO reserved 28.9 GB for 15.5 GB
        # allocated at minibatch 256); expandable segments return that memory, so the 28 GiB guard sees the real need.
        learner_environment = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', **environment}
        learner = subprocess.Popen(learner_command, env=learner_environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while time.monotonic() < deadline:
            if (evidence / 'STOP').exists():
                reason = 'user_stop'
                break
            if learner.poll() is not None:
                reason = ('games_complete' if args.max_games and learner.returncode == 0 and (work / 'GAMES_COMPLETE').exists()
                          else 'learner_exited_' + str(learner.returncode))
                break
            if rollout is None or rollout.poll() is not None:
                if rollout is not None:
                    logs[-1].close()
                    csv_path = evidence / 'rollouts' / f'{chunk:04d}.csv'
                    rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
                    text = (evidence / 'rollouts' / f'{chunk:04d}.log').read_text()
                    if not rollout_ok(rollout.returncode, rows, getattr(args, 'rollout_games', 64), text):
                        reason = 'rollout_failure'
                        break
                chunk += 1
                command = rollout_command(args, schema, work, evidence, chunk)
                write(evidence / 'rollouts' / f'{chunk:04d}-command.json', command)
                write(evidence / 'rollouts' / f'{chunk:04d}-started.json', {'time': time.time()})
                log = (evidence / 'rollouts' / f'{chunk:04d}.log').open('w')
                logs.append(log)
                rollout = subprocess.Popen(command, env=rollout_environment(environment, args), stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            sample = {'elapsed': time.time() - started, 'queue': queue_stats(work / 'traj'),
                      'learner_rss': rss_tree(learner.pid), 'rollout_rss': rss_tree(rollout.pid), **resources(gpus)}
            samples.append(sample)
            if sample['memory_available'] < 2 * 2 ** 30 or sample['disk_free'] < 20 * 2 ** 30:
                reason = 'resource_guard'
                break
            if getattr(args, 'gpu_guard_gib', 0) and gpu_over_limit(sample, args.gpu_guard_gib):
                reason = 'gpu_memory_guard'
                break
            time.sleep(5)
    finally:
        (work / 'STOP').touch()
        stop(rollout)
        if learner is not None and learner.poll() is None:
            try:
                learner.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stop(learner)
        for log in logs:
            log.close()
        write(evidence / 'resources.json', samples)
        if (work / 'ckpt/latest.pt').exists():
            shutil.copyfile(work / 'ckpt/latest.pt', evidence / 'final.pt')
            shutil.copyfile(work / 'ckpt/latest.ts', evidence / 'final.ts')
        if args.purpose != 'preflight':
            shutil.copytree(work / 'snaps', evidence / 'snapshots')
        preserved = evidence / 'trajectories'
        preserved.mkdir()
        for path in trajectory_files(work / 'traj-consumed'):
            shutil.copyfile(path, preserved / path.name)
            shutil.copyfile(path.with_name(path.name + '.meta.json'), preserved / (path.name + '.meta.json'))
        result = {'reason': reason, 'elapsed_seconds': time.time() - started, 'chunks_started': chunk,
                  'learner_returncode': None if learner is None else learner.poll(),
                  'final_queue': queue_stats(work / 'traj')}
        write(evidence / 'result.json', result)
        summarize(evidence, started, result, args)
        # This directory was created exclusively above and is never inside runs/.
        if work.parent == Path('/tmp') and work.name == 'swccg-v31-' + args.tag:
            shutil.rmtree(work)
    if reason not in ('duration_complete', 'games_complete'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
