"""Plan learner-speed G-1: benchmark, profile and parity of the v4 learner on a fixed batch.

Every configuration runs the learner's own functions (predict_values, advantages, ppo_batches, ppo_pass) on the same
trajectory files, from the same checkpoint and optimizer state, with the arms' settings (fp32 critic, lambda_v 0.995,
win 0.5, weight decay 0.01, accumulate 4, minibatch 64, packed actor attention). Plan learner-minibatch: --recipe v5
runs ctx5-n1's learner settings instead, and --minibatch changes the PPO minibatch (token caps scale with it).

  time     N updates; per-phase seconds and peak GPU memory
  profile  one update under torch.profiler: top CPU / CUDA ops, kernels per minibatch
  parity   the first K minibatches' gradients and one full update against the eager baseline (today's settings)

  PYTHONPATH=scripts:src .venv/bin/python scripts/learner_bench.py time --out reports/data/learner-speed/base.json
"""

import argparse
import collections
import contextlib
import copy
import json
import multiprocessing
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from swccg_policy import v4_rl  # noqa: E402
from swccg_policy.learner_speed import Prefetch, compile_tables, eager_tables, single_tables  # noqa: E402
from swccg_policy.ppo_step import PPOConfig, backward  # noqa: E402
from swccg_policy.trajectory import trajectory_files  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402

CHECKPOINT = ROOT / 'runs/dojo/replay-a/final.pt'
DATA = ROOT / 'runs/dojo/collect-a/trajectories'
# The arms' learner settings (scripts/critic_long_chain.sh CRITIC, scripts/v31_stage.py).
ARMS = dict(minibatch=64, infer_batch=96, actor_tokens=32768, critic_tokens=32768, lr=5e-5, value_lr=3e-4,
            max_grad_norm=0.5, entropy_coef=0.01, value_lambda=0.995, win_coef=0.5, value_weight_decay=0.01,
            critic_accumulate=4, critic_fp32=True)
# Plan learner-minibatch: the ctx5-n1 learner (scripts/context_v5_server_chain.sh; v4_rl defaults for the critic).
RECIPES = {'arms': ARMS, 'v5': {**ARMS, 'value_lambda': None, 'win_coef': 0.0, 'value_weight_decay': 0.0,
                                'critic_accumulate': 1, 'critic_fp32': False}}


def recipe(args):
    """The recipe's settings with this run's minibatch; the token caps keep a minibatch in one microbatch."""
    tokens = args.tokens or ARMS['actor_tokens'] * args.minibatch // ARMS['minibatch']
    return {**RECIPES[args.recipe], 'minibatch': args.minibatch, 'actor_tokens': tokens, 'critic_tokens': tokens}


def batch_files(folder, steps):
    """The first files (sorted) whose sidecars add up to at least `steps` steps."""
    files, count = [], 0
    for path in sorted(trajectory_files(folder)):
        files.append(path)
        count += int(json.loads(path.with_name(path.name + '.meta.json').read_text())['steps'])
        if count >= steps:
            break
    return files, count


def setup(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


class Bench:
    """The learner's state for one checkpoint: models, optimizers, config and the decoded batch."""

    def __init__(self, args):
        self.args = args
        setup(args.seed)
        self.recipe = R = recipe(args)
        self.actor, self.critic, self.tables, checkpoint = load(args.checkpoint, 'cuda')
        self.reader = TrajectoryReader(self.tables, [])
        self.actor.clear_tables()
        self.critic.encoder.clear_tables()
        self.actor.packed_attention = True
        self.critic.encoder.packed_attention = not R['critic_fp32']  # fp32 critic: standard attention
        self.config = PPOConfig(actor_tokens=R['actor_tokens'], critic_tokens=R['critic_tokens'],
                                actor_clip=R['max_grad_norm'], bf16=True, entropy=R['entropy_coef'],
                                critic_bf16=False if R['critic_fp32'] else None, win_coef=R['win_coef'],
                                **options_config(args))
        self.optimizers = [torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0, fused=True)
                           for parameters, lr in ((self.actor.parameters(), R['lr']),
                                                  (v4_rl.critic_groups(self.critic, R['value_weight_decay']),
                                                   R['value_lr']))]
        for optimizer, state in zip(self.optimizers, checkpoint.get('optimizer_state', [])):
            optimizer.load_state_dict(state)
        del checkpoint
        if args.single_tables:
            single_tables(self.actor, self.critic)
        if args.compile == 'tables':
            compile_tables(self.actor, self.critic)
        self.files, self.file_steps = batch_files(args.data, args.steps)
        self.initial = self.snapshot()
        self.games = None

    def snapshot(self):
        return {'actor': copy.deepcopy(self.actor.state_dict()), 'critic': copy.deepcopy(self.critic.state_dict()),
                'optimizers': [copy.deepcopy(o.state_dict()) for o in self.optimizers]}

    def restore(self, state=None):
        state = state or self.initial
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        for optimizer, saved in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(copy.deepcopy(saved))

    def pool(self, workers):
        return (multiprocessing.get_context('spawn').Pool(workers, initializer=v4_rl._init_reader,
                                                          initargs=(self.reader,)) if workers > 1 else None)

    def load(self, files, pool):
        if pool is not None:
            return [game for loaded in pool.map(v4_rl._load_file, files) for game in loaded]
        return [game for path in files for game in self.reader.load(path)]

    def decode(self, workers):
        started = time.monotonic()
        pool = self.pool(workers)
        try:
            self.games = self.load(self.files, pool)
        finally:
            if pool is not None:
                pool.close()
        return time.monotonic() - started

    def flatten(self):
        games = self.games
        self.items = [item for game in games for item in game['items']]
        self.critic_items = [item for game in games for item in game['critic_items']]
        self.old_logp = np.concatenate([game['old_logp'] for game in games])
        self.sequences = [sequence for game in games for sequence in game['seqs']]
        self.exclusions = [excluded for game in games for excluded in game['excluded']]
        self.lengths = np.asarray([max(len(item['tok']), len(privileged['tok']))
                                   for item, privileged in zip(self.items, self.critic_items, strict=True)])
        self.wins = v4_rl.win_labels(games)

    def values(self):
        R = self.recipe
        if R['win_coef'] > 0:
            values, _ = v4_rl.predict_values(self.critic, self.critic_items, self.config, R['infer_batch'],
                                             with_win=True)
        else:
            values = v4_rl.predict_values(self.critic, self.critic_items, self.config, R['infer_batch'])
        self.advantages, self.returns = v4_rl.advantages(self.games, values, value_lam=R['value_lambda'])
        return values

    def batches(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        return v4_rl.ppo_batches(self.lengths, self.recipe['minibatch'])

    def ppo(self, batches):
        statistics = collections.defaultdict(list)
        counts = collections.Counter()
        v4_rl.ppo_pass(self.actor, self.critic, self.optimizers, self.items, self.critic_items, batches,
                       self.sequences, self.exclusions, self.old_logp, self.advantages, self.returns, self.wins,
                       self.config, True, self.recipe['critic_accumulate'], statistics, counts)
        return statistics, counts


def options_config(args):
    """PPOConfig fields of the speed options (plan learner-speed S-1, S-3; S-2 compiles, S-4 is the decode)."""
    return {'critic_tf32': args.critic_tf32, 'critic_stream': args.critic_stream}


def timed(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = function()
    torch.cuda.synchronize()
    return result, time.monotonic() - started, torch.cuda.max_memory_allocated() / 2**30


def cmd_time(args):
    bench = Bench(args)
    result = {'files': len(bench.files), 'file_steps': bench.file_steps, 'options': options_label(args),
              'decode_serial_seconds': None, 'updates': []}
    result['decode_seconds'] = bench.decode(args.load_workers)
    bench.flatten()
    result['steps'] = len(bench.items)
    result['games'] = len(bench.games)
    pool = bench.pool(args.load_workers) if args.prefetch else None
    for update in range(args.updates):
        if args.prefetch:
            # S-4: the next batch (the same files) decodes while this update runs.
            done = threading.Event()
            prefetch = Prefetch(None, args.steps, lambda _: bench.files,
                                lambda files: bench.load(files, pool), done.is_set)
        _, values_seconds, values_memory = timed(bench.values)
        batches = bench.batches(args.seed + update)
        (statistics, counts), ppo_seconds, ppo_memory = timed(lambda: bench.ppo(batches))
        extra = {}
        if args.prefetch:
            waited = time.monotonic()
            item = prefetch.get()
            done.set()
            extra = {'prefetch_wait_seconds': time.monotonic() - waited, 'prefetch_decode_seconds': item['decode_seconds'],
                     'prefetch_identical': same_games(item['games'], bench.games)}
        record = {'update': update, 'values_gae_seconds': values_seconds, 'ppo_seconds': ppo_seconds, **extra,
                  'values_peak_gib': values_memory, 'ppo_peak_gib': ppo_memory, 'minibatches': len(batches),
                  'microbatches': dict(counts),
                  'losses': {name: float(torch.stack(values).float().mean()) for name, values in statistics.items()
                             if name in ('pg', 'vloss', 'win_loss', 'entropy')}}
        result['updates'].append(record)
        print(json.dumps(record), flush=True)
    timed_updates = result['updates'][args.warmup:] or result['updates']
    result['summary'] = {key: float(np.median([u[key] for u in timed_updates]))
                         for key in ('values_gae_seconds', 'ppo_seconds')}
    if pool is not None:
        pool.close()
    decode = (float(np.median([u['prefetch_wait_seconds'] for u in timed_updates])) if args.prefetch
              else result['decode_seconds'])
    result['summary']['decode_seconds'] = decode
    result['summary']['learner_seconds'] = (result['summary']['values_gae_seconds'] + result['summary']['ppo_seconds']
                                            + decode)
    if args.prefetch:
        result['summary']['prefetch_identical'] = all(u['prefetch_identical'] for u in result['updates'])
    result['summary']['peak_gib'] = max(max(u['values_peak_gib'], u['ppo_peak_gib']) for u in result['updates'])
    print(json.dumps(result['summary']), flush=True)
    write(args.out, result)


def cmd_profile(args):
    from torch.profiler import ProfilerActivity, profile
    bench = Bench(args)
    bench.decode(args.load_workers)
    bench.flatten()
    bench.values()
    batches = bench.batches(args.seed)
    bench.ppo(batches[:8])  # warm-up (allocator, cuBLAS handles, compiled graphs)
    bench.restore()
    bench.values()
    selected = batches[:args.profile_minibatches]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        _, seconds, _ = timed(lambda: bench.ppo(selected))
    events = profiler.key_averages()
    kernels = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    table = {'cpu': events.table(sort_by='self_cpu_time_total', row_limit=args.rows),
             'cuda': events.table(sort_by='self_device_time_total', row_limit=args.rows)}
    device_total = sum(event.time_range.elapsed_us() for event in kernels) / 1e6  # kernel time only
    result = {'options': options_label(args), 'minibatches': len(selected), 'seconds': seconds,
              'seconds_per_minibatch': seconds / len(selected), 'kernels_per_minibatch': len(kernels) / len(selected),
              'device_busy_seconds': device_total, 'device_busy_fraction': device_total / seconds,
              'top_cpu': [{'name': e.key, 'self_cpu_ms': e.self_cpu_time_total / 1e3, 'calls': e.count}
                          for e in sorted(events, key=lambda e: -e.self_cpu_time_total)[:args.rows]],
              'top_cuda': [{'name': e.key, 'self_cuda_ms': e.self_device_time_total / 1e3, 'calls': e.count}
                           for e in sorted(events, key=lambda e: -e.self_device_time_total)[:args.rows]]}
    print(table['cpu'])
    print(table['cuda'])
    print(json.dumps({k: v for k, v in result.items() if not k.startswith('top')}), flush=True)
    if args.out:
        write(args.out, result)
        args.out.with_suffix('.txt').write_text(table['cpu'] + '\n' + table['cuda'] + '\n')


def gradients(models):
    return {role: torch.cat([p.grad.detach().float().flatten() if p.grad is not None
                             else torch.zeros(p.numel(), device=p.device) for p in model.parameters()])
            for role, model in models.items()}


def parameters(models):
    return {role: torch.cat([p.detach().float().flatten() for p in model.parameters()]) for role, model in models.items()}


def compare(left, right):
    return {'cos': float(torch.nn.functional.cosine_similarity(left, right, 0)),
            'norm_ratio': float(left.norm() / right.norm()) if right.norm() > 0 else None}


def minibatch_gradients(bench, indices):
    bench.actor.zero_grad(set_to_none=True)
    bench.critic.zero_grad(set_to_none=True)
    backward(bench.actor, bench.critic, bench.items, bench.critic_items, indices, bench.sequences, bench.exclusions,
             bench.old_logp, bench.advantages, bench.returns, stack, bench.config, policy_on=True, wins=bench.wins)
    return gradients({'actor': bench.actor, 'critic': bench.critic})


def cmd_parity(args):
    """Gradients of the first K minibatches and one full update: options (this run's flags) vs the eager baseline.
    The baseline is also repeated once: its own run-to-run noise (atomics in index_add / scatter) is the floor."""
    bench = Bench(args)
    bench.decode(args.load_workers)
    bench.flatten()
    bench.values()
    batches = bench.batches(args.seed)
    option_config = bench.config
    base_config = PPOConfig(**{**option_config.__dict__, **{key: PPOConfig.__dataclass_fields__[key].default
                                                            for key in options_config(args)}})
    models = {'actor': bench.actor, 'critic': bench.critic}
    cells = collections.defaultdict(list)
    for indices in batches[:args.minibatches]:
        bench.config = base_config
        with eager_tables():
            reference = minibatch_gradients(bench, indices)
            repeat = minibatch_gradients(bench, indices)
        bench.config = option_config
        candidate = minibatch_gradients(bench, indices)
        for role in models:
            for label, grads in (('repeat', repeat), ('option', candidate)):
                for key, value in compare(grads[role], reference[role]).items():
                    cells[f'{role}:{label}:{key}'].append(value)
    bench.actor.zero_grad(set_to_none=True)
    bench.critic.zero_grad(set_to_none=True)
    start = parameters(models)
    updates = {}
    for label, config in (('base', base_config), ('repeat', base_config), ('option', option_config)):
        bench.restore()
        bench.config = config
        with eager_tables() if label != 'option' else contextlib.nullcontext():
            statistics, _ = bench.ppo(batches)
        updates[label] = {'delta': {role: value - start[role] for role, value in parameters(models).items()},
                          'losses': {name: float(torch.stack(values).float().mean())
                                     for name, values in statistics.items() if name in ('pg', 'vloss', 'win_loss')}}
    result = {'options': options_label(args), 'minibatches': args.minibatches,
              'gradients': {key: {'median': float(np.median(v)), 'min': float(np.min(v))} for key, v in cells.items()},
              'update': {label: {**{f'{role}:delta:{k}': v for role in models
                                    for k, v in compare(updates[label]['delta'][role],
                                                        updates['base']['delta'][role]).items()},
                                 'losses': updates[label]['losses']} for label in ('repeat', 'option')},
              'base_losses': updates['base']['losses']}
    print(json.dumps(result, indent=1), flush=True)
    write(args.out, result)


def options_label(args):
    labels = [label for label, on in (('tf32', args.critic_tf32), (f'compile-{args.compile}', args.compile != 'off'),
                                      ('stream', args.critic_stream), ('prefetch', args.prefetch),
                                      ('single', args.single_tables)) if on]
    if args.recipe != 'arms' or args.minibatch != ARMS['minibatch'] or args.tokens:
        labels += [args.recipe, f'mb{args.minibatch}'] + ([f'tok{args.tokens}'] if args.tokens else [])
    return '+'.join(labels) or 'base'


def same_games(left, right):
    """S-4 parity: the decoded games are identical (every array of every item, and the per-game fields)."""
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.keys() != b.keys() or len(a['items']) != len(b['items']):
            return False
        if not np.array_equal(a['old_logp'], b['old_logp']) or a['reward'] != b['reward'] or a['seqs'] != b['seqs']:
            return False
        for role in ('items', 'critic_items'):
            for x, y in zip(a[role], b[role]):
                if x.keys() != y.keys() or not all(np.array_equal(x[k], y[k]) for k in x):
                    return False
    return True


def write(path, result):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['time', 'profile', 'parity'])
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=883040)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1, help='updates left out of the summary median')
    parser.add_argument('--load-workers', type=int, default=4)
    parser.add_argument('--profile-minibatches', type=int, default=24)
    parser.add_argument('--rows', type=int, default=30)
    parser.add_argument('--minibatches', type=int, default=8, help='parity: minibatches with gradient comparisons')
    parser.add_argument('--out', type=Path)
    # Plan learner-speed options (v4_rl has the same flags).
    parser.add_argument('--critic-tf32', action='store_true', help='S-1')
    parser.add_argument('--compile', choices=['off', 'tables'], default='off', help='S-2')
    parser.add_argument('--critic-stream', action='store_true', help='S-3')
    parser.add_argument('--prefetch', action='store_true', help='S-4 (time only)')
    parser.add_argument('--single-tables', action='store_true', help='S-5 (not in the plan): one GEMM per table layer')
    # Plan learner-minibatch.
    parser.add_argument('--recipe', choices=sorted(RECIPES), default='arms')
    parser.add_argument('--minibatch', type=int, default=ARMS['minibatch'])
    parser.add_argument('--tokens', type=int, default=0, help='microbatch token cap (default 512 x minibatch)')
    args = parser.parse_args()
    {'time': cmd_time, 'profile': cmd_profile, 'parity': cmd_parity}[args.command](args)


if __name__ == '__main__':
    main()
