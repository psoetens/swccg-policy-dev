"""Plan critic-ceiling stage B: train a critic offline on Monte-Carlo outcome targets of a frozen policy's play.

    critic_fit.py --checkpoint runs/critic-noise/D.pt --data runs/dojo/cn-e1/trajectories \
        --data runs/dojo/qc-collect/trajectories --view critic --out runs/critic-ceiling/cp.pt [--minutes 120]

Targets: the game result from the recorded seat's view (+1 win, -1 loss, 0 no winner), the same for every decision of
the game: no bootstrapping, no Dojo rewards. The critic starts from the checkpoint's critic and sees the privileged view
(`--view critic`, the critic items) or the actor's view (`--view actor`, the actor items through the same network).

Split (the same for every view): every 20th file is held out; the held-out files alternate between a selection half
(every 10 min: MSE and a checkpoint; the convergence test; the checkpoint choice) and a test half (written to the split
file for the readouts, never used here). Training: MSE, AdamW (no weight decay), a constant learning rate for
`--minutes - --decay-minutes`, then linear to 0; minibatches of `--minibatch` decisions in token-budget microbatches
(gradient accumulation), bf16 autocast with packed attention (as the learner's critic), gradient clip 1.0. Files are
decoded by `--workers` spawned processes ahead of the GPU.

Outputs: OUT (a full checkpoint, D's actor + the scored critic: the lowest selection MSE among the 10-min checkpoints
and the final one), OUT with suffix -final.pt (D's actor + the final critic), OUT.log.jsonl, OUT.split.json and
OUT.fit.json (selection MSE per checkpoint, the convergence test, the scored checkpoint).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import critic_diagnostics as cd  # noqa: E402
import critic_probes as cp  # noqa: E402
from swccg_policy.ppo_step import microbatches  # noqa: E402
from swccg_policy.trajectory import trajectory_files  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402
from swccg_policy.v4_rl import _load_file, load_pool  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402

TOKENS = 32768  # the learner's critic token budget (PPOConfig.critic_tokens)


def outcome_target(game):
    """+1 the recorded seat won, -1 it lost, 0 no winner."""
    return {1: 1.0, 0: -1.0, -1: 0.0}[cd.outcome(game)]


def split_files(folders):
    """(train, selection, test): every 20th file (sorted over all folders) held out, alternately selection / test."""
    files = sorted(str(path) for folder in folders for path in trajectory_files(folder))
    held = files[::20]
    return [path for index, path in enumerate(files) if index % 20], held[0::2], held[1::2]


def game_rows(games, view, copy=True):
    """[(item, target)] of the view's items of each game. copy False for games from the decoding processes (their
    unpickled arrays are already independent of the file buffer)."""
    key = 'critic_items' if view == 'critic' else 'items'
    return [(cd.detach(item) if copy else item, outcome_target(game)) for game in games for item in game[key]]


def learning_rate(peak, elapsed, budget, decay):
    """Constant until budget - decay, then linear to 0 at the budget."""
    start = budget - decay
    if elapsed <= start:
        return peak
    return peak * max(0.0, (budget - elapsed) / decay) if decay > 0 else 0.0


def converged(history, at, before, tolerance=0.01):
    """Plan §1: the selection MSE at minute `at` is < tolerance below its value at minute `before` (None if missing)."""
    marks = {round(row['minute']): row['mse'] for row in history}
    if at not in marks or before not in marks:
        return None
    return bool(marks[before] - marks[at] < tolerance * marks[before])


def scored(history):
    """The checkpoint with the lowest selection MSE (the earliest of equals)."""
    return min(history, key=lambda row: (row['mse'], row['minute']))


def values(critic, items):
    out = np.zeros(len(items), np.float32)
    with torch.no_grad():
        for selected in cp.batches(items):
            batch = stack([items[index] for index in selected], 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[selected] = critic(batch).float().cpu().numpy()
    return out


def selection_mse(critic, rows):
    critic.eval()
    predicted = values(critic, [row[0] for row in rows])
    critic.train()
    targets = np.array([row[1] for row in rows], np.float32)
    mse = float(((predicted - targets) ** 2).mean())
    return {'mse': round(mse, 6), 'ev': round(1 - mse / float(targets.var()), 5) if targets.var() > 0 else None}


def save_full(path, checkpoint, critic_state, meta):
    payload = {key: value for key, value in checkpoint.items() if key != 'optimizer_state'}
    payload['value_state'] = critic_state
    payload['critic_fit'] = meta
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', type=Path, action='append', help='trajectory folders (the 1-in-20 split)')
    parser.add_argument('--split', type=Path, help='plan critic-ceiling-2: an explicit split JSON {train, selection, '
                                                   'test} instead of --data')
    parser.add_argument('--view', choices=('critic', 'actor'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=120)
    parser.add_argument('--decay-minutes', type=float, default=20)
    parser.add_argument('--every-minutes', type=float, default=10, help='selection MSE and a checkpoint')
    parser.add_argument('--converge', type=int, nargs=2, default=[70, 100], metavar=('BEFORE', 'AT'))
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--chunk-files', type=int, default=120, help='files per shuffle buffer (~240 games)')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261020)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    actor, critic, tables, checkpoint = load(args.checkpoint, 'cuda')
    del actor
    critic.encoder.packed_attention = True
    critic.train()
    if (args.data is None) == (args.split is None):
        parser.error('give exactly one of --data and --split')
    if args.split:
        given = json.loads(args.split.read_text())
        train, selection, test = given['train'], given['selection'], given['test']
    else:
        train, selection, test = split_files(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix('')
    snapshots = Path(f'{stem}-snapshots')
    snapshots.mkdir(exist_ok=True)
    Path(f'{stem}.split.json').write_text(json.dumps({'train': train, 'selection': selection, 'test': test}))
    reader = TrajectoryReader(tables, [])
    pool = load_pool(reader, args.workers)
    load_many = (lambda paths: pool.map(_load_file, paths)) if pool else (lambda paths: [reader.load(p) for p in paths])
    selection_rows = [row for games in load_many(selection) for row in game_rows(games, args.view, copy=not pool)]
    log = Path(f'{stem}.log.jsonl').open('w')

    def write(record):
        log.write(json.dumps(record) + '\n')
        log.flush()

    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=0.0)
    budget, decay, every = args.minutes * 60, args.decay_minutes * 60, args.every_minutes * 60
    history = []

    def checkpoint_at(minute):
        result = {'minute': minute, **selection_mse(critic, selection_rows)}
        path = snapshots / f'm{minute:03d}.pt'
        torch.save({key: value.detach().cpu().clone() for key, value in critic.state_dict().items()}, path)
        history.append({**result, 'path': str(path)})
        write({'selection': result, 'step': step, 'decisions': seen})
        print(json.dumps({'minute': minute, **result, 'step': step}), flush=True)

    step, seen, epoch = 0, 0, 0
    write({'files': {'train': len(train), 'selection': len(selection), 'test': len(test)},
           'selection_decisions': len(selection_rows), 'view': args.view, 'args': {k: str(v) for k, v in vars(args).items()}})
    checkpoint_at(0)
    started = time.monotonic()
    next_mark, done = every, False
    while not done:
        order = train[:]
        random.Random(args.seed + epoch).shuffle(order)
        chunks = [order[offset:offset + args.chunk_files] for offset in range(0, len(order), args.chunk_files)]
        pending = pool.map_async(_load_file, chunks[0]) if pool else None
        for index, chunk in enumerate(chunks):
            games_list = pending.get() if pool else [reader.load(p) for p in chunk]
            if pool and index + 1 < len(chunks):
                pending = pool.map_async(_load_file, chunks[index + 1])
            buffer = [row for games in games_list for row in game_rows(games, args.view, copy=not pool)]
            del games_list
            random.Random(args.seed + epoch * 7919 + index).shuffle(buffer)
            for offset in range(0, len(buffer), args.minibatch):
                elapsed = time.monotonic() - started
                if elapsed >= budget:
                    done = True
                    break
                if elapsed >= next_mark:
                    checkpoint_at(round(next_mark / 60))
                    next_mark += every
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(args.lr, elapsed, budget, decay)
                part = buffer[offset:offset + args.minibatch]
                items = [row[0] for row in part]
                targets = np.array([row[1] for row in part], np.float32)
                optimizer.zero_grad(set_to_none=True)
                order_by_length = np.argsort([len(item['tok']) for item in items], kind='stable')
                total = 0.0
                for selected in microbatches(order_by_length, items, TOKENS):
                    batch = stack([items[i] for i in selected], 'cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = critic(batch)
                    loss = F.mse_loss(prediction.float(), torch.as_tensor(targets[selected], device='cuda'))
                    (loss * len(selected) / len(items)).backward()
                    total += float(loss.detach()) * len(selected) / len(items)
                if not np.isfinite(total):
                    raise FloatingPointError('nonfinite critic loss')
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                step += 1
                seen += len(items)
                if step % 200 == 0:
                    write({'step': step, 'decisions': seen, 'loss': round(total, 5), 'epoch': epoch,
                           'lr': optimizer.param_groups[0]['lr'], 'minutes': round(elapsed / 60, 2)})
            if done:
                break
        epoch += 1
    if pool:
        pool.terminate()
    final_minute = round(args.minutes)
    if not history or history[-1]['minute'] != final_minute:
        checkpoint_at(final_minute)
    best = scored(history)
    fit = {'view': args.view, 'steps': step, 'decisions': seen, 'epochs_started': epoch,
           'selection_history': [{k: row[k] for k in ('minute', 'mse', 'ev')} for row in history],
           'converged': converged(history, args.converge[1], args.converge[0]),
           'convergence_rule': f'selection MSE at {args.converge[1]} min < 1% below its value at {args.converge[0]} min',
           'scored_minute': best['minute'], 'scored_mse': best['mse'], 'final_minute': history[-1]['minute']}
    save_full(args.out, checkpoint, torch.load(best['path'], map_location='cpu'), {**fit, 'checkpoint': args.checkpoint})
    save_full(Path(f'{stem}-final.pt'), checkpoint, torch.load(history[-1]['path'], map_location='cpu'),
              {**fit, 'checkpoint': args.checkpoint, 'which': 'final'})
    Path(f'{stem}.fit.json').write_text(json.dumps(fit, indent=1))
    write({'fit': fit})
    log.close()
    print(json.dumps(fit))


if __name__ == '__main__':
    main()
