"""Plan precision-capacity-check: is bf16 enough, and is the 6-layer V3.1 network capacity-limited at this data scale?

    v31_precision_capacity.py bf16        [--n 6000] [--minibatches 24]
    v31_precision_capacity.py critic-fit  [--minutes 30]
    v31_precision_capacity.py actor-fit   [--minutes 30] [--train-decisions 50000]
    v31_precision_capacity.py report

Common options: --checkpoint runs/dojo/focus-s2/final.pt, --data runs/dojo/collect-s2/trajectories (split by whole
files: every 7th file is held out), --shifted runs/dojo/n2-train8h/trajectories (a second, distribution-shifted held-out
set), --out reports/data/precision-capacity.

P-1 bf16: the actor (raw = pre-cap logits, from the parent forward; capped = cap * tanh(raw / cap)) and the critic in
fp32 (TF32 off) vs bf16 autocast, unpacked and packed attention (the learner's path): top-1 agreement, total variation,
|delta log p| of the recorded action (the PPO ratio error), value |delta|, per decision type and top-2 gap bin; raw logit
size per type; gradient cosine / norm ratio on PPO-style minibatches (policy + value loss as the learner), with the
fp32-vs-fp32 noise floor.
P-2 critic fit: the s2 critic trained further in fp32 on the train files (targets: GAE returns from the s2 critic, as
the learner computes them); train / held-out / shifted MSE and explained variance over time.
P-3 actor fit: the s2 actor trained in fp32 to imitate the recorded actions of a fixed train set (NLL); the entropy floor
is mean(-old_logp) (actions were sampled from the behaviour policy, so that is an unbiased estimate of the expected NLL);
train / held-out NLL and top-1 accuracy, the latter on confident decisions (behaviour probability >= 0.9).
Nothing here trains a run: weights are copies in memory; nothing is written to runs/.
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from swccg_policy.ppo_step import microbatches, policy_loss, sequence_tensors, PPOConfig  # noqa: E402
from swccg_policy.rl import seq_logprob  # noqa: E402
from swccg_policy.trajectory import trajectory_files  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402
from swccg_policy.v4_rl import advantages, predict_values  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402

GAP_BINS = ((0.1, '<0.1'), (0.5, '<0.5'), (1.0, '<1'), (float('inf'), '>=1'))
TOKENS = 32768


def setup(seed=920001):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_files(folder):
    files = sorted(trajectory_files(folder))
    return [f for i, f in enumerate(files) if i % 7], [f for i, f in enumerate(files) if i % 7 == 0]


class Decisions:
    """Flattened decisions of some games: items, critic items, sequences, exclusions, old logp, types, returns."""

    def __init__(self, games, critic=None, critic_config=None, targets='gae'):
        self.items = [item for game in games for item in game['items']]
        self.critic_items = [item for game in games for item in game['critic_items']]
        self.seqs = [seq for game in games for seq in game['seqs']]
        self.excluded = [excluded for game in games for excluded in game['excluded']]
        self.old_logp = np.concatenate([game['old_logp'] for game in games]).astype(np.float32)
        self.types = [kind for game in games for kind in game['types']]
        self.returns = self.adv = None
        if targets == 'mc':
            # Monte Carlo reward-to-go (gamma 1, lambda 1): the same targets for every critic compared.
            self.adv, self.returns = advantages(games, np.zeros(len(self.items), np.float32), lam=1.0)
            self.adv = self.adv.astype(np.float32)
            self.returns = self.returns.astype(np.float32)
        elif critic is not None:
            values = predict_values(critic, self.critic_items, critic_config, 96)
            self.adv, self.returns = advantages(games, values)
            self.adv = self.adv.astype(np.float32)
            self.returns = self.returns.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def subset(self, indices):
        new = copy.copy(self)
        for name in ('items', 'critic_items', 'seqs', 'excluded', 'types'):
            setattr(new, name, [getattr(self, name)[i] for i in indices])
        for name in ('old_logp', 'adv', 'returns'):
            value = getattr(self, name)
            setattr(new, name, None if value is None else value[indices])
        return new


def read_games(reader, files, limit=None):
    games, count = [], 0
    for path in files:
        for game in reader.load(path):
            games.append(game)
            count += len(game['items'])
        if limit and count >= limit:
            break
    return games


def actor_forward(actor, batch, mode):
    """(raw, capped) logits as fp32 tensors; mode fp32 | bf16 | bf16-packed."""
    actor.packed_attention = mode == 'bf16-packed'
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode != 'fp32'):
        raw = super(type(actor), actor).forward(batch)
    raw = raw.float()
    cap = actor.cfg.logit_cap
    capped = torch.where(batch['omask'], cap * torch.tanh(raw.masked_fill(~batch['omask'], 0.0) / cap), raw) if cap else raw
    return raw, capped


def critic_forward(critic, batch, mode):
    critic.encoder.packed_attention = mode == 'bf16-packed'
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode != 'fp32'):
        return critic(batch).float()


def gap_bin(gap):
    return next(label for limit, label in GAP_BINS if gap < limit)


# ---------------------------------------------------------------- P-1

def run_bf16(args, actor, critic, reader):
    train, held = split_files(args.data)
    # Plan night-integer-fp32: --files all (nothing is trained here, so every kept file is usable).
    games = read_games(reader, sorted(train + held) if args.files == 'all' else held, args.n)
    data = Decisions(games, critic, PPOConfig(bf16=False))
    order = np.arange(min(args.n, len(data)))
    modes = ('fp32-repeat', 'bf16', 'bf16-packed')
    rows = []  # per decision: type, gap, raw max, and per mode: top1 agree, tv, |dlogp|, |dvalue|
    with torch.no_grad():
        for indices in microbatches(order, data.items, TOKENS):
            batch = stack([data.items[i] for i in indices], 'cuda')
            cbatch = stack([data.critic_items[i] for i in indices], 'cuda')
            picks, lengths, excluded = sequence_tensors(indices, data.seqs, data.excluded, batch['omask'])
            none = batch['opt'][:, :, 5] > 0
            ref_raw, ref = actor_forward(actor, batch, 'fp32')
            ref_logp = seq_logprob(ref, batch['omask'], none, picks, lengths, excluded, sync_free=True)
            ref_p = torch.softmax(ref.masked_fill(~batch['omask'], -1e9), 1)
            ref_v = critic_forward(critic, cbatch, 'fp32')
            masked = ref.masked_fill(~batch['omask'], -1e9)
            top2 = masked.topk(2, dim=1).values if masked.shape[1] > 1 else masked.repeat(1, 2)
            raw_max = ref_raw.masked_fill(~batch['omask'], 0).abs().max(1).values
            per = []
            for mode in modes:
                raw, logits = actor_forward(actor, batch, 'fp32' if mode == 'fp32-repeat' else mode)
                logp = seq_logprob(logits, batch['omask'], none, picks, lengths, excluded, sync_free=True)
                p = torch.softmax(logits.masked_fill(~batch['omask'], -1e9), 1)
                v = critic_forward(critic, cbatch, 'fp32' if mode == 'fp32-repeat' else mode)
                per.append((
                    (logits.masked_fill(~batch['omask'], -1e9).argmax(1) == masked.argmax(1)).cpu().numpy(),
                    (0.5 * (p - ref_p).abs().sum(1)).cpu().numpy(),
                    (logp - ref_logp).abs().cpu().numpy(),
                    (v - ref_v).abs().cpu().numpy()))
            options = batch['omask'].sum(1).cpu().numpy()
            gaps = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            for row, index in enumerate(indices):
                rows.append({'type': data.types[index], 'options': int(options[row]),
                             'gap': float(gaps[row]) if options[row] > 1 else None, 'raw_max': float(raw_max[row]),
                             **{f'{mode}:{name}': float(per[m][k][row]) for m, mode in enumerate(modes)
                                for k, name in enumerate(('agree', 'tv', 'dlogp', 'dvalue'))}})
    summary = summarize_bf16(rows, modes)
    summary['gradients'] = gradients(args, actor, critic, data)
    summary['decisions'] = len(rows)
    return summary


def summarize_bf16(rows, modes):
    def stats(selected):
        out = {'n': len(selected)}
        for mode in modes:
            for name in ('agree', 'tv', 'dlogp', 'dvalue'):
                values = np.array([r[f'{mode}:{name}'] for r in selected])
                if len(values):
                    out[f'{mode}:{name}'] = float(values.mean())
                    if name in ('dlogp', 'tv'):
                        out[f'{mode}:{name}_p99'] = float(np.quantile(values, 0.99))
        return out
    multi = [r for r in rows if r['gap'] is not None]
    result = {'all': stats(rows), 'multi_option': stats(multi),
              'by_gap': {label: stats([r for r in multi if gap_bin(r['gap']) == label]) for _, label in GAP_BINS},
              'by_type': {}}
    for kind in sorted({r['type'] for r in rows}):
        selected = [r for r in rows if r['type'] == kind]
        raw = np.array([r['raw_max'] for r in selected])
        result['by_type'][kind] = {**stats(selected), 'raw_max_median': float(np.median(raw)),
                                   'raw_max_p99': float(np.quantile(raw, 0.99)), 'raw_max_max': float(raw.max()),
                                   'share_raw_over_30': float((raw > 30).mean())}
    return result


FAMILIES = {'all': None, 'integer': ('INTEGER',), 'card': ('CARD_SELECTION', 'ARBITRARY_CARDS')}
INTEGER_BRANCH = ('option_value.', 'option_scalars.')


def gradients(args, actor, critic, data):
    """Cosine and norm ratio of bf16 (unpacked and packed) vs fp32 gradients; fp32 vs fp32 as the noise floor.
    Plan night-integer-fp32: per decision family (all; integer; card choice: minibatches drawn from that family only)
    and, besides the whole actor and critic, the actor's integer branch (option_value / option_scalars) alone."""
    config = PPOConfig(bf16=False)
    rng = np.random.default_rng(920002)
    out = defaultdict(list)
    branch = [p for name, p in actor.named_parameters() if name.startswith(INTEGER_BRANCH)]
    for family, kinds in FAMILIES.items():
        pool = np.arange(len(data)) if kinds is None else np.array([i for i, kind in enumerate(data.types) if kind in kinds])
        if len(pool) < 16:
            continue
        previous = None  # fp32 gradients of the previous (different) minibatch: the SGD sampling-noise reference
        for _ in range(args.minibatches if kinds is None else max(1, args.minibatches * 2 // 3)):
            indices = rng.choice(pool, size=min(64, len(pool)), replace=False)
            grads = {}
            for run in ('fp32-a', 'fp32-b', 'bf16', 'bf16-packed'):
                mode = 'fp32' if run.startswith('fp32') else run
                # --critic-mode fp32: the learner's --critic-fp32 path (actor bf16-packed, critic fp32 unpacked).
                critic_mode = 'fp32' if run.startswith('fp32') or args.critic_mode == 'fp32' else run
                actor.zero_grad(set_to_none=True)
                critic.zero_grad(set_to_none=True)
                for selected in microbatches(indices, data.items, TOKENS):
                    batch = stack([data.items[i] for i in selected], 'cuda')
                    picks, lengths, excluded = sequence_tensors(selected, data.seqs, data.excluded, batch['omask'])
                    _, logits = actor_forward(actor, batch, mode)
                    loss, _ = policy_loss(logits, batch, picks, lengths, excluded,
                                          torch.as_tensor(data.old_logp[selected], device='cuda'),
                                          torch.as_tensor(data.adv[selected], device='cuda'), config)
                    (loss * len(selected) / len(indices)).backward()
                for selected in microbatches(indices, data.critic_items, TOKENS):
                    cbatch = stack([data.critic_items[i] for i in selected], 'cuda')
                    value = critic_forward(critic, cbatch, critic_mode)
                    loss = F.mse_loss(value, torch.as_tensor(data.returns[selected], device='cuda'))
                    (loss * len(selected) / len(indices)).backward()
                grads[run] = {name: torch.cat([p.grad.flatten().float() for p in parameters if p.grad is not None])
                              for name, parameters in (('actor', list(actor.parameters())),
                                                       ('critic', list(critic.parameters())),
                                                       ('actor_integer_branch', branch))
                              if any(p.grad is not None for p in parameters)}
            names = [name for name in grads['fp32-a'] if grads['fp32-a'][name].norm() > 0]
            if previous is not None:
                for name in names:
                    if name in previous:
                        out[f'{family}:{name}:cos_between_minibatches_fp32'].append(
                            float(F.cosine_similarity(grads['fp32-a'][name], previous[name], 0)))
            previous = grads['fp32-a']
            for name in names:
                a, b = grads['fp32-a'][name], grads['fp32-b'][name]
                out[f'{family}:{name}:cos_fp32_fp32'].append(float(F.cosine_similarity(a, b, 0)))
                out[f'{family}:{name}:fp32_norm'].append(float(a.norm()))
                for run in ('bf16', 'bf16-packed'):
                    c = grads[run][name]
                    out[f'{family}:{name}:cos_{run}_fp32'].append(float(F.cosine_similarity(c, a, 0)))
                    out[f'{family}:{name}:norm_ratio_{run}_fp32'].append(float(c.norm() / a.norm()))
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    return {key: {'median': float(np.median(v)), 'min': float(np.min(v)), 'max': float(np.max(v)), 'n': len(v)}
            for key, v in out.items()}


# ---------------------------------------------------------------- P-2 / P-3

def ev(pred, target):
    return float(1 - np.var(target - pred) / max(np.var(target), 1e-12))


def critic_fit(args, actor, critic, reader):
    train_files, held = split_files(args.data)
    frozen = copy.deepcopy(critic).eval()
    config = PPOConfig(bf16=False)
    # Fixed evaluation sets (in RAM, ~5k decisions each); training streams the train files.
    targets = args.targets
    probes = {'train': Decisions(read_games(reader, train_files[:8], 5000), frozen, config, targets),
              'heldout': Decisions(read_games(reader, held, args.eval_decisions), frozen, config, targets)}
    if args.shifted:
        probes['shifted'] = Decisions(read_games(reader, sorted(trajectory_files(args.shifted)), args.eval_decisions),
                                      frozen, config, targets)
    model = copy.deepcopy(critic).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 3e-4, weight_decay=0.0)

    def evaluate():
        out = {}
        with torch.no_grad():
            for name, data in probes.items():
                pred = predict_values(model, data.critic_items, config, 96)
                out[name] = {'mse': float(np.mean((pred - data.returns) ** 2)), 'ev': ev(pred, data.returns)}
        return out
    curve = [{'step': 0, 'seconds': 0.0, 'decisions_seen': 0, **evaluate()}]
    started, steps, seen, epoch = time.monotonic(), 0, 0, 0
    deadline = started + args.minutes * 60
    while time.monotonic() < deadline:
        epoch += 1
        files = train_files[:]
        random.shuffle(files)
        for path in files:
            if time.monotonic() >= deadline:
                break
            data = Decisions(reader.load(path), frozen, config)
            order = np.random.permutation(len(data))
            for offset in range(0, len(order), 64):
                indices = order[offset:offset + 64]
                optimizer.zero_grad(set_to_none=True)
                for selected in microbatches(indices, data.critic_items, TOKENS):
                    cbatch = stack([data.critic_items[i] for i in selected], 'cuda')
                    loss = F.mse_loss(model(cbatch).float(), torch.as_tensor(data.returns[selected], device='cuda'))
                    (loss * len(selected) / len(indices)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                seen += len(indices)
            if steps and (len(curve) == 1 or time.monotonic() - started >= len(curve) * args.eval_every * 60):
                model.eval()
                curve.append({'step': steps, 'seconds': time.monotonic() - started, 'decisions_seen': seen,
                              'epoch': epoch, **evaluate()})
                model.train()
                print(json.dumps(curve[-1]), flush=True)
    model.eval()
    curve.append({'step': steps, 'seconds': time.monotonic() - started, 'decisions_seen': seen, 'epoch': epoch, **evaluate()})
    return {'train_files': len(train_files), 'curve': curve}


def nll_and_accuracy(model, data, indices=None):
    indices = np.arange(len(data)) if indices is None else indices
    nll = np.zeros(len(indices), np.float32)
    correct = np.zeros(len(indices), bool)
    with torch.no_grad():
        position = 0
        for selected in microbatches(indices, data.items, TOKENS):
            batch = stack([data.items[i] for i in selected], 'cuda')
            picks, lengths, excluded = sequence_tensors(selected, data.seqs, data.excluded, batch['omask'])
            _, logits = actor_forward(model, batch, 'fp32')
            logp = seq_logprob(logits, batch['omask'], batch['opt'][:, :, 5] > 0, picks, lengths, excluded, sync_free=True)
            first = torch.as_tensor([data.seqs[i][0] if data.seqs[i] else -1 for i in selected], device='cuda')
            argmax = logits.masked_fill(~batch['omask'], -1e9).argmax(1)
            nll[position:position + len(selected)] = (-logp).cpu().numpy()
            correct[position:position + len(selected)] = (argmax == first).cpu().numpy()
            position += len(selected)
    return nll, correct


def actor_fit(args, actor, critic, reader):
    train_files, held = split_files(args.data)
    train = Decisions(read_games(reader, train_files, args.train_decisions))
    held_data = Decisions(read_games(reader, held, 5000))
    model = copy.deepcopy(actor).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 1e-4, weight_decay=0.0)
    probe = np.random.default_rng(920003).choice(len(train), size=min(5000, len(train)), replace=False)

    def evaluate():
        model.eval()
        out = {}
        for name, data, idx in (('train', train, probe), ('heldout', held_data, None)):
            nll, correct = nll_and_accuracy(model, data, idx)
            old = data.old_logp if idx is None else data.old_logp[idx]
            confident = old >= math.log(0.9)
            single = np.array([len(data.seqs[i]) == 1 for i in (idx if idx is not None else range(len(data)))])
            out[name] = {'nll': float(nll.mean()), 'entropy_floor': float((-old).mean()),
                         'top1_confident': float(correct[confident & single].mean()) if (confident & single).any() else None,
                         'top1_all': float(correct[single].mean())}
        model.train()
        return out
    curve = [{'step': 0, 'seconds': 0.0, **evaluate()}]
    print(json.dumps(curve[-1]), flush=True)
    started, steps, epoch = time.monotonic(), 0, 0
    deadline = started + args.minutes * 60
    while time.monotonic() < deadline:
        epoch += 1
        order = np.random.permutation(len(train))
        for offset in range(0, len(order), 64):
            if time.monotonic() >= deadline:
                break
            indices = order[offset:offset + 64]
            optimizer.zero_grad(set_to_none=True)
            for selected in microbatches(indices, train.items, TOKENS):
                batch = stack([train.items[i] for i in selected], 'cuda')
                picks, lengths, excluded = sequence_tensors(selected, train.seqs, train.excluded, batch['omask'])
                _, logits = actor_forward(model, batch, 'fp32')
                logp = seq_logprob(logits, batch['omask'], batch['opt'][:, :, 5] > 0, picks, lengths, excluded, sync_free=True)
                ((-logp).mean() * len(selected) / len(indices)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            steps += 1
            if time.monotonic() - started >= len(curve) * args.eval_every * 60:
                curve.append({'step': steps, 'seconds': time.monotonic() - started, 'epoch': epoch, **evaluate()})
                print(json.dumps(curve[-1]), flush=True)
    curve.append({'step': steps, 'seconds': time.monotonic() - started, 'epoch': epoch, **evaluate()})
    return {'train_decisions': len(train), 'heldout_decisions': len(held_data), 'curve': curve}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['bf16', 'critic-fit', 'actor-fit'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/dojo/focus-s2/final.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'runs/dojo/collect-s2/trajectories')
    parser.add_argument('--shifted', type=Path, default=ROOT / 'runs/dojo/n2-train8h/trajectories')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/data/precision-capacity')
    parser.add_argument('--n', type=int, default=6000)
    parser.add_argument('--minibatches', type=int, default=24)
    parser.add_argument('--minutes', type=float, default=30)
    parser.add_argument('--eval-every', type=float, default=5, help='minutes between evaluations')
    parser.add_argument('--train-decisions', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--critic-mode', choices=['bf16', 'fp32'], default='bf16',
                        help='bf16: gradients of the learner path today; fp32: the --critic-fp32 learner path')
    parser.add_argument('--name', help='output file stem (default: the command)')
    parser.add_argument('--targets', choices=['gae', 'mc'], default='gae',
                        help="critic-fit targets: 'gae' as the learner (from the critic's own values); 'mc' reward-to-go, "
                             "identical for every critic (use to compare critics; --minutes 0 = evaluation only)")
    parser.add_argument('--eval-decisions', type=int, default=5000)
    parser.add_argument('--files', choices=['heldout', 'all'], default='heldout',
                        help='bf16: decisions from the held-out files only (default) or from every file')
    args = parser.parse_args()
    setup()
    actor, critic, tables, checkpoint = load(args.checkpoint, 'cuda')
    actor.clear_tables()
    critic.encoder.clear_tables()
    reader = TrajectoryReader(tables, [])
    run = {'bf16': run_bf16, 'critic-fit': critic_fit, 'actor-fit': actor_fit}[args.command]
    started = time.time()
    result = run(args, actor, critic, reader)
    result.update({'checkpoint': str(args.checkpoint), 'data': str(args.data), 'iteration': checkpoint['rl_meta']['iteration'],
                   'seconds': time.time() - started, 'args': {k: str(v) for k, v in vars(args).items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or {'bf16': 'bf16', 'critic-fit': 'critic_fit', 'actor-fit': 'actor_fit'}[args.command]
    (args.out / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(args.out / f'{name}.json')


if __name__ == '__main__':
    main()
