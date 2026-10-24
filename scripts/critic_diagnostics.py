"""Plan critic-diagnostics: does the V3.1 critic contribute? Offline diagnostics on fixed checkpoints (no RL training).

    critic_diagnostics.py split      writes <work>/eval_ids.txt (games of every 5th trajectory file = evaluation)
    critic_diagnostics.py values     --checkpoint X --name s2   per-decision fp32 values + Monte Carlo returns, eval games
    critic_diagnostics.py matchup    D-1: critics vs the matchup oracle (first decision, progress deciles, two views)
    critic_diagnostics.py grads      D-2: critic gradient numerics (modes x targets x minibatch size, per layer group)
    critic_diagnostics.py functional D-3: the same learner-style critic updates in bf16-packed / fp32 / fp32 at lr 1e-4
    critic_diagnostics.py pg-snr     D-4: actor policy-gradient signal-to-noise per baseline; plan critic-noise E1:
                                     --sizes 64 256 1024 4096 --n 98304 --all-files: noise scale B_noise + verdict
    critic_diagnostics.py train-mc   D-5: offline critic training against Monte Carlo returns (--critic-mode fp32|bf16-packed)
    critic_diagnostics.py screen     plan links-round2 phase 3: learner-style critic updates on fixed games, per variant
                                     (--value-lambda, --accumulate, --win-coef); held-out first-decision AUC / MC fit

Common: --data runs/dojo/collect-s2b/trajectories, --rollouts runs/dojo/collect-s2b/rollouts, --work runs/dojo/critic-diag,
--out reports/data/critic-diagnostics. Weights are copies in memory; nothing is written to the checkpoints.
Holding decoded decisions costs ~40 KB each, so evaluation games are streamed file by file and only what's needed is kept.
"""
import argparse
import collections
import copy
import csv
import json
import multiprocessing
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from swccg_policy.ppo_step import PPOConfig, critic_precision, microbatches, policy_loss, sequence_tensors  # noqa: E402
from swccg_policy.rl import seq_logprob  # noqa: E402
from swccg_policy.trajectory import FRAME, _read, trajectory_files  # noqa: E402
from swccg_policy.v4_checkpoint import load, save  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402
from swccg_policy.ppo_step import win_loss  # noqa: E402
from swccg_policy.v4_rl import advantages, critic_groups, predict_values  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402
from v31_precision_capacity import TOKENS, actor_forward, ev, setup  # noqa: E402

FP32 = PPOConfig(bf16=False)
MODES = ('fp32', 'fp32-packed', 'bf16', 'bf16-packed', 'fp32-noise', 'tf32')  # tf32: plan learner-speed S-1
VALUE_NOISE = 0.004  # the bf16 value-output |delta| measured in P-1


# ---------------------------------------------------------------- data

def split_files(folder):
    files = sorted(trajectory_files(folder))
    return [f for i, f in enumerate(files) if i % 5], [f for i, f in enumerate(files) if i % 5 == 0]


def scan_meta(path):
    """Game metadata of one trajectory file without decoding the tensors."""
    out = []
    with open(path, 'rb') as stream:
        _read(stream, 8 + 1 + 32)
        while header := stream.read(FRAME.size):
            metadata_size, _, stored_size, _ = FRAME.unpack(header)
            out.append(orjson.loads(_read(stream, metadata_size)))
            stream.seek(stored_size, 1)
    return out


def outcome(game):
    """1 = the recorded seat won, 0 = lost, -1 = no winner (cap / error)."""
    return int(game['w'] == game['s']) if game.get('w') in ('ds', 'ls') else -1


def detach(item):
    # Decoded arrays are views into the game's raw buffer; copy so a kept item doesn't pin the whole game.
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in item.items()}


def mc_returns(games):
    return advantages(games, np.zeros(sum(len(g['items']) for g in games), np.float32), lam=1.0)[1].astype(np.float32)


def read_decisions(reader, files, limit):
    games, count = [], 0
    for path in files:
        for game in reader.load(path):
            games.append(game)
            count += len(game['items'])
        if count >= limit:
            break
    return games


def rollout_rows(folder):
    rows = {}
    for path in sorted(Path(folder).glob('*.csv')):
        for row in csv.DictReader(path.open()):
            rows[row['game']] = row
    return rows


_DECODER = None  # plan critic-v2: the reader the forked decode workers use


class Prefetch:
    """Plan critic-v2: decode FILES with FUNCTION in POOL, at most DEPTH files ahead of the consumer (Pool.imap would
    decode everything ahead and hold it in memory). close() waits for the files in flight, so the pool can shut down
    without workers writing into a closed pipe."""

    def __init__(self, pool, function, files, depth):
        self.pool, self.function, self.files, self.pending = pool, function, iter(files), collections.deque()
        for _ in range(depth):
            self._submit()

    def _submit(self):
        path = next(self.files, None)
        if path is not None:
            self.pending.append(self.pool.apply_async(self.function, (path,)))

    def __iter__(self):
        while self.pending:
            result = self.pending.popleft().get()
            self._submit()
            yield result

    def close(self):
        self.files = iter(())
        while self.pending:
            self.pending.popleft().wait()


def decode_pool(reader, workers):
    """Plan critic-v2: a fork pool whose workers decode trajectory files with READER (numpy only, no CUDA)."""
    global _DECODER
    _DECODER = reader
    return multiprocessing.get_context('fork').Pool(workers)


def _sample_file(task):
    """Worker: sample_decisions for one file without GAE (counter = global decision index of its first decision)."""
    path, counter, stride = task
    keys = ('items', 'critic_items', 'seqs', 'excluded')
    out = {k: [] for k in keys} | {'old_logp': [], 'mc': [], 'game': [], 'step': []}
    games = _DECODER.load(path)
    mc = mc_returns(games)
    metas, position = [], 0
    for local, game in enumerate(games):
        for index in range(len(game['items'])):
            if counter % stride == 0:
                for k in keys:
                    value = game[k][index]
                    out[k].append(detach(value) if isinstance(value, dict) else value)
                out['old_logp'].append(float(game['old_logp'][index]))
                out['mc'].append(mc[position + index])
                out['game'].append(local)
                out['step'].append(index)
            counter += 1
        metas.append({'g': game['g'], 's': game['s'], 'won': outcome(game), 'n': len(game['items'])})
        position += len(game['items'])
    return out, metas


def _load_file(path):
    return _DECODER.load(path)


POOL = None  # plan critic-replay: the command's decode pool (--workers), used by every sampler below


def sample_decisions(reader, files, target, critic, pool=None, gae=True, critic_items=True):
    """About TARGET decisions spread evenly over all games of FILES (every k-th decision), detached, with their
    Monte Carlo returns and GAE returns from CRITIC (fp32, as the learner's targets), and their game's metadata.
    With gae=False and a decode pool, the workers sample the files themselves and GAE is NaN (fast path). With GAE,
    the pool decodes whole files ahead (bounded prefetch) while this process computes the critic's values.
    Plan critic-noise: 'value' is the critic's fp32 value of each sampled decision (NaN without GAE), and
    critic_items=False keeps no critic inputs (pg-snr needs only the values; halves the memory)."""
    pool = pool or POOL
    total = sum(meta['n_steps'] for path in files for meta in scan_meta(path))
    stride = max(1, total // target)
    keys = ('items', 'critic_items', 'seqs', 'excluded') if critic_items else ('items', 'seqs', 'excluded')
    out = {k: [] for k in keys} | {'old_logp': [], 'mc': [], 'gae': [], 'value': [], 'game': [], 'step': []}
    games_meta, counter = [], 0
    if pool is not None and not gae:
        tasks = []
        for path in files:
            tasks.append((path, counter, stride))
            counter += sum(meta['n_steps'] for meta in scan_meta(path))
        for part, metas in pool.imap(_sample_file, tasks):
            for k in part:
                if k in out:
                    out[k].extend([len(games_meta) + g for g in part[k]] if k == 'game' else part[k])
            out['gae'].extend([np.nan] * len(part['mc']))
            out['value'].extend([np.nan] * len(part['mc']))
            games_meta.extend(metas)
        files = []
    critic.encoder.packed_attention = False
    stream = (Prefetch(pool, _load_file, files, 2 * pool._processes) if pool is not None and files
              else (reader.load(path) for path in files))
    for games in stream:
        items = [item for game in games for item in game['critic_items']]
        values = predict_values(critic, items, FP32, 384, tokens=131072)
        mc = mc_returns(games)
        gae = advantages(games, values)[1].astype(np.float32)
        position = 0
        for game in games:
            for index in range(len(game['items'])):
                if counter % stride == 0:
                    for k in keys:
                        value = game[k][index]
                        out[k].append(detach(value) if isinstance(value, dict) else value)
                    out['old_logp'].append(float(game['old_logp'][index]))
                    out['mc'].append(mc[position + index])
                    out['gae'].append(gae[position + index])
                    out['value'].append(values[position + index])
                    out['game'].append(len(games_meta))
                    out['step'].append(index)
                counter += 1
            games_meta.append({'g': game['g'], 's': game['s'], 'won': outcome(game), 'n': len(game['items'])})
            position += len(game['items'])
        del games, items
    for k in ('old_logp', 'mc', 'gae', 'value'):
        out[k] = np.asarray(out[k], np.float32)
    out['game'] = np.asarray(out['game'])
    out['games'] = games_meta
    out['stride'] = stride
    return out


def oracle_seat(oracle, game, row):
    """Oracle P(the recorded seat wins) and its logit."""
    record = {'ds_deck': Path(row['ds_deck']).name, 'ls_deck': Path(row['ls_deck']).name,
              'ds_ai': row['ds_ai'], 'ls_ai': row['ls_ai']}
    p_ds = float(oracle.p_ds([record])[0])
    return p_ds if game['s'] == 'ds' else 1 - p_ds


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- critic modes

def critic_value(critic, batch, mode, generator=None):
    critic.encoder.packed_attention = mode.endswith('packed')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode.startswith('bf16')):
        value = critic(batch).float()
    if mode == 'fp32-noise':
        value = value + VALUE_NOISE * torch.randn(value.shape, device=value.device, generator=generator)
    return value


def critic_heads(critic, batch, mode):
    """Plan critic-v2: (value, win logit) under the same precision modes as critic_value."""
    critic.encoder.packed_attention = mode.endswith('packed')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode.startswith('bf16')):
        value, logits = critic.heads(batch)
    return value.float(), logits.float()


def group_of(name):
    if '.layers.' in name:
        return 'block' + name.split('.layers.')[1].split('.')[0]
    if name.startswith('value.'):
        return 'head'
    if name.startswith('encoder.cards') or name.startswith('encoder.context'):
        return 'card+context tables'
    return 'embed/other'


def critic_gradient(critic, items, returns, indices, mode, generator=None):
    critic.zero_grad(set_to_none=True)
    # Plan learner-speed S-1: 'tf32' is the fp32 critic with TF32 matmuls (the learner's --critic-tf32 scope).
    with critic_precision(PPOConfig(critic_tf32=mode == 'tf32')):
        for selected in microbatches(indices, items, TOKENS):
            batch = stack([items[i] for i in selected], 'cuda')
            value = critic_value(critic, batch, mode, generator)
            loss = F.mse_loss(value, torch.as_tensor(returns[selected], device='cuda'))
            (loss * len(selected) / len(indices)).backward()
    return {name: p.grad.detach().flatten().float().clone() for name, p in critic.named_parameters() if p.grad is not None}


def flat(grads):
    return torch.cat(list(grads.values()))


def cosine(a, b):
    return float(F.cosine_similarity(a, b, 0))


# ---------------------------------------------------------------- split / values

def cmd_split(args, reader=None):
    train, held = split_files(args.data)
    ids = [meta['g'] + ':' + meta['s'] for path in held for meta in scan_meta(path)]
    games = sorted({i.split(':')[0] for i in ids})
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / 'eval_ids.txt').write_text('\n'.join(games) + '\n')
    train_sides = sum(len(scan_meta(p)) for p in train)
    rows = rollout_rows(args.rollouts)
    matched = sum(g in rows for g in games)
    result = {'train_files': len(train), 'eval_files': len(held), 'eval_game_sides': len(ids), 'eval_games': len(games),
              'train_game_sides': train_sides, 'csv_games': len(rows), 'eval_games_matched_to_csv': matched,
              'match_fraction': matched / max(len(games), 1)}
    print(json.dumps(result))
    return result


def cmd_values(args, actor, critic, reader):
    _, held = split_files(args.data)
    critic.eval()
    critic.encoder.packed_attention = False
    values, returns, game_index, step, games_out = [], [], [], [], []
    for path in held:
        games = reader.load(path)
        items = [item for game in games for item in game['critic_items']]
        values.append(predict_values(critic, items, FP32, 96))
        returns.append(mc_returns(games))
        for game in games:
            n = len(game['items'])
            game_index.append(np.full(n, len(games_out), np.int32))
            step.append(np.arange(n, dtype=np.int32))
            games_out.append({'g': game['g'], 's': game['s'], 'won': outcome(game), 'n': n})
        del games, items
    path = args.work / f'values_{args.name}.npz'
    np.savez_compressed(path, value=np.concatenate(values), mc=np.concatenate(returns),
                        game=np.concatenate(game_index), step=np.concatenate(step),
                        games=np.array(json.dumps(games_out)))
    return {'decisions': int(sum(len(v) for v in values)), 'game_sides': len(games_out), 'file': str(path)}


# ---------------------------------------------------------------- D-1

def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def cv_combined_auc(y, columns, seed=940001):
    """Out-of-fold AUC and Brier of a logistic model on the given score columns (5-fold, stratified)."""
    x = np.column_stack(columns)
    p = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        p[test] = LogisticRegression(max_iter=1000).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, p)), float(np.mean((p - y) ** 2))


def bootstrap_diff(y, a, b, n=1000, seed=940002):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        index = rng.integers(0, len(y), len(y))
        if len(set(y[index])) == 2:
            diffs.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def matchup_view(y, oracle_p, first, deciles, mc_by_decile):
    o_logit = logit(oracle_p)
    out = {'games': int(len(y)), 'win_rate': float(y.mean()),
           'oracle': {'auc': auc(y, oracle_p), 'brier': float(np.mean((oracle_p - y) ** 2))}, 'critics': {}}
    for name, value in first.items():
        critic_auc = auc(y, value)
        combined_auc, combined_brier = cv_combined_auc(y, [value, o_logit])
        alone_auc, alone_brier = cv_combined_auc(y, [value])
        gain = combined_auc - critic_auc
        oracle_auc = out['oracle']['auc']
        verdict = ('knows matchups' if critic_auc >= oracle_auc - 0.02 and gain <= 0.02 else
                   'misses matchups' if critic_auc < oracle_auc - 0.05 or gain >= 0.05 else 'partial')
        out['critics'][name] = {
            'first_auc': critic_auc, 'first_brier_calibrated': alone_brier, 'first_value_sd': float(value.std()),
            'corr_with_oracle_logit': float(np.corrcoef(value, o_logit)[0, 1]),
            'combined_auc': combined_auc, 'oracle_gain': gain, 'combined_brier': combined_brier,
            'auc_minus_oracle_ci95': bootstrap_diff(y, value, oracle_p),
            'auc_by_decile': deciles[name], 'mc_by_decile': mc_by_decile[name], 'verdict': verdict}
    out['premise_oracle_auc_ge_0.65'] = out['oracle']['auc'] >= 0.65
    return out


def cmd_matchup(args, *_):
    with open(args.work / 'oracle.pkl', 'rb') as stream:
        oracle = pickle.load(stream)
    rows = rollout_rows(args.rollouts)
    loaded = {name: np.load(args.work / f'values_{name}.npz') for name in args.critics}
    games = json.loads(str(loaded[args.critics[0]]['games']))
    for name in args.critics[1:]:
        if json.loads(str(loaded[name]['games'])) != games:
            raise ValueError('critic value files cover different games')
    keep = [i for i, g in enumerate(games) if g['won'] >= 0 and g['g'] in rows]
    y = np.array([games[i]['won'] for i in keep])
    oracle_p = np.array([oracle_seat(oracle, games[i], rows[games[i]['g']]) for i in keep])
    opponent = [rows[games[i]['g']]['ls_ai' if games[i]['s'] == 'ds' else 'ds_ai'] for i in keep]
    pvp = np.array([o.startswith('policy') for o in opponent])
    keep_games = np.zeros(len(games), bool)
    keep_games[keep] = True
    pvp_games = np.zeros(len(games), bool)
    pvp_games[np.array(keep)[pvp]] = True
    views = {'all': (np.ones(len(y), bool), keep_games), 'policy_vs_policy': (pvp, pvp_games)}
    first, per_decision = {}, {}
    won_all = np.array([g['won'] for g in games])
    lengths = np.array([g['n'] for g in games])
    for name, data in loaded.items():
        game, step, value, mc = data['game'], data['step'], data['value'], data['mc']
        first[name] = value[np.flatnonzero(step == 0)][keep]
        per_decision[name] = (game, np.minimum((step / lengths[game] * 10).astype(int), 9), value, mc)
    result = {}
    for view, (mask, game_mask) in views.items():
        deciles, mc_deciles = {}, {}
        for name, (game, decile, value, mc) in per_decision.items():
            selected = game_mask[game]
            won = won_all[game]
            deciles[name], mc_deciles[name] = {}, {}
            for d in range(10):
                in_decile = selected & (decile == d)
                deciles[name][d] = auc(won[in_decile], value[in_decile])
                mc_deciles[name][d] = {'mse': float(np.mean((value[in_decile] - mc[in_decile]) ** 2)),
                                       'ev': ev(value[in_decile], mc[in_decile])}
        result[view] = matchup_view(y[mask], oracle_p[mask], {k: v[mask] for k, v in first.items()}, deciles, mc_deciles)
    result['excluded_no_winner_or_csv'] = len(games) - len(keep)
    return result


# ---------------------------------------------------------------- D-2

def cmd_grads(args, actor, critic, reader):
    _, held = split_files(args.data)
    sample = sample_decisions(reader, held, args.n, critic)
    items = sample['critic_items']
    targets = {'mc': sample['mc'], 'gae': sample['gae']}
    generator = torch.Generator(device='cuda').manual_seed(940003)
    rng = np.random.default_rng(940004)
    supported = {}
    for mode in args.modes or MODES:  # fp32-packed may be unsupported by the varlen attention kernel
        try:
            critic_value(critic, stack(items[:2], 'cuda'), mode, generator).sum().backward()
            supported[mode] = True
        except Exception as error:  # noqa: BLE001 - recorded in the result
            supported[mode] = repr(error)[:300]
    critic.zero_grad(set_to_none=True)
    result = {'decisions': len(items), 'supported': supported, 'cells': {}}
    for target, returns in targets.items():
        for size in args.sizes:
            cell = defaultdict(list)
            previous = None
            for _ in range(args.minibatches):
                indices = rng.choice(len(items), size=size, replace=False)
                reference = critic_gradient(critic, items, returns, indices, 'fp32')
                ref_flat = flat(reference)
                for mode in ('fp32',) + tuple(m for m in MODES if m != 'fp32' and supported.get(m) is True):
                    grads = critic_gradient(critic, items, returns, indices, mode, generator)
                    label = 'fp32-repeat' if mode == 'fp32' else mode
                    cell[f'{label}:cos'].append(cosine(flat(grads), ref_flat))
                    cell[f'{label}:norm_ratio'].append(float(flat(grads).norm() / ref_flat.norm()))
                    groups = defaultdict(list)
                    for name in reference:
                        groups[group_of(name)].append(name)
                    for group, names in groups.items():
                        a = torch.cat([grads[n] for n in names])
                        b = torch.cat([reference[n] for n in names])
                        if b.norm() > 0:
                            cell[f'{label}:cos:{group}'].append(cosine(a, b))
                    del grads
                if previous is not None:
                    cell['fp32:cos_between_minibatches'].append(cosine(ref_flat, previous))
                previous = ref_flat
            result['cells'][f'{target}:{size}'] = {k: {'median': float(np.median(v)), 'min': float(np.min(v)),
                                                       'max': float(np.max(v))} for k, v in cell.items()}
            print(target, size, json.dumps({k: round(v['median'], 4) for k, v in result['cells'][f'{target}:{size}'].items()
                                            if k.count(':') == 1}), flush=True)
    result['readout'] = grads_readout(result)
    return result


def grads_readout(result):
    cells = result['cells']

    def med(cell, key):
        return cells.get(cell, {}).get(key, {}).get('median')
    sizes = sorted({int(k.split(':')[1]) for k in cells})
    small, large = f'gae:{sizes[0]}', f'gae:{sizes[-1]}'
    fp32_packed = med(large, 'fp32-packed:cos')
    h4 = bool((fp32_packed is not None and fp32_packed < 0.99) or
              ((med(small, 'bf16:cos') or 0) >= 0.98 and (med(small, 'bf16-packed:cos') or 1) < 0.9))
    h2 = bool((med(f'mc:{sizes[0]}', 'bf16-packed:cos') or 0) >= 0.95 or
              (med(large, 'bf16-packed:cos') or 0) - (med(small, 'bf16-packed:cos') or 0) >= 0.1)
    defect = bool((med(f'mc:{sizes[-1]}', 'bf16:cos') or 1) < 0.9)
    return {'H4_packing_or_reduction': h4, 'H2_noise_dominated': h2, 'real_bf16_defect': defect,
            'learner_cell_bf16_packed_cos': med(small, 'bf16-packed:cos'),
            'noise_floor_fp32_repeat': med(small, 'fp32-repeat:cos')}


# ---------------------------------------------------------------- D-3

def learner_batches(lengths, minibatch, rng):
    """v4_rl's minibatching: permute, sort by length within groups of 16 minibatches, shuffle the minibatches."""
    permutation = rng.permutation(len(lengths))
    batches = []
    for offset in range(0, len(lengths), minibatch * 16):
        part = permutation[offset:offset + minibatch * 16]
        part = part[np.argsort(lengths[part], kind='stable')]
        batches.extend(part[i:i + minibatch] for i in range(0, len(part), minibatch))
    random.Random(int(rng.integers(1 << 30))).shuffle(batches)
    return batches


def _first_file(path):
    """Worker: (first critic item, won) of every game side with a winner in one file."""
    return [(detach(game['critic_items'][0]), outcome(game)) for game in _DECODER.load(path) if outcome(game) >= 0]


def first_decisions(reader, files, pool=None):
    """(critic item, won) of the first decision of every game side with a winner."""
    pool = pool or POOL
    items, won = [], []
    if pool is not None:  # plan critic-v2: parallel decoding
        for pairs in pool.imap(_first_file, files):
            items.extend(item for item, _ in pairs)
            won.extend(result for _, result in pairs)
        return items, np.array(won)
    for path in files:
        for game in reader.load(path):
            if outcome(game) >= 0:
                items.append(detach(game['critic_items'][0]))
                won.append(outcome(game))
    return items, np.array(won)


def probe_metrics(model, probe, firsts):
    model.eval()
    model.encoder.packed_attention = False
    win_head = getattr(model.cfg, 'win_head', False)
    with torch.no_grad():
        pred = predict_values(model, probe['items'], FP32, 96)
        first = predict_values(model, firsts[0], FP32, 96, with_win=win_head)
    model.train()
    first, first_win = first if win_head else (first, None)
    metrics = {'heldout_mc_mse': float(np.mean((pred - probe['mc']) ** 2)), 'heldout_mc_ev': ev(pred, probe['mc']),
               'heldout_gae_mse': float(np.mean((pred - probe['gae']) ** 2)), 'first_auc': auc(firsts[1], first)}
    if win_head:
        metrics['first_win_auc'] = auc(firsts[1], first_win)  # plan critic-v2
    return pred, metrics


def make_probe(reader, files, limit, critic, pool=None):
    sample = sample_decisions(reader, files, limit, critic, pool, gae=False)
    return {'items': sample['critic_items'], 'mc': sample['mc'], 'gae': sample['gae']}


def cmd_functional(args, actor, critic, reader):
    train_files, held = split_files(args.data)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    games = read_decisions(reader, train_files, args.steps * args.minibatch)
    items = [item for game in games for item in game['critic_items']]
    critic.encoder.packed_attention = False
    gae = advantages(games, predict_values(critic, items, FP32, 96))[1].astype(np.float32)
    lengths = np.array([len(item['tok']) for item in items])
    probe = make_probe(reader, held, args.eval_decisions, critic)
    firsts = first_decisions(reader, held)
    start = {name: p.detach().clone() for name, p in critic.named_parameters()}
    base_pred, base = probe_metrics(critic, probe, firsts)
    runs = {'bf16-packed@3e-4': ('bf16-packed', 3e-4), 'fp32@3e-4': ('fp32', 3e-4), 'fp32@1e-4': ('fp32', 1e-4)}
    checkpoints = sorted({s for s in (200, 500, 1000, 2000, args.steps) if s <= args.steps})
    out = {'start': base, 'runs': {}, 'decisions': len(items)}
    deltas, pred_deltas = {}, {}
    for label, (mode, lr) in runs.items():
        model = copy.deepcopy(critic).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        optimizer.load_state_dict(checkpoint['optimizer_state'][1])  # the learner's Adam moments at s2
        for group in optimizer.param_groups:
            group['lr'] = lr
        rng = np.random.default_rng(940005)  # the same minibatch sequence for every run
        batches = [b for _ in range(args.steps * args.minibatch // len(items) + 2)
                   for b in learner_batches(lengths, args.minibatch, rng)][:args.steps]
        curve, update_norms = [], []
        for number, indices in enumerate(batches, 1):
            before = [p.detach().clone() for p in model.parameters()] if number % 10 == 1 else None
            optimizer.zero_grad(set_to_none=True)
            for selected in microbatches(indices, items, TOKENS):
                batch = stack([items[i] for i in selected], 'cuda')
                loss = F.mse_loss(critic_value(model, batch, mode), torch.as_tensor(gae[selected], device='cuda'))
                (loss * len(selected) / len(indices)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if before is not None:
                update_norms.append(float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before)))))
            if number in checkpoints:
                pred, metrics = probe_metrics(model, probe, firsts)
                delta = torch.cat([(p.detach() - start[n]).flatten() for n, p in model.named_parameters()])
                deltas.setdefault(label, {})[number] = delta.cpu()
                pred_deltas.setdefault(label, {})[number] = pred - base_pred
                curve.append({'step': number, **metrics, 'param_delta_norm': float(delta.norm()),
                              'update_norm_mean': float(np.mean(update_norms))})
                print(label, json.dumps(curve[-1]), flush=True)
        out['runs'][label] = curve
        del model, optimizer
    comparisons = {}
    labels = list(runs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            comparisons[f'{a} vs {b}'] = {
                step: {'param_delta_cos': cosine(deltas[a][step], deltas[b][step]),
                       'value_change_corr': float(np.corrcoef(pred_deltas[a][step], pred_deltas[b][step])[0, 1])}
                for step in checkpoints}
    out['comparisons'] = comparisons
    last = checkpoints[-1]
    bf16, fp32, slow = (out['runs'][label][-1] for label in labels)
    out['readout'] = {'bf16_update_norm_vs_fp32': bf16['update_norm_mean'] / fp32['update_norm_mean'],
                      'H3_supported': bool(bf16['update_norm_mean'] <= 0.8 * fp32['update_norm_mean']
                                           and slow['heldout_mc_mse'] <= bf16['heldout_mc_mse']),
                      'at_step': last}
    return out


# ---------------------------------------------------------------- D-4 / critic-noise E1

SOURCES = {'gae': 'critic-gae (learner)', 'mc': 'critic-mc', 'none': 'none (mean only)', 'oracle': 'oracle'}


def actor_group_of(name):
    """Plan critic-noise: the actor's parameter groups (encoder blocks, embedding tables, decoder head, the rest)."""
    if '.layers.' in name:
        return 'block' + name.split('.layers.')[1].split('.')[0]
    root = name.split('.')[0]
    if root in ('cards', 'context', 'qtmpl', 'otmpl'):
        return 'tables'
    if root in ('decoder', 'score', 'names_step'):
        return 'head'
    return 'embed/other'


def pg_minibatches(n, games, size, count, rng, game_disjoint=False):
    """COUNT disjoint minibatches of SIZE sampled decisions. game_disjoint: whole games go to one minibatch (no game
    is shared between minibatches; the overflow of a minibatch's last game is left out)."""
    if not game_disjoint:
        if count * size > n:
            raise ValueError(f'{count} x {size} decisions needed, {n} loaded (raise --n)')
        order = rng.permutation(n)
        return [order[m * size:(m + 1) * size] for m in range(count)]
    by_game = defaultdict(list)
    for index, game in enumerate(games):
        by_game[game].append(index)
    batches, current = [], []
    for game in rng.permutation(list(by_game)):
        current.extend(by_game[game])
        if len(current) >= size:
            batches.append(np.array(rng.permutation(current)[:size]))
            current = []
            if len(batches) == count:
                return batches
    raise ValueError(f'{count} game-disjoint minibatches of {size} need more decisions than the {n} loaded')


def pg_gradients(actor, data, advantages_by_source, indices, params, offsets, total, config):
    """One minibatch's actor PPO-loss gradient (ratio 1: old logp = the actor's own, detached; no entropy) for every
    advantage source, as flat fp32 vectors in parameter-group order. One forward per microbatch serves all sources."""
    buffers = {name: torch.zeros(total, device='cuda') for name in advantages_by_source}
    names = list(advantages_by_source)
    for selected in microbatches(indices, data['items'], TOKENS):
        batch = stack([data['items'][i] for i in selected], 'cuda')
        picks, lengths, excluded = sequence_tensors(selected, data['seqs'], data['excluded'], batch['omask'])
        _, logits = actor_forward(actor, batch, 'fp32')
        own = seq_logprob(logits.float(), batch['omask'], batch['opt'][:, :, 5] > 0, picks, lengths, excluded,
                          sync_free=True).detach()
        for number, name in enumerate(names):
            loss, _ = policy_loss(logits, batch, picks, lengths, excluded, own,
                                  torch.as_tensor(advantages_by_source[name][selected], device='cuda'), config)
            grads = torch.autograd.grad(loss * len(selected) / len(indices), params, allow_unused=True,
                                        retain_graph=number + 1 < len(names))
            for grad, (lo, hi) in zip(grads, offsets):
                if grad is not None:
                    buffers[name][lo:hi] += grad.flatten().float()
        del batch, logits, own
    return buffers


def gram(vectors, lo, hi, chunk=1 << 20):
    """k x k dot products of the rows of VECTORS over columns [lo, hi), in fp64 (chunked to bound memory)."""
    out = torch.zeros(len(vectors), len(vectors), dtype=torch.float64, device=vectors.device)
    for start in range(lo, hi, chunk):
        part = vectors[:, start:min(start + chunk, hi)].double()
        out += part @ part.T
    return out.cpu().numpy()


def cmd_pg_snr(args, actor, critic, reader):
    """D-4, extended by plan critic-noise (E1): per minibatch size (--sizes) and advantage source (--sources), the
    mean pairwise cosine c of the actor's gradient over --minibatches disjoint minibatches, its bootstrap interval
    over minibatches, the noise scale B_noise = B (1 - c) / c, per parameter group, and the E1 verdict."""
    from pg_noise import e1_verdict, noise_scale
    _, held = split_files(args.data)
    files = sorted(trajectory_files(args.data)) if args.all_files else held
    started = time.monotonic()
    data = sample_decisions(reader, files, args.n, critic, critic_items=False)
    old_logp, mc, gae_s2, v_s2 = data['old_logp'], data['mc'], data['gae'], data['value']
    n = len(data['items'])
    load_seconds = time.monotonic() - started
    print(json.dumps({'decisions': n, 'games': len(data['games']), 'stride': data['stride'],
                      'load_seconds': round(load_seconds, 1)}), flush=True)
    # Raw (unnormalised) advantages; every source is normalised the same way below. GAE needs whole games, so the
    # learner's advantage is the sampled decision's GAE return minus its value (both from the full game).
    raw = {'gae': gae_s2 - v_s2, 'mc': mc - v_s2, 'none': mc.copy()}
    out = {}
    if 'oracle' in args.sources:
        with open(args.work / 'oracle.pkl', 'rb') as stream:
            oracle = pickle.load(stream)
        rows = rollout_rows(args.rollouts)
        covered = np.mean([g['g'] in rows for g in data['games']])
        if covered < 0.9:
            raise ValueError(f'oracle: only {covered:.0%} of the games have a rollout row (--rollouts)')
        game_logit = np.array([logit(oracle_seat(oracle, g, rows[g['g']])) if g['g'] in rows else 0.0
                               for g in data['games']])
        seat = game_logit[data['game']]
        fit = np.polyfit(seat, mc, 1)  # the oracle baseline: the best linear map of its logit onto the return
        raw['oracle'] = mc - np.polyval(fit, seat)
        out['oracle_fit'] = [float(x) for x in fit]
    raw = {SOURCES[k]: raw[k] for k in args.sources}
    if args.other_critic:
        _, other, _, _ = load(args.other_critic, 'cuda')
        other.encoder.clear_tables()
        other_sample = sample_decisions(reader, files, args.n, other, critic_items=False)
        if not np.array_equal(other_sample['mc'], mc):
            raise ValueError('the second sample differs')
        raw['other-critic-gae'] = other_sample['gae'] - other_sample['value']
        del other, other_sample
    raw_var = {k: float(np.var(v)) for k, v in raw.items()}
    baselines = {k: ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32) for k, v in raw.items()}
    reference = SOURCES['gae'] if SOURCES['gae'] in baselines else next(iter(baselines))
    config = PPOConfig(bf16=False, entropy=0.0)
    # Parameters in group order, so every group is one contiguous slice of the flat gradient.
    named = sorted(((n_, p) for n_, p in actor.named_parameters() if p.requires_grad),
                   key=lambda item: actor_group_of(item[0]))
    params, offsets, slices, position = [p for _, p in named], [], {}, 0
    for name, p in named:
        offsets.append((position, position + p.numel()))
        group = actor_group_of(name)
        slices[group] = (slices.get(group, (position,))[0], position + p.numel())
        position += p.numel()
    rng = np.random.default_rng(940006)
    k = args.minibatches
    if not args.game_disjoint and k * max(args.sizes) > n:
        raise ValueError(f'{k} x {max(args.sizes)} decisions needed, {n} loaded (raise --n)')
    cells, sizes_out = {}, {}
    for size in args.sizes:
        size_started = time.monotonic()
        batches = pg_minibatches(n, data['game'], size, k, rng, args.game_disjoint)
        store = {name: torch.empty(k, position, device='cuda') for name in baselines}
        for m, indices in enumerate(batches):
            grads = pg_gradients(actor, data, baselines, np.sort(indices), params, offsets, position, config)
            for name, g in grads.items():
                store[name][m] = g
            del grads
            if (m + 1) % 6 == 0:
                print(f'size {size} minibatch {m + 1}/{k} {time.monotonic() - size_started:.0f}s', flush=True)
        means = {name: vectors.sum(0) for name, vectors in store.items()}
        for name, vectors in store.items():
            cell = noise_scale(gram(vectors, 0, position), size, args.resamples)
            cell['groups'] = {group: noise_scale(gram(vectors, lo, hi), size, args.resamples)
                              for group, (lo, hi) in slices.items()}
            cell['cos_mean_gradient_vs_learner'] = cosine(means[name], means[reference])
            cells.setdefault(name, {})[size] = cell
        del store, means
        torch.cuda.empty_cache()
        sizes_out[size] = round(time.monotonic() - size_started, 1)
        print(size, json.dumps({name: {key: round(cells[name][size][key], 4) for key in ('c', 'c_lo', 'c_hi')}
                                | {'b_noise': round(cells[name][size]['b_noise'], 0)} for name in baselines}), flush=True)
    gae_cells = cells.get(SOURCES['gae'], {})
    out.update({'decisions': n, 'games': len(data['games']), 'stride': data['stride'], 'minibatches': k,
                'sizes': list(args.sizes), 'game_disjoint': args.game_disjoint, 'sources': list(baselines),
                'group_parameters': {group: hi - lo for group, (lo, hi) in slices.items()},
                'raw_advantage_variance': raw_var,
                'advantage_corr_vs_learner': {name: float(np.corrcoef(baselines[name], baselines[reference])[0, 1])
                                              for name in baselines},
                'cells': {name: {str(size): cell for size, cell in by_size.items()} for name, by_size in cells.items()},
                'timing': {'load_seconds': round(load_seconds, 1), 'gradient_seconds_per_size': sizes_out,
                           'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}})
    # D-4's fields at one size (--size if measured, else the largest), as critic_diag_report.py reads them.
    at = args.size if args.size in args.sizes else max(args.sizes)
    snr = {name: cells[name][at]['c'] for name in baselines}
    out.update({'minibatch': at, 'mean_pairwise_cos': snr,
                'cos_mean_gradient_vs_learner': {name: cells[name][at]['cos_mean_gradient_vs_learner'] for name in baselines}})
    readout = {}
    none = SOURCES['none']
    if reference == SOURCES['gae'] and none in snr:
        ratio = snr[reference] / snr[none] if snr[none] > 0 else float('inf')  # none resolves no signal
        readout['critic_vs_none_snr_ratio'] = ratio
        if SOURCES['oracle'] in snr:
            readout['critic_vs_oracle_snr_ratio'] = snr[reference] / max(snr[SOURCES['oracle']], 1e-12)
            readout['verdict'] = ('critic contributes' if ratio >= 1.5 and snr[reference] >= snr[SOURCES['oracle']]
                                  else 'no contribution' if ratio <= 1.1 else 'in between')
    out['readout'] = readout
    out['e1'] = e1_verdict(gae_cells, cells.get(none)) if gae_cells else {'E1': 'unmeasured', 'reason': 'no gae source'}
    print(json.dumps(out['e1']), flush=True)
    return out


# ---------------------------------------------------------------- D-5

def _train_file(path):
    """Worker: one training file's critic items, Monte Carlo returns and results (NaN = no winner)."""
    games = _DECODER.load(path)
    items = [item for game in games for item in game['critic_items']]
    results = np.concatenate([np.full(len(game['items']), outcome(game) if outcome(game) >= 0 else np.nan, np.float32)
                              for game in games])
    return items, mc_returns(games), results


def _train_file_serial(reader, path):
    global _DECODER
    _DECODER = reader
    return _train_file(path)


def cmd_train_mc(args, actor, critic, reader):
    train_files, held = split_files(args.data)
    # Plan critic-v2: decode in --workers forked processes (setup and the training stream); the probe skips GAE.
    pool = POOL
    setup_started = time.monotonic()
    probe = make_probe(reader, held, args.eval_decisions, critic, pool)
    firsts = first_decisions(reader, held, pool)
    print(json.dumps({'setup_seconds': time.monotonic() - setup_started, 'probe': len(probe['items']),
                      'firsts': len(firsts[0])}), flush=True)
    model = copy.deepcopy(critic).train()
    optimizer = torch.optim.AdamW(critic_groups(model, args.weight_decay), lr=args.lr, weight_decay=0.0)
    mode = args.critic_mode
    # Plan critic-v2: the win head's cross-entropy on the game result (outcome: 1 won, 0 lost, -1 = NaN, masked).
    use_win = args.win_coef > 0
    if use_win and not model.cfg.win_head:
        raise ValueError('--win-coef needs a critic with win_head')

    def evaluate(steps, seen, epoch):
        _, metrics = probe_metrics(model, probe, firsts)
        return {'step': steps, 'seconds': time.monotonic() - started, 'decisions_seen': seen, 'epoch': epoch, **metrics}
    started, steps, seen, epoch = time.monotonic(), 0, 0, 0
    curve = [evaluate(0, 0, 0)]
    print(json.dumps(curve[-1]), flush=True)
    deadline = started + args.minutes * 60
    while time.monotonic() < deadline:
        epoch += 1
        files = train_files[:]
        random.shuffle(files)
        stream = Prefetch(pool, _train_file, files, 2 * args.workers) if pool is not None else map(
            lambda path: _train_file_serial(reader, path), files)
        for items, returns, results in stream:
            if time.monotonic() >= deadline:
                break
            for indices in learner_batches(np.array([len(i['tok']) for i in items]), args.size, np.random.default_rng(steps)):
                optimizer.zero_grad(set_to_none=True)
                for selected in microbatches(indices, items, args.tokens):
                    batch = stack([items[i] for i in selected], 'cuda')
                    if use_win:
                        value, logits = critic_heads(model, batch, mode)
                        loss = (F.mse_loss(value, torch.as_tensor(returns[selected], device='cuda'))
                                + args.win_coef * win_loss(logits, torch.as_tensor(results[selected], device='cuda')))
                    else:
                        loss = F.mse_loss(critic_value(model, batch, mode),
                                          torch.as_tensor(returns[selected], device='cuda'))
                    (loss * len(selected) / len(indices)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                seen += len(indices)
            if time.monotonic() - started >= len(curve) * args.eval_every * 60:
                curve.append(evaluate(steps, seen, epoch))
                print(json.dumps(curve[-1]), flush=True)
        if pool is not None:
            stream.close()
    curve.append(evaluate(steps, seen, epoch))

    if args.save:
        # Plan critic-v2 warm-start: the source's actor (and its AdamW state) with the trained critic.
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        optimizers = None
        if checkpoint.get('optimizer_state'):
            state = checkpoint['optimizer_state'][0]
            optimizers = [torch.optim.AdamW(actor.parameters(), lr=state['param_groups'][0]['lr'], weight_decay=0.0)]
            optimizers[0].load_state_dict(state)
        metadata = dict(checkpoint.get('rl_meta', {}))
        metadata['critic_warm_start'] = {'source': str(args.checkpoint), 'steps': steps, 'decisions_seen': seen,
                                         'final': curve[-1]}
        model.eval()
        save(args.save, actor, model, load(args.checkpoint)[2], metadata, optimizers)
    return {'mode': mode, 'lr': args.lr, 'minibatch': args.size, 'weight_decay': args.weight_decay,
            'win_coef': args.win_coef, 'train_files': len(train_files), 'curve': curve,
            'saved': str(args.save) if args.save else None}


# ---------------------------------------------------------------- links-round2 phase-3 screen

def cmd_screen(args, actor, critic, reader):
    """The learner's critic update on fixed games: per iteration ~--iteration-steps decisions of the next training
    games, values from the current critic, targets advantages(lam 0.95, value_lam --value-lambda), one epoch at
    minibatch --minibatch (an optimizer step every --accumulate minibatches, the loss scaled 1/accumulate as in v4_rl),
    MSE + --win-coef x win loss, clip 1.0, --lr, Adam moments from the checkpoint. The file order is the same for every
    variant (seed 960001). The actor does not change: this screens the critic's learning only."""
    train_files, held = split_files(args.data)
    order = train_files[:]
    random.Random(960001).shuffle(order)
    probe = make_probe(reader, held, args.eval_decisions, critic, POOL)
    firsts = first_decisions(reader, held, POOL)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = copy.deepcopy(critic).train()
    model.encoder.packed_attention = args.critic_mode == 'bf16-packed'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state'][1])
    for group in optimizer.param_groups:
        group['lr'] = args.lr
    use_win = args.win_coef > 0
    if use_win and not model.cfg.win_head:
        raise ValueError('--win-coef needs a critic with win_head (extend_dna.py --win-head)')
    from swccg_policy.v4_rl import win_labels
    started = time.monotonic()

    def evaluate(steps, seen):
        _, metrics = probe_metrics(model, probe, firsts)
        model.encoder.packed_attention = args.critic_mode == 'bf16-packed'
        return {'updates': steps, 'decisions_seen': seen, 'seconds': round(time.monotonic() - started, 1), **metrics}
    curve = [evaluate(0, 0)]
    print(json.dumps(curve[-1]), flush=True)
    steps = seen = 0
    files = iter(order * 10)
    rng = np.random.default_rng(960002)
    while seen < args.decisions:
        games, count = [], 0
        while count < args.iteration_steps:
            for game in reader.load(next(files)):
                games.append(game)
                count += len(game['items'])
        items = [item for game in games for item in game['critic_items']]
        model.eval()
        model.encoder.packed_attention = False  # fp32 inference (FlashAttention needs bf16)
        with torch.no_grad():
            values = predict_values(model, items, FP32, 96)
        model.train()
        model.encoder.packed_attention = args.critic_mode == 'bf16-packed'
        returns = advantages(games, values, lam=0.95, value_lam=args.value_lambda)[1].astype(np.float32)
        wins = win_labels(games)
        batches = learner_batches(np.array([len(item['tok']) for item in items]), args.minibatch, rng)
        optimizer.zero_grad(set_to_none=True)
        for number, indices in enumerate(batches, 1):
            for selected in microbatches(indices, items, TOKENS):
                batch = stack([items[i] for i in selected], 'cuda')
                if use_win:
                    value, logits = critic_heads(model, batch, args.critic_mode)
                    loss = (F.mse_loss(value, torch.as_tensor(returns[selected], device='cuda'))
                            + args.win_coef * win_loss(logits, torch.as_tensor(wins[selected], device='cuda')))
                else:
                    loss = F.mse_loss(critic_value(model, batch, args.critic_mode),
                                      torch.as_tensor(returns[selected], device='cuda'))
                (loss * len(selected) / len(indices) / args.accumulate).backward()
            if number % args.accumulate == 0 or number == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        seen += len(items)
        if seen >= len(curve) * args.eval_every_decisions:
            curve.append(evaluate(steps, seen))
            print(json.dumps(curve[-1]), flush=True)
    if curve[-1]['decisions_seen'] != seen:
        curve.append(evaluate(steps, seen))
    return {'value_lambda': args.value_lambda, 'accumulate': args.accumulate, 'win_coef': args.win_coef,
            'lr': args.lr, 'minibatch': args.minibatch, 'mode': args.critic_mode, 'train_files': len(train_files),
            'held_files': len(held), 'probe': len(probe['items']), 'firsts': len(firsts[0]), 'curve': curve}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['split', 'values', 'matchup', 'grads', 'functional', 'pg-snr', 'train-mc', 'screen'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/dojo/focus-s2/final.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'runs/dojo/collect-s2b/trajectories')
    parser.add_argument('--rollouts', type=Path, default=ROOT / 'runs/dojo/collect-s2b/rollouts')
    parser.add_argument('--work', type=Path, default=ROOT / 'runs/dojo/critic-diag')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/data/critic-diagnostics')
    parser.add_argument('--name', help='output stem (values: the critic name)')
    parser.add_argument('--critics', nargs='+', default=['s2', 'cfp32-base', 'cfp32-test'])
    parser.add_argument('--n', type=int, default=40000, help='decisions loaded (grads, pg-snr)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 256, 1024])
    parser.add_argument('--minibatches', type=int, default=24)
    parser.add_argument('--modes', nargs='+', choices=MODES, help='grads: the modes compared (default all)')
    parser.add_argument('--size', type=int, default=1024, help='minibatch (pg-snr: 1024; train-mc: 256)')
    parser.add_argument('--minibatch', type=int, default=64, help='functional: the learner minibatch')
    parser.add_argument('--steps', type=int, default=2000, help='functional: updates per run')
    parser.add_argument('--eval-decisions', type=int, default=20000)
    parser.add_argument('--other-critic', type=Path, help='pg-snr: a second critic checkpoint as a baseline')
    # Plan critic-noise E1 (pg-snr).
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES), default=['gae', 'mc', 'none'],
                        help='pg-snr: advantage sources (oracle needs <work>/oracle.pkl and --rollouts rows)')
    parser.add_argument('--all-files', action='store_true', help='pg-snr: every file of --data (held-out data)')
    parser.add_argument('--game-disjoint', action='store_true', help='pg-snr: no game shared between minibatches')
    parser.add_argument('--resamples', type=int, default=2000, help='pg-snr: bootstrap resamples')
    parser.add_argument('--critic-mode', choices=['fp32', 'bf16-packed'], default='fp32')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--minutes', type=float, default=90)
    parser.add_argument('--eval-every', type=float, default=10)
    # Plan critic-v2 (train-mc as the offline warm-start of a fresh critic from scripts/reinit_critic.py).
    parser.add_argument('--win-coef', type=float, default=0.0, help='train-mc: weight of the win head loss')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='train-mc: critic weight decay (critic_groups)')
    parser.add_argument('--save', type=Path, help='train-mc: write the actor + trained critic checkpoint here')
    parser.add_argument('--workers', type=int, default=8, help='decode processes for every sampler (1 = serial)')
    parser.add_argument('--value-lambda', type=float, default=None, help='screen: lambda of the value targets')
    parser.add_argument('--accumulate', type=int, default=1, help='screen: minibatches per critic optimizer step')
    parser.add_argument('--decisions', type=int, default=1_200_000, help='screen: training decisions per variant')
    parser.add_argument('--iteration-steps', type=int, default=10_000, help='screen: decisions per learner iteration')
    parser.add_argument('--eval-every-decisions', type=int, default=200_000)
    parser.add_argument('--tokens', type=int, default=TOKENS,
                        help='train-mc: microbatch token budget (larger = fewer kernel launches; same loss)')
    args = parser.parse_args()
    setup()
    started = time.time()
    if args.command in ('split', 'matchup'):
        result = {'split': cmd_split, 'matchup': cmd_matchup}[args.command](args)
        iteration = None
    else:
        actor, critic, tables, checkpoint = load(args.checkpoint, 'cuda')
        actor.clear_tables()
        critic.encoder.clear_tables()
        reader = TrajectoryReader(tables, [])
        global POOL
        POOL = decode_pool(reader, args.workers) if args.workers > 1 else None
        run = {'values': cmd_values, 'grads': cmd_grads, 'functional': cmd_functional, 'pg-snr': cmd_pg_snr,
               'train-mc': cmd_train_mc, 'screen': cmd_screen}[args.command]
        result = run(args, actor, critic, reader)
        if POOL is not None:
            POOL.terminate()
        iteration = checkpoint['rl_meta']['iteration']
    result.update({'command': args.command, 'checkpoint': str(args.checkpoint), 'iteration': iteration,
                   'seconds': time.time() - started, 'args': {k: str(v) for k, v in vars(args).items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (args.command + (f'_{args.name}' if args.name else '') + '.json')
    path.write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(path)


if __name__ == '__main__':
    main()
