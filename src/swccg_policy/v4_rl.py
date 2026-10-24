"""Schema-4 PPO with independent packing, owned tables, and cached TorchScript publication.

Transient spool retention is allowed only outside the repository's protected runs tree.
The stage supervisor preserves logs, manifests and final checkpoints under runs/.
"""

import argparse
import collections
import multiprocessing
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .critic_replay import CriticReplay
from .learner_speed import Prefetch, compile_tables, single_tables
from .credit import turn_anchors, turn_credit
from .ppo_step import PPOConfig, critic_autocast, critic_precision, microbatches, sequence_tensors, step
from .qnet import chosen_option, from_actor, load_qnet, q_advantages, q_loss
from .trajectory import trajectory_files
from .table_bundle import TableBundle
from .v4_checkpoint import load, save
from .v4_data import stack
from .v4_native import export_ts, update_traced
from .v4_trajectory import TrajectoryReader


ROOT = Path(__file__).resolve().parents[2]


def atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def iteration_order(path):
    """Plan nvf-b-local: learner-NNNN / snap-NNNN by iteration number. A name sort put learner-10003 before
    learner-9999, so from iteration 10,000 on the newest checkpoint and snapshot were the ones pruned (nvf-a-r4 and
    nvf-b-wsd-r crashed at 10,003 loading a pruned learner-10003.ts)."""
    digits = ''.join(ch for ch in path.name.split('.')[0].rsplit('-', 1)[-1] if ch.isdigit())
    return (int(digits) if digits else -1, path.name)


def prune(paths, keep, run):
    if keep < 1:
        raise ValueError('Retention must keep at least one artifact')
    if run.resolve().is_relative_to((ROOT / 'runs').resolve()):
        return
    for path in sorted(paths, key=iteration_order)[:-keep]:
        path.unlink()
        path.with_name(path.name + '.meta.json').unlink(missing_ok=True)
        if path.suffix == '.pt':
            path.with_suffix('.ts').unlink(missing_ok=True)


def scheduled_lr(base, schedule, elapsed, total):
    """D15: optional wall-clock cosine decay to 0 over the stage; 'constant' keeps the base rate."""
    if schedule == 'constant' or not total:
        return base
    fraction = min(max(elapsed / total, 0.0), 1.0)
    return base * 0.5 * (1.0 + np.cos(np.pi * fraction))


def wsd_factor(games, total, warmup, decay_fraction, floor):
    """Plan nvf-b-local A1: warm-up, stable, decay over the arm's games. Linear 0 -> 1 over the first `warmup` games, 1
    until (1 - decay_fraction) of `total`, then a cosine from 1 down to `floor` over the rest (a trunk checkpoint is
    taken where the decay starts)."""
    if games < warmup:
        return games / warmup
    start = total * (1.0 - decay_fraction)
    if games < start:
        return 1.0
    fraction = min((games - start) / max(total - start, 1.0), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * fraction))


def milestones_due(args, segment_games, saved):
    """Plan nvf-b-local A2: (name, games) checkpoints crossed in this segment (arm games, so restarts continue)."""
    marks = [(f'games-{games}', games) for games in args.save_at_games]
    if args.trunk_games:
        marks.append(('trunk', args.trunk_games))
    games = args.lr_schedule_offset_games + segment_games
    return [(name, mark) for name, mark in marks
            if name not in saved and args.lr_schedule_offset_games < mark <= games]


def arm_lr(base, args, segment_seconds, segment_games=0):
    """Plan night-lr-decay: the scheduled rate at this point of the arm. The schedule spans --lr-schedule-minutes
    (default: this segment's --minutes); a restarted segment starts at --lr-schedule-offset-minutes into it.
    Plan nvf-b-local: with --lr-schedule-games the schedule spans learner games instead of wall clock (the same
    decay per game on a machine of another speed); a restart continues from --lr-schedule-offset-games."""
    if getattr(args, 'lr_schedule_games', 0):
        games = args.lr_schedule_offset_games + segment_games
        if args.lr_schedule == 'wsd':
            return base * wsd_factor(games, args.lr_schedule_games, args.lr_warmup_games, args.lr_decay_fraction,
                                     args.lr_floor)
        return scheduled_lr(base, args.lr_schedule, games, args.lr_schedule_games)
    elapsed = args.lr_schedule_offset_minutes * 60 + segment_seconds
    return scheduled_lr(base, args.lr_schedule, elapsed, (args.lr_schedule_minutes or args.minutes) * 60)


# Plan probe-economics 6c': the damage potential from an item's globals (head[4:22]): the opponent's Lost Pile minus
# the seat's own, in cards (head[16] own, head[17] opponent).
HEAD_LOST, HEAD_LOST_OPP = 16, 17


def shaping_terms(phi, form):
    """F_i = Phi_{i+1} - Phi_i and, in the potential form, F at the seat's last decision = -Phi there (Phi = 0 at the
    game end), as scripts/shaping_readout.shaping_term."""
    phi = np.asarray(phi, dtype=np.float64)
    return np.append(np.diff(phi), -phi[-1] if form == 'potential' else 0.0)


def shaping_advantage(phi, gamma, lam, form):
    """Per step t: sum_l (gamma lam)^l F_{t+l} (shaping_terms)."""
    f = shaping_terms(phi, form)
    out = np.zeros(len(f))
    last = 0.0
    for index in range(len(phi) - 1, -1, -1):
        last = f[index] + gamma * lam * last
        out[index] = last
    return out


def advantages(games, values, gamma=1.0, lam=0.95, value_lam=None, credit='decision', round_lambda=None, rounds=None,
               shaping=None, shaping_weight=0.0, shaping_form='potential', shaping_target='both'):
    """Normalised GAE(lam) advantages and the value targets. Plan critic-v2: value_lam (None = lam) gives the value
    targets their own lambda-return; the advantages keep lam. Plan turn-credit: credit='turn' replaces the actor's
    advantages by turn-anchored ones (swccg_policy.credit: round_lambda or rounds); the value targets stay the
    per-decision lambda-returns, so the critic trains as with credit='decision'. Plan probe-economics 6c': shaping
    'damage' adds shaping_weight * the lambda-weighted damage-potential changes to the actor's advantages only (the
    value targets are computed without it). Plan shaping-consistent: shaping_target 'both' (the default) adds
    shaping_weight * F to the per-step rewards instead, so the value targets and the advantages see the same shaped
    rewards; 'actor' keeps the 6c' behaviour."""
    if credit not in ('decision', 'turn'):
        raise ValueError(f'unknown credit {credit!r}')
    if shaping not in (None, 'damage') or shaping_form not in ('potential', 'plain'):
        raise ValueError(f'unknown shaping {shaping!r} / form {shaping_form!r}')
    if shaping_target not in ('actor', 'both'):
        raise ValueError(f'unknown shaping target {shaping_target!r}')
    if shaping and credit == 'turn':
        raise ValueError('shaping and turn credit are not combined')
    if credit == 'turn' and gamma != 1.0:
        raise ValueError('turn credit needs gamma 1')
    advantage = np.zeros_like(values)
    returns = np.zeros_like(values)
    position = 0
    for game in games:
        count = len(game['items'])
        current = values[position:position + count]
        last = 0.0
        last_value = 0.0
        rewards = np.zeros(count)
        rewards[-1] = game['reward']
        # Plan objective-dojo: signed [step, value, item, milestone] entries. The older step/milestone keys stay
        # readable for trajectories written before the Dojo.
        for entry in (*game.get('dojo_rewards', ()), *game.get('step_rewards', ()), *game.get('milestone_rewards', ())):
            rewards[int(entry[0])] += entry[1]
        if shaping:
            phi = [float(item['head'][HEAD_LOST_OPP]) - float(item['head'][HEAD_LOST]) for item in game['items']]
            if shaping_target == 'both':
                rewards += shaping_weight * shaping_terms(phi, shaping_form)
        for index in range(count - 1, -1, -1):
            following = current[index + 1] if index + 1 < count else 0.0
            reward = rewards[index]
            delta = reward + gamma * following - current[index]
            last = delta + gamma * lam * last
            advantage[position + index] = last
            if value_lam is not None:
                last_value = delta + gamma * value_lam * last_value
                returns[position + index] = last_value + current[index]
        if value_lam is None:
            returns[position:position + count] = advantage[position:position + count] + current
        if shaping and shaping_target == 'actor':
            advantage[position:position + count] += shaping_weight * shaping_advantage(phi, gamma, lam, shaping_form)
        if credit == 'turn':
            anchors = turn_anchors([item['head'] for item in game['items']])
            advantage[position:position + count] = turn_credit(current, rewards, anchors, round_lambda=round_lambda,
                                                               rounds=rounds)
        position += count
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8), returns


def win_labels(games):
    """Plan critic-v2: per step 1 (this seat won), 0 (lost) or NaN (capped or undecided: terminal reward not +-1)."""
    labels = []
    for game in games:
        result = {1.0: 1.0, -1.0: 0.0}.get(float(game['reward']), np.nan)
        labels.append(np.full(len(game['items']), result, np.float32))
    return np.concatenate(labels) if labels else np.zeros(0, np.float32)


def rank_auc(scores, labels):
    """Mann-Whitney AUC (ties averaged); None without both classes."""
    labels = np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def early_aucs(games, values):
    """Plan actor-v7-recipe: the value's rank AUC against the result (decided seats only) at each seat's first decision
    and over the first fifth of its decisions; None without both outcomes."""
    labels = win_labels(games)
    position = np.concatenate([np.arange(len(game['items'])) for game in games]) if games else np.zeros(0)
    length = np.concatenate([np.full(len(game['items']), len(game['items'])) for game in games]) if games else np.zeros(0)
    decided = ~np.isnan(labels)
    first = decided & (position == 0)
    early = decided & (position < np.maximum(1, length / 5))
    return {'value_first_auc': rank_auc(values[first], labels[first] > 0.5),
            'value_early_auc': rank_auc(values[early], labels[early] > 0.5)}


def critic_groups(critic, weight_decay):
    """Plan critic-v2: decoupled weight decay on the critic's matrices and embeddings, not on biases or norm gains."""
    if not weight_decay:
        return critic.parameters()
    parameters = list(critic.parameters())
    return [{'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.0}]


def parse_lr_scales(values):
    """Plan night-integer-fp32: --param-lr-scale NAME_PREFIX=FACTOR (repeatable) -> [(prefix, factor)]."""
    scales = []
    for value in values or []:
        prefix, separator, factor = value.partition('=')
        if not separator or not prefix or float(factor) <= 0:
            raise ValueError(f'--param-lr-scale needs NAME_PREFIX=FACTOR with FACTOR > 0, got {value!r}')
        scales.append((prefix, float(factor)))
    return scales


def actor_groups(actor, lr, scales):
    """Plan night-integer-fp32: the actor's parameters in AdamW groups by learning-rate factor. A parameter takes the
    factor of the first prefix its name starts with (1 otherwise). Each group keeps its names ('param_names') and its
    factor ('lr_scale'), so a restart restores the optimizer state by name and a schedule scales from the base rate."""
    named = list(actor.named_parameters())
    unused = [prefix for prefix, _ in scales if not any(name.startswith(prefix) for name, _ in named)]
    if unused:
        raise ValueError(f'--param-lr-scale prefixes match no actor parameter: {unused}')
    groups = {}
    for name, parameter in named:
        factor = next((factor for prefix, factor in scales if name.startswith(prefix)), 1.0)
        group = groups.setdefault(factor, {'params': [], 'param_names': [], 'lr': lr * factor, 'lr_scale': factor})
        group['params'].append(parameter)
        group['param_names'].append(name)
    return list(groups.values())


def load_by_name(optimizer, state, names):
    """Plan night-integer-fp32: restore AdamW state by parameter name into a differently grouped optimizer (the
    checkpoint's groups may be one plain group, then in `names` order). The current groups keep their hyperparameters;
    a parameter missing from the saved state starts fresh."""
    index = {}
    for group in state['param_groups']:
        index.update(zip(group.get('param_names', []), group['params']))
    if not index:
        positions = [position for group in state['param_groups'] for position in group['params']]
        if len(positions) != len(names):
            raise ValueError(f'Optimizer state has {len(positions)} parameters, the actor {len(names)}: '
                             'cannot map by order')
        index = dict(zip(names, positions))
    current = optimizer.state_dict()
    mapped = {}
    for group in current['param_groups']:
        for name, position in zip(group['param_names'], group['params']):
            if name in index and index[name] in state['state']:
                mapped[position] = state['state'][index[name]]
    optimizer.load_state_dict({'state': mapped, 'param_groups': current['param_groups']})
    return len(mapped)


def critic_monitor(critic):
    """Plan critic-v2: total parameter norm and the mean card-part norm over the card table (P-1's growth signal)."""
    with torch.no_grad():
        total = torch.sqrt(sum(p.float().pow(2).sum() for p in critic.parameters()))
        card = critic.encoder.part('card', critic.encoder.cards.table()).float().norm(dim=-1).mean()
    return {'critic_param_norm': float(total), 'critic_card_norm': float(card)}


def critic_warmup(metadata, run, iteration, count):
    """Plan critic-v2: critic-only iterations counted from this run's first iteration. The state is kept in rl_meta,
    so a restart of the same run does not repeat them; a new run (another --run) gets its own. None = no warm-up."""
    warmup = metadata.get('critic_warmup')
    if warmup and warmup.get('run') == str(run):
        return warmup
    return {'run': str(run), 'until': iteration + count} if count else None


_READER = None  # plan critic-v2 speed-up: each load worker's own reader


def _init_reader(reader):
    global _READER
    _READER = reader


def _load_file(path):
    return _READER.load(path)


def load_pool(reader, workers):
    """Plan critic-v2 speed-up: processes that decode trajectory files (numpy only, no CUDA). Spawned, not forked:
    forked children of the multi-GB learner gradually copy its memory (8 forks added ~8 GB and tripped the stage's
    resource guard); a spawned worker holds only its pickled reader."""
    if workers <= 1:
        return None
    return multiprocessing.get_context('spawn').Pool(workers, initializer=_init_reader, initargs=(reader,))


def predict_values(critic, items, config, infer_batch, with_win=False, tokens=None):
    """Values for every item; with_win also returns the win head's logits (plan critic-v2). tokens: the microbatch
    token budget (default config.critic_tokens); inference only, so a larger budget changes nothing but speed."""
    device = next(critic.parameters()).device
    order = np.argsort([len(item['tok']) for item in items], kind='stable')
    values = np.zeros(len(items), np.float32)
    wins = np.zeros(len(items), np.float32) if with_win else None
    critic.encoder.cache_card_table()
    try:
        with torch.no_grad(), critic_precision(config):
            for offset in range(0, len(order), infer_batch):
                for indices in microbatches(order[offset:offset + infer_batch], items, tokens or config.critic_tokens):
                    batch = stack([items[index] for index in indices], device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=critic_autocast(config)):
                        if with_win:
                            prediction, logits = critic.heads(batch)
                            wins[indices] = logits.float().cpu().numpy()
                        else:
                            prediction = critic(batch)
                    values[indices] = prediction.float().cpu().numpy()
    finally:
        critic.encoder.clear_tables()
    return (values, wins) if with_win else values


def ppo_batches(lengths, minibatch):
    """One epoch's minibatches: permute, sort by length within groups of 16 minibatches, shuffle the minibatches
    (numpy's and random's global generators, as the learner always did)."""
    permutation = np.random.permutation(len(lengths))
    batches = []
    for offset in range(0, len(lengths), minibatch * 16):
        part = permutation[offset:offset + minibatch * 16]
        part = part[np.argsort(lengths[part], kind='stable')]
        batches.extend(part[index:index + minibatch] for index in range(0, len(part), minibatch))
    random.shuffle(batches)
    return batches


def ppo_pass(actor, critic, optimizers, items, critic_items, batches, sequences, exclusions, old_logp, advantages,
             returns, wins, config, policy_on, accumulate, statistics, batch_counts, actor_accumulate=1):
    """One PPO pass over the minibatches; the statistics and microbatch counts are appended to the given containers."""
    for number, indices in enumerate(batches):
        # Plan critic-v2: the critic steps once per group of `accumulate` minibatches (the last group may
        # be shorter); each minibatch's critic loss is divided by its group size.
        first = number - number % accumulate
        size = min(accumulate, len(batches) - first)
        # Plan critic-noise Amendment 2: the actor likewise, per group of `actor_accumulate` minibatches.
        a_first = number - number % actor_accumulate
        a_size = min(actor_accumulate, len(batches) - a_first)
        stats, counts = step(actor, critic, optimizers, items, critic_items, indices,
                             sequences, exclusions, old_logp, advantages, returns,
                             stack=stack, config=config, policy_on=policy_on,
                             critic_zero=number == first, critic_step=number == first + size - 1,
                             actor_zero=number == a_first, actor_step=number == a_first + a_size - 1,
                             wins=wins, critic_scale=1.0 / size if accumulate > 1 else 1.0,
                             actor_scale=1.0 / a_size if actor_accumulate > 1 else 1.0)
        batch_counts.update(counts)
        for name, value in stats.items():
            statistics[name].append(value)


def logit_distribution(statistics, vocabulary, threshold=20.0, top=5):
    """Plan actor-v6-inputs (BL-01): the distribution of each decision's largest |logit| in this update: p50 / p99,
    the share above the threshold overall and for the question templates with the most such decisions, and the
    question / option template of the maximum."""
    logit = torch.cat(statistics['rows_logit']).float().cpu().numpy()
    question = torch.cat(statistics['rows_qtmpl']).cpu().numpy()
    option = torch.cat(statistics['rows_otmpl']).cpu().numpy()
    names = {family: {index: key for key, index in vocabulary[family].items()} for family in ('q', 'o')}
    high = logit > threshold
    templates = collections.Counter(question[high].tolist())
    worst = int(np.argmax(logit))
    return {'decisions': int(len(logit)), 'p50': float(np.percentile(logit, 50)), 'p99': float(np.percentile(logit, 99)),
            f'share_above_{threshold:g}': float(high.mean()),
            'top_templates': [{'question': names['q'].get(index, str(index)), 'decisions': int((question == index).sum()),
                               f'share_above_{threshold:g}': float(high[question == index].mean())}
                              for index, _ in templates.most_common(top)],
            'max': {'logit': float(logit[worst]), 'question': names['q'].get(int(question[worst]), str(question[worst])),
                    'option': names['o'].get(int(option[worst]), str(option[worst]))}}


def q_setup(args, loaded, fused):
    """Plan q-critic stage 2: the Q-net and its AdamW. A checkpoint that carries `qnet_state` (latest.pt of a running
    stage, or a restarted segment's start) continues it; otherwise --qnet (qnet.save_qnet) on the loaded tables. Table
    handling and precision follow the actor: tables cleared, single tables / compiled tables and packed attention as
    the actor has them (off for an fp32 actor)."""
    actor, _, _, checkpoint = loaded
    if 'qnet_state' in checkpoint:
        qnet = from_actor(actor)
        qnet.load_state_dict(checkpoint['qnet_state'])
        source = 'checkpoint'
    elif args.qnet is not None:
        qnet, _ = load_qnet(args.qnet, loaded, args.device)
        source = str(args.qnet)
    else:
        raise ValueError('--advantage q needs --qnet or a checkpoint with qnet_state')
    qnet.eval()
    qnet.clear_tables()
    qnet.packed_attention = getattr(actor, 'packed_attention', False)
    if args.single_tables:
        single_tables(qnet)
    if args.compile == 'tables':
        compile_tables(qnet)
    optimizer = torch.optim.AdamW(qnet.parameters(), lr=args.qnet_lr, weight_decay=0.0, fused=fused)
    if source == 'checkpoint' and 'qnet_optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['qnet_optimizer'])
    return qnet, optimizer, source


def option_advantages(actor, qnet, items, sequences, exclusions, config, stack=stack):
    """Plan q-critic: per decision, without gradients, A_Q = Q(s, chosen) - sum_o pi(o) Q(s, o) and Q(s, chosen). pi is
    the current actor's first-pick policy at T = 1 (softmax over the legal, not excluded options, as sampled); chosen
    is the decision's first pick (qnet.chosen_option). Microbatches by the actor's token budget and precision."""
    device = next(qnet.parameters()).device
    order = np.argsort([len(item['tok']) for item in items], kind='stable')
    advantage = np.zeros(len(items), np.float32)
    chosen_values = np.zeros(len(items), np.float32)
    actor.cache_card_table()
    qnet.cache_card_table()
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16):
            for indices in microbatches(order, items, config.actor_tokens):
                batch = stack([items[index] for index in indices], device)
                _, _, excluded = sequence_tensors(indices, sequences, exclusions, batch['omask'])
                valid = batch['omask'] & ~excluded
                pi = torch.softmax(actor(batch).float().masked_fill(~valid, float('-inf')), 1)
                values = qnet(batch).float()
                chosen = torch.as_tensor([chosen_option(sequences[index]) for index in indices], device=device)
                advantage[indices] = q_advantages(values, valid, pi, chosen).cpu().numpy()
                chosen_values[indices] = values.gather(1, chosen.view(-1, 1)).squeeze(1).cpu().numpy()
    finally:
        actor.clear_tables()
        qnet.clear_tables()
    return advantage, chosen_values


def q_train(qnet, optimizer, items, sequences, returns, minibatch, config, max_grad_norm, stack=stack):
    """Plan q-critic: one epoch of the Q-net over the batch's decisions (those with a pick): MSE of Q(s, chosen)
    against the critic's value targets, minibatches as the PPO pass's, microbatches by the actor's token budget, the
    actor's gradient clip, one AdamW step per minibatch. Returns the mean loss and the mean gradient norm."""
    device = next(qnet.parameters()).device
    picked = np.asarray([index for index, sequence in enumerate(sequences) if len(sequence)], np.int64)
    losses, norms = [], []
    for positions in ppo_batches(np.asarray([len(items[index]['tok']) for index in picked]), minibatch):
        indices = picked[positions]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for selected in microbatches(indices, items, config.actor_tokens):
            batch = stack([items[index] for index in selected], device)
            chosen = torch.as_tensor([chosen_option(sequences[index]) for index in selected], device=device)
            targets = torch.as_tensor(returns[selected], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16):
                loss, _ = q_loss(qnet, batch, chosen, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Q loss; update not published')
            weight = len(selected) / len(indices)
            (loss * weight).backward()
            total = total + loss.detach() * weight
        norms.append(torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_grad_norm, error_if_nonfinite=True))
        optimizer.step()
        losses.append(total)
    if not losses:
        return None, None
    return float(torch.stack(losses).mean()), float(torch.stack(norms).mean())


def q_extra(qnet, optimizer):
    """Plan q-critic: the Q-net's entries of every checkpoint the learner writes."""
    return {'qnet_state': {key: value.detach().cpu() for key, value in qnet.state_dict().items()},
            'qnet_optimizer': optimizer.state_dict()}


def correlation(a, b):
    """Pearson correlation; None when either side is constant."""
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def publish(run, actor, critic, tables, meta, optimizers, traced, snap_every, extra=None):
    started = time.monotonic()
    iteration = meta['iteration']
    path = run / 'ckpt' / f'learner-{iteration:04d}.pt'
    # Plan q-critic: the Q-net's entries only when it is active (the default call is the previous one).
    save(path, actor, critic, tables, meta, optimizers, *((extra,) if extra else ()))
    atomic_copy(path, run / 'ckpt/latest.pt')
    checkpoint_seconds = time.monotonic() - started
    export_started = time.monotonic()
    retraced = traced is None
    ts_path = path.with_suffix('.ts')
    if traced is None:
        traced = export_ts(actor, ts_path)
    else:
        update_traced(traced, actor)
        temporary = ts_path.with_name(ts_path.name + '.tmp')
        traced.save(str(temporary))
        os.replace(temporary, ts_path)
    atomic_copy(ts_path, run / 'ckpt/latest.ts')
    marker = run / 'ckpt/latest.ts.json'
    temporary = marker.with_name(marker.name + '.tmp')
    temporary.write_text(json.dumps({'iteration': iteration, 'file': ts_path.name}) + '\n')
    os.replace(temporary, marker)
    if iteration % snap_every == 0:
        # Native opponents consume only the actor export; full resumable states live in ckpt/.
        atomic_copy(ts_path, run / 'snaps' / f'snap-{iteration:04d}.ts')
    return traced, {'checkpoint_save_seconds': checkpoint_seconds,
                    'export_seconds': time.monotonic() - export_started, 'retraced': retraced}


def learn(args):
    run = args.run
    for name in ('ckpt', 'snaps', 'traj', 'traj-consumed'):
        (run / name).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    latest = run / 'ckpt/latest.pt'
    actor, critic, tables, checkpoint = load(latest if latest.exists() else args.initial, args.device)
    reader = TrajectoryReader(tables, [TableBundle.load(path) for path in args.acting_tables])
    pool = load_pool(reader, args.load_workers)
    actor.clear_tables()
    critic.encoder.clear_tables()
    if args.actor_fp32 and not args.bf16:
        raise ValueError('--actor-fp32 keeps the critic in bf16: use --no-bf16 for both in fp32')
    if args.packed_attention:
        if not args.bf16 or actor.cfg.dropout or critic.cfg.dropout:
            raise ValueError('Packed attention requires BF16 and no dropout')
        # Plan night-integer-fp32: an fp32 actor (--actor-fp32) keeps the standard attention, like an fp32 critic.
        actor.packed_attention = not args.actor_fp32
        # Packed attention needs bf16: an fp32 critic (--critic-fp32) keeps the standard attention.
        critic.encoder.packed_attention = not args.critic_fp32
    config = PPOConfig(actor_tokens=args.actor_tokens, critic_tokens=args.critic_tokens,
                       actor_clip=args.max_grad_norm, bf16=args.bf16 and not args.actor_fp32,
                       entropy=args.entropy_coef,
                       critic_bf16=False if args.critic_fp32 else (True if args.actor_fp32 else None),
                       win_coef=args.win_coef,
                       critic_tf32=args.critic_tf32, critic_stream=args.critic_stream)
    if args.single_tables:
        single_tables(actor, critic)
    if args.compile == 'tables':
        compile_tables(actor, critic)
    if not 0 < args.gamma <= 1:
        raise ValueError('--gamma must be in (0, 1]')
    if args.credit == 'turn' and (args.round_lambda is None) == (args.rounds is None):
        raise ValueError('--credit turn needs exactly one of --round-lambda and --rounds')
    if args.critic_accumulate < 1 or args.actor_accumulate < 1 or not (args.value_lambda is None or 0 <= args.value_lambda <= 1) or not 0 <= args.lam <= 1:
        raise ValueError('--critic-accumulate must be >= 1 and --lam / --value-lambda in [0, 1]')
    use_win = args.win_coef > 0
    if use_win and not critic.cfg.win_head:
        raise ValueError('--win-coef needs a critic with win_head (scripts/reinit_critic.py)')
    fused = args.device.startswith('cuda')
    scales = parse_lr_scales(args.param_lr_scale)
    optimizers = [torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0, fused=fused)
                  for parameters, lr in ((actor_groups(actor, args.lr, scales) if scales else actor.parameters(),
                                          args.lr),
                                         (critic_groups(critic, args.value_weight_decay), args.value_lr))]
    for number, (optimizer, state) in enumerate(zip(optimizers, checkpoint.get('optimizer_state', []))):
        if number == 0 and scales:
            restored = load_by_name(optimizer, state, [name for name, _ in actor.named_parameters()])
            print(json.dumps({'param_lr_scale': scales, 'actor_state_restored': restored,
                              'groups': [{'lr': group['lr'], 'parameters': len(group['params'])}
                                         for group in optimizer.param_groups]}), flush=True)
        else:
            optimizer.load_state_dict(state)
    # Plan q-critic stage 2: the actor's advantage from an option-value network (off by default: nothing below runs).
    qnet = q_optimizer = None
    if args.advantage == 'q':
        qnet, q_optimizer, q_source = q_setup(args, (actor, critic, tables, checkpoint), fused)
        print(json.dumps({'advantage': 'q', 'qnet_source': q_source, 'qnet_lr': args.qnet_lr}), flush=True)
    elif args.qnet is not None:
        raise ValueError('--qnet applies to --advantage q')
    metadata = checkpoint.get('rl_meta', {})
    iteration = int(metadata.get('iteration', 0))
    games_total = int(metadata.get('games', 0))
    steps_total = int(metadata.get('steps', 0))
    start_games = games_total
    warmup = critic_warmup(metadata, run, iteration, args.critic_warmup)
    # Plan critic-replay: critic-only replay over the last --critic-replay-updates updates (not kept across restarts).
    replay = (CriticReplay(args.critic_replay_updates, args.critic_replay_stride, args.seed, args.critic_replay_mode)
              if args.critic_replay_updates else None)
    del checkpoint
    traced = None
    started = time.monotonic()
    saved_milestones = set()  # plan nvf-b-local A2: checkpoints at arm-game marks
    last_archive = started  # plan critic-replay rev. 2: full checkpoints copied out of the RAM spool every N minutes
    deadline = started + args.minutes * 60 if args.minutes else float('inf')

    def stopped():
        return time.monotonic() >= deadline or (run / 'STOP').exists()

    def decode(files):
        return ([game for loaded in pool.map(_load_file, files) for game in loaded] if pool is not None
                else [game for path in files for game in reader.load(path)])
    # Plan learner-speed S-4: the next batch decodes while this update runs.
    prefetch = Prefetch(run / 'traj', args.batch_steps, trajectory_files, decode, stopped) if args.prefetch else None
    with (run / 'learner.jsonl').open('a') as log:
        while not stopped():
            collected = time.monotonic()
            files = []
            count = 0
            prefetched = None
            if prefetch is not None:
                prefetched = prefetch.get()
                if prefetched is None:
                    return
                files = prefetched['files']
            while count < args.batch_steps and prefetch is None:
                if stopped():
                    return
                for path in trajectory_files(run / 'traj'):
                    if path in files:
                        continue
                    sidecar = path.with_name(path.name + '.meta.json')
                    if not sidecar.exists():
                        continue
                    count += int(json.loads(sidecar.read_text())['steps'])
                    files.append(path)
                    if count >= args.batch_steps:
                        break
                if count < args.batch_steps:
                    time.sleep(1)
            queue_wait = time.monotonic() - collected
            update_started = time.monotonic()
            phases = {}
            phase_memory = {}
            last_phase = update_started
            if fused:
                torch.cuda.reset_peak_memory_stats()

            def phase(name):
                nonlocal last_phase
                if fused:
                    torch.cuda.synchronize()
                    phase_memory[name] = {'allocated_peak': torch.cuda.max_memory_allocated(),
                                          'reserved_peak': torch.cuda.max_memory_reserved()}
                    torch.cuda.reset_peak_memory_stats()
                now = time.monotonic()
                phases[name] = now - last_phase
                last_phase = now

            games = prefetched['games'] if prefetched is not None else decode(files)
            phase('load')
            for path in files:
                shutil.move(path, run / 'traj-consumed' / path.name)
                sidecar = path.with_name(path.name + '.meta.json')
                shutil.move(sidecar, run / 'traj-consumed' / sidecar.name)
            prune(trajectory_files(run / 'traj-consumed'), args.keep_consumed, run)
            accepted = []
            stale = 0
            lags = []
            for game in games:
                versions = game['policy_versions']
                if versions and (min(versions) < iteration - args.max_policy_lag or max(versions) > iteration):
                    stale += len(game['items'])
                else:
                    accepted.append(game)
                    lags.extend(iteration - version for version in versions)
            if stale:
                with (run / 'stale-discards.jsonl').open('a') as discarded:
                    discarded.write(json.dumps({'iteration': iteration, 'steps': stale}) + '\n')
            games = accepted
            if not games:
                continue
            items = [item for game in games for item in game['items']]
            critic_items = [item for game in games for item in game['critic_items']]
            if use_win:
                values, win_logits = predict_values(critic, critic_items, config, args.infer_batch, with_win=True,
                                                    tokens=args.infer_tokens)
            else:
                values = predict_values(critic, critic_items, config, args.infer_batch, tokens=args.infer_tokens)
            normalized_advantages, returns = advantages(games, values, gamma=args.gamma, lam=args.lam,
                                                             value_lam=args.value_lambda, credit=args.credit,
                                                             round_lambda=args.round_lambda, rounds=args.rounds,
                                                             shaping=args.shaping, shaping_weight=args.shaping_weight,
                                                             shaping_form=args.shaping_form,
                                                             shaping_target=args.shaping_target)
            wins = win_labels(games) if use_win else None
            phase('values_gae')
            old_logp = np.concatenate([game['old_logp'] for game in games])
            sequences = [sequence for game in games for sequence in game['seqs']]
            exclusions = [excluded for game in games for excluded in game['excluded']]
            lengths = np.asarray([max(len(item['tok']), len(privileged['tok']))
                                  for item, privileged in zip(items, critic_items, strict=True)])
            actor_advantages = normalized_advantages
            q_stats = {}
            if qnet is not None:
                # Plan q-critic: A_Q replaces the GAE advantages in the actor's loss (normalised the same way); the GAE
                # ones stay for the correlation, the value targets are unchanged.
                raw_q, chosen_q = option_advantages(actor, qnet, items, sequences, exclusions, config)
                actor_advantages = (raw_q - raw_q.mean()) / (raw_q.std() + 1e-8)
                q_variance = float(np.var(returns))
                q_stats = {'q_adv_std': float(raw_q.std()), 'corr_q_gae': correlation(raw_q, normalized_advantages),
                           'q_mean': float(chosen_q.mean()),
                           'q_explained_variance': 1 - float(np.var(returns - chosen_q)) / q_variance
                           if q_variance > 1e-8 else None}
                phase('q_advantage')
            statistics = collections.defaultdict(list)
            batch_counts = collections.Counter()
            policy_on = iteration >= args.value_warmup and (warmup is None or iteration >= warmup['until'])
            if args.lr_schedule != 'constant':
                # Plan night-lr-decay: the schedule spans the whole arm (--lr-schedule-minutes), and a restarted segment
                # continues it from the arm's earlier training time (--lr-schedule-offset-minutes).
                segment_seconds = time.monotonic() - started
                for optimizer, base in zip(optimizers, (args.lr, args.value_lr)):
                    for group in optimizer.param_groups:
                        group['lr'] = arm_lr(base * group.get('lr_scale', 1.0), args, segment_seconds,
                                             games_total - start_games)
            for _ in range(args.epochs):
                ppo_pass(actor, critic, optimizers, items, critic_items, ppo_batches(lengths, args.minibatch),
                         sequences, exclusions, old_logp, actor_advantages, returns, wins, config, policy_on,
                         args.critic_accumulate, statistics, batch_counts, args.actor_accumulate)
            phase('ppo')
            if qnet is not None:
                q_stats['q_loss'], q_stats['q_grad_norm'] = q_train(qnet, q_optimizer, items, sequences, returns,
                                                                    args.minibatch, config, args.max_grad_norm)
                phase('q_train')
                q_stats['q_seconds'] = phases['q_advantage'] + phases['q_train']
                q_stats['qnet_source'] = q_source
            replay_stats = {}
            if replay is not None:
                replay.add(games)
                replay_stats = replay.train(critic, optimizers[1], config, stack,
                                            replay.steps(len(items), args.critic_replay_ratio, 256), 256,
                                            tokens=args.critic_replay_tokens or None)
                replay_stats['replay_size'] = len(replay)
                # First-decision AUC over every game in the window (~W x 30 games: far less noisy than one batch).
                cached, _, labels, first = replay.flat()
                firsts = [item for item, is_first in zip(cached, first) if is_first]
                decided = ~np.isnan(labels[first])
                if use_win and firsts:
                    _, first_logits = predict_values(critic, firsts, config, args.infer_batch, with_win=True)
                    replay_stats['replay_first_win_auc'] = rank_auc(first_logits[decided], labels[first][decided] > 0.5)
                if firsts:
                    first_values = predict_values(critic, firsts, config, args.infer_batch)
                    replay_stats['replay_first_value_auc'] = rank_auc(first_values[decided], labels[first][decided] > 0.5)
                phase('replay')
            iteration += 1
            games_total += len(games)
            steps_total += len(items)
            metrics = {name: float(torch.stack(values).max() if name in ('max_abs_logit', 'raw_max_abs_logit')
                                   else torch.stack(values).mean()) for name, values in statistics.items()
                       if not name.startswith('rows_')}
            if statistics.get('rows_logit'):
                metrics['logits'] = logit_distribution(statistics, tables.vocabulary.payload)
            variance = float(np.var(returns))
            explained = 1 - float(np.var(returns - values)) / variance if variance > 1e-8 else None
            meta = {'iteration': iteration, 'games': games_total, 'steps': steps_total,
                    'args': {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                    'ppo_config': asdict(config)}
            if warmup is not None:
                meta['critic_warmup'] = warmup
            traced, publication = publish(run, actor, critic, tables, meta, optimizers, traced, args.snap_every,
                                          extra=q_extra(qnet, q_optimizer) if qnet is not None else None)
            if args.archive and args.archive_minutes and time.monotonic() - last_archive >= args.archive_minutes * 60:
                last_archive = time.monotonic()
                args.archive.mkdir(parents=True, exist_ok=True)
                atomic_copy(run / 'ckpt/latest.pt', args.archive / f'learner-{iteration:04d}.pt')
                (args.archive / f'learner-{iteration:04d}.json').write_text(json.dumps(
                    {'iteration': iteration, 'games': games_total, 'steps': steps_total,
                     'elapsed': last_archive - started}) + '\n')
            for name, mark in milestones_due(args, games_total - start_games, saved_milestones):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                atomic_copy(run / 'ckpt/latest.pt', args.save_dir / f'{name}.pt')
                (args.save_dir / f'{name}.json').write_text(json.dumps(
                    {'iteration': iteration, 'games': games_total, 'arm_games': args.lr_schedule_offset_games
                     + games_total - start_games, 'mark': mark}) + '\n')
                saved_milestones.add(name)
            prune((run / 'ckpt').glob('learner-*.pt'), args.keep_ckpt, run)
            prune((run / 'snaps').glob('snap-*.ts'), args.keep_snapshots, run)
            phase('checkpoint')
            opponents = {}
            for opponent in sorted({game['opp'] for game in games}):
                subset = [game for game in games if game['opp'] == opponent]
                opponents[opponent] = {'games': len(subset), 'win_rate': float(np.mean([g['won'] for g in subset])),
                                       'unfinished': sum(g['o'] != 'finished' for g in subset)}
            record = {'iteration': iteration, 'time': time.time(), 'elapsed': time.monotonic() - started,
                      'steps': len(items), 'steps_total': steps_total, 'games': len(games), 'games_total': games_total,
                      'policy_on': policy_on, 'discarded_stale_steps': stale, 'queue_wait_seconds': queue_wait,
                      'update_seconds': time.monotonic() - update_started,
                      'phases': phases, 'phase_memory': phase_memory,
                      'policy_lag_mean': float(np.mean(lags)), 'policy_lag_max': max(lags),
                      'value_explained_variance': explained, 'value_mean': float(values.mean()),
                      'return_mean': float(returns.mean()), 'win_by_opp': opponents,
                      'mean_reward': float(np.mean([game['reward'] for game in games])),
                      'microbatches': dict(batch_counts), 'publication': publication, **metrics, **replay_stats,
                      **q_stats}
            if prefetched is not None:
                record['prefetch'] = {key: prefetched[key] for key in ('collect_seconds', 'decode_seconds')}
            action_counts = collections.defaultdict(lambda: collections.Counter())
            for item, sequence in zip(items, sequences, strict=True):
                if sequence:
                    action_counts[str(int(item['head'][0]))].update(
                        decisions=1, first=int(sequence[0] == 0),
                        pass_choice=int(item['opt'][sequence[0], 5] > 0))
            record['action_counts'] = {key: dict(value) for key, value in action_counts.items()}
            record['actor_lr'] = optimizers[0].param_groups[0]['lr']
            record['critic_lr'] = optimizers[1].param_groups[0]['lr']
            # Plan critic-v2 monitoring, always on since plan actor-v7-recipe: AUC of the value against the result at
            # each seat's first decision and over the first fifth of its decisions (v6-n1's early-game blind spot).
            record.update(early_aucs(games, values))
            if use_win or args.value_weight_decay or args.critic_accumulate > 1 or args.value_lambda is not None:
                starts = np.cumsum([0] + [len(game['items']) for game in games[:-1]])
                results = win_labels(games)[starts] if games else np.zeros(0)
                decided = ~np.isnan(results)
                if use_win:
                    record['win_first_auc'] = rank_auc(win_logits[starts][decided], results[decided] > 0.5)
                record.update(critic_monitor(critic))
            record['observation_tokens'] = {
                name: {'median': float(np.median(lengths)), 'p95': float(np.quantile(lengths, .95)),
                       'max': int(max(lengths))}
                for name, lengths in (('actor', [len(item['tok']) for item in items]),
                                      ('critic', [len(item['tok']) for item in critic_items]))}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
            if args.max_iterations and iteration >= args.max_iterations:
                break
            # Plan dojo-focus: a games budget counted from this run's start (rl_meta 'games' of --initial).
            if args.max_games and games_total - start_games >= args.max_games:
                (run / 'GAMES_COMPLETE').write_text(json.dumps({'iteration': iteration, 'games': games_total - start_games}))
                break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--acting-tables', type=Path, nargs='*', default=[],
                        help='Explicit ancestor bundles for trajectories produced before manual table growth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=883040)
    parser.add_argument('--minutes', type=float, default=0)
    parser.add_argument('--max-iterations', type=int, default=0)
    parser.add_argument('--max-games', type=int, default=0, help='Stop after the first update past this many games')
    parser.add_argument('--batch-steps', type=int, default=12000)
    parser.add_argument('--minibatch', type=int, default=64)
    parser.add_argument('--infer-batch', type=int, default=96)
    parser.add_argument('--actor-tokens', type=int, default=32768)
    parser.add_argument('--critic-tokens', type=int, default=32768)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--value-lr', type=float, default=3e-4)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)  # user 2026-10-03: one PPO pass per batch
    parser.add_argument('--entropy-coef', type=float, default=PPOConfig.entropy)
    parser.add_argument('--lr-schedule-minutes', type=float, default=0,
                        help='Plan night-lr-decay: the schedule\'s total length (0 = --minutes, this segment alone)')
    parser.add_argument('--lr-schedule-offset-minutes', type=float, default=0,
                        help='Plan night-lr-decay: training minutes of the arm\'s earlier segments (a restart continues)')
    parser.add_argument('--lr-schedule-games', type=int, default=0,
                        help='Plan nvf-b-local: the schedule spans this many learner games of the arm (0 = wall clock)')
    parser.add_argument('--lr-schedule-offset-games', type=int, default=0,
                        help='Plan nvf-b-local: learner games of the arm\'s earlier segments (a restart continues)')
    parser.add_argument('--lr-warmup-games', type=int, default=0, help='Plan nvf-b-local A1: wsd linear warm-up')
    parser.add_argument('--lr-decay-fraction', type=float, default=0.1, help='wsd: the last fraction of the games decays')
    parser.add_argument('--lr-floor', type=float, default=0.05, help='wsd: the decay ends at this fraction of the rate')
    parser.add_argument('--save-at-games', type=lambda text: [int(v) for v in text.split(',') if v], default=[],
                        help='Plan nvf-b-local A2: full checkpoints at these arm games (comma-separated)')
    parser.add_argument('--trunk-games', type=int, default=0, help='Plan nvf-b-local A1: trunk.pt at these arm games')
    parser.add_argument('--save-dir', type=Path, help='where the milestone and trunk checkpoints go')
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine', 'wsd'], default='constant',
                        help='cosine: actor and critic LR decay to 0 over --minutes of wall clock')
    parser.add_argument('--value-warmup', type=int, default=2)
    parser.add_argument('--max-policy-lag', type=int, default=3)
    parser.add_argument('--snap-every', type=int, default=5)
    parser.add_argument('--keep-consumed', type=int, default=4)
    parser.add_argument('--keep-ckpt', type=int, default=3)
    parser.add_argument('--keep-snapshots', type=int, default=6)
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--critic-fp32', action='store_true',
                        help='Plan critic-fp32: run the critic (forward, backward, values) in fp32')
    parser.add_argument('--packed-attention', action='store_true')
    parser.add_argument('--actor-fp32', action='store_true',
                        help='Plan night-integer-fp32: run the actor in fp32 with standard attention (critic as set)')
    parser.add_argument('--param-lr-scale', action='append', default=[], metavar='NAME_PREFIX=FACTOR',
                        help='Plan night-integer-fp32: actor parameters starting with NAME_PREFIX learn at FACTOR x --lr')
    # Plan critic-v2 (critic only; the defaults are the previous behaviour).
    # Plan actor-v6-inputs: the GAE lambda of the advantages (and of the value targets unless --value-lambda).
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda of the advantages, per decision')
    parser.add_argument('--value-lambda', type=float, default=None,
                        help='lambda of the value targets (default: the advantage lambda, --lam)')
    # Plan probe-economics 6c': damage shaping of the actor's advantages (the critic's targets are unchanged).
    parser.add_argument('--shaping', choices=['damage'], default=None)
    parser.add_argument('--shaping-weight', type=float, default=0.0, help='reward per card of net damage')
    parser.add_argument('--shaping-form', choices=['potential', 'plain'], default='potential')
    # Plan shaping-consistent: 'both' shapes the per-step rewards (critic targets and advantages); 'actor' is 6c'.
    parser.add_argument('--shaping-target', choices=['actor', 'both'], default='both')
    # Plan links-round2: the discount per policy step (1 = the previous behaviour).
    parser.add_argument('--gamma', type=float, default=1.0, help='discount per decision of the learner seat')
    parser.add_argument('--win-coef', type=float, default=0.0, help='weight of the win head loss (needs win_head)')
    parser.add_argument('--value-weight-decay', type=float, default=0.0,
                        help='decoupled AdamW weight decay on the critic matrices and embeddings')
    parser.add_argument('--critic-accumulate', type=int, default=1, help='minibatches per critic optimizer step')
    # Plan turn-credit: the actor's advantages anchored at the seat's own-turn starts (value targets unchanged).
    parser.add_argument('--credit', choices=['decision', 'turn'], default='decision')
    parser.add_argument('--round-lambda', type=float, help='turn credit: geometric weight per further round')
    parser.add_argument('--rounds', type=int, help='turn credit: the first n rounds, undiscounted')
    parser.add_argument('--actor-accumulate', type=int, default=1,
                        help='plan critic-noise Amendment 2: minibatches per actor optimizer step')
    parser.add_argument('--critic-warmup', type=int, default=0,
                        help='critic-only iterations counted from the start of this run')
    parser.add_argument('--load-workers', type=int, default=0, help='processes decoding each batch (0 = serial)')
    parser.add_argument('--archive', type=Path, help='directory for periodic full-checkpoint copies (outside the spool)')
    parser.add_argument('--archive-minutes', type=float, default=0, help='minutes between archived checkpoints (0 = off)')
    # Plan critic-replay.
    parser.add_argument('--critic-replay-updates', type=int, default=0,
                        help='critic replay window in updates (0 = off)')
    parser.add_argument('--critic-replay-stride', type=int, default=8,
                        help='replay keeps the first decision and every k-th decision of each game')
    parser.add_argument('--critic-replay-mode', choices=['uniform', 'once'], default='uniform',
                        help='uniform: sample the window every update; once: every stored decision trained exactly once')
    parser.add_argument('--critic-replay-tokens', type=int, default=0,
                        help='microbatch token budget of the replay steps (0 = --critic-tokens)')
    parser.add_argument('--critic-replay-ratio', type=float, default=1.0,
                        help='replay rows per update, as a multiple of the update\'s new rows (256-row steps)')
    parser.add_argument('--infer-tokens', type=int, default=0,
                        help='token budget of the critic value pass (0 = --critic-tokens)')
    # Plan learner-speed (all off by default).
    parser.add_argument('--critic-tf32', action='store_true', help='S-1: TF32 matmuls for the fp32 critic only')
    parser.add_argument('--compile', choices=['off', 'tables'], default='off',
                        help='S-2: torch.compile the embedding-table builds of both models')
    parser.add_argument('--critic-stream', action='store_true',
                        help='S-3: the critic forward / backward on its own CUDA stream')
    parser.add_argument('--prefetch', action='store_true', help='S-4: decode the next batch during the update')
    parser.add_argument('--single-tables', action='store_true',
                        help='S-5 (user 2026-10-09): build each embedding table in one GEMM per layer (exports unchanged)')
    # Plan q-critic stage 2: the actor's advantage from an option-value network Q(s, o) (swccg_policy.qnet).
    parser.add_argument('--advantage', choices=['gae', 'q'], default='gae',
                        help='q: A_Q = Q(s, chosen) - sum pi Q(s, .) for the actor; the Q-net trains on the value targets')
    parser.add_argument('--qnet', type=Path,
                        help='initial Q-net (qnet.save_qnet); not needed when the checkpoint carries qnet_state')
    parser.add_argument('--qnet-lr', type=float, default=1e-4, help='AdamW rate of the Q-net (constant)')
    learn(parser.parse_args(argv))


if __name__ == '__main__':
    main()
