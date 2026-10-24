"""Plan critic-ceiling: the readouts of stages B, C and D2, the verdicts and the trigger.

    critic_ceiling_readout.py outcome --split R/cp.split.json --critic D=CKPT:critic --critic CA=CKPT:actor ... --out J
        Stage B readout 1 on the test half: explained variance against the game result, AUC at the first decision of a
        game, AUC by own-turn number (1-3, 4-6, 7-9, 10+), and turn starts (credit.turn_anchors) against the other
        own-turn decisions of the same turn buckets (bucket-averaged AUC, calibration); bootstrap over games.
    critic_ceiling_readout.py option --rows FILE ... --pair CP-D --pair CA-D --pair CP-CA --out J
        Stage B readout 2: the one-step option signal of each critic on E2 (advantage_readout.py power --rows) and the
        paired differences, one bootstrap over positions for all.
    critic_ceiling_readout.py verdict --fit R/cp.fit.json --fit R/ca.fit.json --outcome J --option J --out R/stageB.json
        The fixed verdicts (plan §1, rev. 2) and the trigger of stages C and D2.
    critic_ceiling_readout.py luck --a G --b G_B --rows FILE --qnet Q --checkpoint D --out J
        Stage C: the luck share between the fixed-order truth (a) and the reshuffled truth (b), every signal against
        both truths, the reshuffle check (report lines) and the exactness of both rollout sets.
    critic_ceiling_readout.py d2-positions --source EVAL --record DIR --states DIR --out G [--starts 60 --mid 60]
        Stage D2: turn-start and mid-turn positions of the source games, rollout jobs with option -2 (the AI answers).
    critic_ceiling_readout.py turnstart --a G --b G_B --critic NAME=CKPT:VIEW ... --out J
        Stage D2: V(s) against the truths V*(s) at turn starts and mid-turn; the verdict on the reshuffled truth.
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advantage_readout import (load_states, pearson, rollout_outcomes, spearman_brown,  # noqa: E402
                               true_advantages)

BUCKETS = ((1, 3), (4, 6), (7, 9), (10, 10 ** 6))
NOT_LEARNED_MIN = 0.25
TRIGGERED = 'C+D2'


# ---------------------------------------------------------------- statistics

def auc(scores, labels):
    """P(score of a win > score of a loss), ties counted half (Mann-Whitney); None without both classes."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    wins, losses = labels == 1, labels == 0
    if not wins.any() or not losses.any():
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2)[inverse]  # average ranks of ties, O(n log n)
    n1, n0 = wins.sum(), losses.sum()
    return float((ranks[wins].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def interval(values, minimum):
    return None if len(values) < minimum else [round(float(np.percentile(values, 2.5)), 4),
                                               round(float(np.percentile(values, 97.5)), 4)]


def bootstrap(statistic, units, boot, seed):
    """statistic(list of units) -> {key: value}; value and 95% interval per key, resampling the units."""
    main = statistic(units)
    rng = np.random.default_rng(seed)
    samples = collections.defaultdict(list)
    for _ in range(boot):
        result = statistic([units[i] for i in rng.integers(0, len(units), len(units))])
        for key, value in result.items():
            if value is not None:
                samples[key].append(value)
    return {key: {'value': None if value is None else round(float(value), 4), 'ci95': interval(samples[key], boot // 2)}
            for key, value in main.items()}


def disattenuated(r, reliability):
    return None if r is None or reliability is None or reliability <= 0 else r / math.sqrt(reliability)


def split_half(h1, h2):
    """Split-half Pearson of two halves, Spearman-Brown corrected to the full count."""
    half = pearson(h1, h2) if len(h1) >= 3 else None
    return None if half is None else spearman_brown(half)


# ---------------------------------------------------------------- models

def parse_critic(spec):
    """NAME=CKPT:VIEW (VIEW critic or actor; default critic)."""
    name, rest = spec.split('=', 1)
    path, _, view = rest.rpartition(':') if rest.endswith((':critic', ':actor')) else (rest, '', 'critic')
    return name, path, view


def critic_values(path, items_by_view, device='cuda'):
    """{view: values} of one checkpoint's critic (fp32, unpacked: as advantage_readout.py power)."""
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    actor, critic, _, _ = load(path, device)
    del actor
    critic.eval()
    critic.encoder.packed_attention = False
    out = {view: predict_values(critic, items, PPOConfig(bf16=False), 64) for view, items in items_by_view.items()}
    del critic
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- stage B: outcome readout

def decision_rows(games, view):
    """Per game: (items of the view, targets, turn numbers, own-turn flags, anchor flags)."""
    import critic_diagnostics as cd
    from critic_fit import outcome_target
    from swccg_policy.credit import HEAD_MY_TURN, HEAD_TURN, turn_anchors
    out = []
    for game in games:
        heads = [item['head'] for item in game['items']]
        anchors = set(turn_anchors(heads))
        out.append({'items': [cd.detach(item) for item in game['critic_items' if view == 'critic' else 'items']],
                    'target': outcome_target(game), 'turn': np.array([int(h[HEAD_TURN]) for h in heads]),
                    'own': np.array([int(h[HEAD_MY_TURN]) == 1 for h in heads]),
                    'anchor': np.array([i in anchors for i in range(len(heads))])})
    return out


def bucket_of(turn):
    return next((i for i, (lo, hi) in enumerate(BUCKETS) if lo <= turn <= hi), None)


def outcome_statistic(games):
    """games: dicts with values (per decision), target, turn, own, anchor."""
    values = np.concatenate([g['values'] for g in games])
    targets = np.concatenate([np.full(len(g['values']), g['target']) for g in games])
    out = {'ev': 1 - float(((values - targets) ** 2).mean()) / float(targets.var()) if targets.var() > 0 else None,
           'mse': float(((values - targets) ** 2).mean())}
    decided = [g for g in games if g['target'] != 0]
    out['auc_first'] = auc([g['values'][0] for g in decided], [int(g['target'] > 0) for g in decided])
    turns = np.concatenate([g['turn'] for g in games])
    own = np.concatenate([g['own'] for g in games])
    anchor = np.concatenate([g['anchor'] for g in games])
    labels = (targets > 0).astype(int)
    keep = targets != 0
    for b, (lo, hi) in enumerate(BUCKETS):
        mask = keep & (turns >= lo) & (turns <= hi)
        out[f'auc_turn_{lo}_{hi if hi < 10 ** 6 else "up"}'] = auc(values[mask], labels[mask]) if mask.any() else None
    by_set = {}
    for name, members in (('start', anchor), ('other', own & ~anchor)):
        per_bucket = []
        for lo, hi in BUCKETS:
            mask = keep & members & (turns >= lo) & (turns <= hi)
            per_bucket.append(auc(values[mask], labels[mask]) if mask.any() else None)
        usable = [a for a in per_bucket if a is not None]
        by_set[name] = per_bucket
        out[f'auc_{name}'] = float(np.mean(usable)) if len(usable) == len(BUCKETS) else None
        mask = members & (turns >= 1)
        out[f'calibration_{name}'] = float((values[mask] - targets[mask]).mean()) if mask.any() else None
    out['auc_start_minus_other'] = (None if out['auc_start'] is None or out['auc_other'] is None
                                    else out['auc_start'] - out['auc_other'])
    return out


def cmd_outcome(args):
    """Streams the test files (one file decoded at a time; every critic loaded once), so a test set of thousands of
    games fits in memory."""
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    from swccg_policy.v4_trajectory import TrajectoryReader
    split = json.loads(Path(args.split).read_text())
    critics = [parse_critic(spec) for spec in args.critic]
    models, tables = [], None
    for name, path, view in critics:
        actor, critic, tables, _ = load(path, 'cuda', tables=tables)
        del actor
        critic.eval()
        critic.encoder.packed_attention = False
        models.append((name, path, view, critic))
    reader = TrajectoryReader(tables, [])
    views = sorted({view for _, _, view in critics})
    per_critic = {name: [] for name, _, _ in critics}
    games_seen = decisions = 0
    for path in split['test']:
        games = reader.load(path)
        rows = {view: decision_rows(games, view) for view in views}
        games_seen += len(games)
        decisions += sum(len(g['items']) for g in games)
        for name, _, view, critic in models:
            flat = [item for g in rows[view] for item in g['items']]
            values = predict_values(critic, flat, PPOConfig(bf16=False), 64)
            offsets = np.cumsum([0] + [len(g['items']) for g in rows[view]])
            per_critic[name].extend({**{k: g[k] for k in ('target', 'turn', 'own', 'anchor')},
                                     'values': values[offsets[i]:offsets[i + 1]]} for i, g in enumerate(rows[view]))
        del games, rows
    del models
    torch.cuda.empty_cache()
    first = per_critic[critics[0][0]]
    out = {'test_files': len(split['test']), 'games': games_seen, 'decisions': decisions,
           'games_decided': sum(1 for g in first if g['target'] != 0), 'boot': args.boot, 'critics': {}}
    for name, path, view in critics:
        out['critics'][name] = {'checkpoint': path, 'view': view,
                                **bootstrap(outcome_statistic, per_critic[name], args.boot, args.seed)}
        print(name, json.dumps(out['critics'][name]), flush=True)
    out['paired'] = {}
    for pair in args.pair or []:
        a, b = pair.split('-')
        if a in per_critic and b in per_critic:
            units = list(zip(per_critic[a], per_critic[b]))
            def statistic(chosen):
                left, right = outcome_statistic([u[0] for u in chosen]), outcome_statistic([u[1] for u in chosen])
                return {key: None if left[key] is None or right[key] is None else left[key] - right[key]
                        for key in ('ev', 'mse', 'auc_first', 'auc_start', 'auc_other', 'auc_start_minus_other')}
            out['paired'][pair] = bootstrap(statistic, units, args.boot, args.seed + 1)
            print(pair, json.dumps(out['paired'][pair]), flush=True)
    out['order'] = [name for name, _, _ in critics]
    Path(args.out).write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- stage B: option signal

def option_units(rows_by_name):
    """Per position present for every name: {'a_star', 'h1', 'h2', signals {name: raw adv}, 'centred' {name: ...}}."""
    from advantage_readout import power_pairs
    names = list(rows_by_name)
    by_id = {name: {row['id']: row for row in rows} for name, rows in rows_by_name.items()}
    ids = [row['id'] for row in rows_by_name[names[0]] if all(row['id'] in by_id[n] for n in names)]
    units = []
    for state_id in ids:
        pairs = {name: power_pairs([by_id[name][state_id]]) for name in names}
        if any(not p for p in pairs.values()):
            continue
        first = pairs[names[0]][0]
        if any(len(pairs[n][0]['a_star']) != len(first['a_star']) for n in names):
            continue
        units.append({'id': state_id, 'a_star': first['a_star'], 'h1': first['h1'], 'h2': first['h2'],
                      'signals': {n: pairs[n][0]['adv'] for n in names},
                      'centred': {n: pairs[n][0]['centred'] for n in names}})
    return units


def signal_statistic(units, names, pairs, truth='a_star', h=('h1', 'h2'), keys=('signals', 'centred')):
    """Pooled disattenuated Pearson of every signal with the truth, and the paired differences."""
    a_star = np.concatenate([u[truth] for u in units])
    split = [u for u in units if u[h[0]] is not None and u[h[1]] is not None]
    reliability = split_half(np.concatenate([u[h[0]] for u in split]), np.concatenate([u[h[1]] for u in split])) \
        if split else None
    out = {'reliability': reliability}
    for key in keys:
        suffix = '' if key == 'signals' else '_centred'
        for name in names:
            out[name + suffix] = disattenuated(pearson(np.concatenate([u[key][name] for u in units]), a_star),
                                               reliability)
        for pair in pairs:
            a, b = pair.split('-')
            out[pair + suffix] = None if out[a + suffix] is None or out[b + suffix] is None \
                else out[a + suffix] - out[b + suffix]
    return out


def pooled_rows(specs, pref=None):
    """Rows of several E2-like sets: each spec is [PREFIX=]FILE; a name's positions from every set are pooled, with
    position ids prefixed by the set (plan critic-ceiling-2). pref NAME adds NAME-pref: D's own preferences (pi-centred
    log pi) as a one-step signal, from the pi of NAME's rows."""
    rows = collections.defaultdict(list)
    for spec in specs:
        prefix, _, path = spec.rpartition('=')
        for name, entries in json.loads(Path(path).read_text()).items():
            rows[name].extend({**entry, 'id': f"{prefix}:{entry['id']}" if prefix else entry['id']}
                              for entry in entries)
    if pref:
        rows[pref + '-pref'] = [{**entry, 'adv': [float(x) for x in preferences(entry['pi'])]} for entry in rows[pref]]
    return dict(rows)


def cmd_option(args):
    rows = pooled_rows(args.rows, args.pref)
    units = option_units(rows)
    names = list(rows)
    result = bootstrap(lambda chosen: signal_statistic(chosen, names, args.pair), units, args.boot, args.seed)
    out = {'positions': len(units), 'pairs': int(sum(len(u['a_star']) for u in units)), 'boot': args.boot,
           'names': names, 'paired': args.pair, 'readout': result}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


# ---------------------------------------------------------------- stage B: verdicts

def verdict(cp_value, cp_minus_d_ci):
    """Plan §1 (rev. 2): exactly one of 'not learned', 'partly learned', 'ceiling holds'."""
    if cp_minus_d_ci is not None and cp_minus_d_ci[0] > 0:
        return 'not learned' if cp_value is not None and cp_value >= NOT_LEARNED_MIN else 'partly learned'
    return 'ceiling holds'


def privileged(cp_minus_ca_ci):
    if cp_minus_ca_ci is None:
        return 'no difference'
    return 'helps' if cp_minus_ca_ci[0] > 0 else 'hurts' if cp_minus_ca_ci[1] < 0 else 'no difference'


def cmd_verdict(args):
    option = json.loads(Path(args.option).read_text())['readout']
    fits = {Path(path).name.split('.')[0]: json.loads(Path(path).read_text()) for path in args.fit}
    cp_value = option['CP']['value']
    ci = option['CP-D']['ci95']
    result = verdict(cp_value, ci)
    out = {'verdict': result, 'trigger': 'none' if result == 'not learned' else TRIGGERED,
           'privileged_view': privileged(option['CP-CA']['ci95']),
           'option_signal': {key: option[key] for key in ('D', 'CP', 'CA', 'CP-D', 'CA-D', 'CP-CA', 'reliability')},
           'resolution_cp_minus_d': None if ci is None else round((ci[1] - ci[0]) / 2, 4),
           'fits': {name: {key: fit[key] for key in ('converged', 'scored_minute', 'scored_mse', 'steps', 'decisions')}
                    for name, fit in fits.items()},
           'rules': {'not learned': f'CP - D interval above 0 and CP >= {NOT_LEARNED_MIN}',
                     'partly learned': f'CP - D interval above 0 and CP < {NOT_LEARNED_MIN}',
                     'ceiling holds': 'CP - D interval not above 0',
                     'trigger': 'C and D2 run unless "not learned"'}}
    if args.outcome:
        out['outcome'] = json.loads(Path(args.outcome).read_text())
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({key: out[key] for key in ('verdict', 'trigger', 'privileged_view', 'resolution_cp_minus_d')}))


# ---------------------------------------------------------------- plan critic-ceiling-2: verdicts

E2_D_INTERVAL = (0.024, 0.254)  # D's option signal on E2 (critic-ceiling stage B), the replication band
TRANSFER_MAX = 0.10


def gate_passed(cp2_minus_d_ev_ci):
    """A better critic exists: CP2's explained variance on test set B above D's, paired interval above 0."""
    return cp2_minus_d_ev_ci is not None and cp2_minus_d_ev_ci[0] > 0


def verdict2(gate, pooled_ci):
    """Plan critic-ceiling-2 §1, V2 (on the pooled CP2 - D option-signal interval)."""
    if not gate:
        return 'no better critic produced'
    if pooled_ci is None:
        return 'inconclusive'
    if pooled_ci[0] > 0:
        return 'critic quality transfers'
    if pooled_ci[1] < TRANSFER_MAX:
        return 'critic quality does not transfer to options'
    return 'inconclusive'


def replicates(d_e3, band=E2_D_INTERVAL):
    return d_e3 is not None and band[0] <= d_e3 <= band[1]


def cmd_verdict2(args):
    outcome = json.loads(Path(args.outcome).read_text())
    pooled = json.loads(Path(args.pooled).read_text())['readout']
    e3 = json.loads(Path(args.e3).read_text())['readout'] if args.e3 and Path(args.e3).exists() else None
    gate_ci = outcome['paired']['CP2-D']['ev']['ci95']
    gate = gate_passed(gate_ci)
    out = {'gate': {'passed': gate, 'cp2_minus_d_ev': outcome['paired']['CP2-D']['ev'],
                    'd_ev': outcome['critics']['D']['ev'], 'cp2_ev': outcome['critics']['CP2']['ev'],
                    'test_games': outcome['games']},
           'v2': verdict2(gate, pooled['CP2-D']['ci95']), 'pooled_cp2_minus_d': pooled['CP2-D'],
           'pooled': {k: pooled[k] for k in pooled if not k.endswith('_centred')},
           'v1': None if e3 is None else {'d_e3': e3['D'], 'replicates': replicates(e3['D']['value']),
                                          'band': list(E2_D_INTERVAL)},
           'rules': {'gate': 'CP2 - D explained variance on test set B, paired interval above 0',
                     'v1': f'D on E3 inside {list(E2_D_INTERVAL)}',
                     'v2': f'gate passed; transfers if pooled CP2 - D interval above 0; does not transfer if not '
                           f'above 0 and upper bound < {TRANSFER_MAX}; else inconclusive'}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({'gate': gate, 'v2': out['v2'], 'v1': out['v1']}))


# ---------------------------------------------------------------- stages C and D2: reports

def report_lines(gt):
    """{global job index: {'mismatch': bool, 'end': (status, winner) or None, 'reshuffle': [...] or None}}."""
    out = collections.defaultdict(lambda: {'mismatch': False, 'end': None, 'reshuffle': None})
    for path in sorted(glob.glob(os.path.join(gt, 'report-gt.c*.txt'))) or [os.path.join(gt, 'report-gt.txt')]:
        for line in Path(path).read_text().splitlines():
            parts = line.split('\t')
            g = int(parts[0].rsplit('-', 1)[1])
            if parts[1] == 'mismatch':
                out[g]['mismatch'] = True
            elif parts[1] == 'end' and out[g]['end'] is None:
                out[g]['end'] = (parts[3], parts[4])
            elif parts[1] == 'reshuffle':
                out[g]['reshuffle'] = parse_reshuffle(parts[2], parts[3])
    return out


def parse_reshuffle(scope, text):
    """[(owner, pile, size, same cards, same order, known)] of one reshuffle report line."""
    rows = []
    for entry in text.split():
        owner, pile, size, cards, order, known = entry.split(':')
        rows.append((owner, pile, int(size), cards == '1', order == '1', int(known)))
    return {'scope': scope, 'piles': rows}


def reshuffle_check(lines):
    """Plan §3 block 6: the named piles keep their cards (order free), every other pile is unchanged; the share of
    named piles (2+ cards) whose order changed; the share of jobs where the branching seat knew a position there."""
    named_ok = other_ok = changed = multi = known_jobs = jobs = 0
    bad = []
    for g, line in lines.items():
        info = line['reshuffle']
        if info is None:
            continue
        jobs += 1
        knew = False
        for owner, pile, size, same_cards, same_order, known in info['piles']:
            named = pile in ('RESERVE_DECK', 'FORCE_PILE') and (owner == 'self' or info['scope'] == 'both')
            if named:
                named_ok += same_cards
                if size >= 2:
                    multi += 1
                    changed += not same_order
                if known:
                    knew = True
                if not same_cards:
                    bad.append(g)
            else:
                other_ok += same_cards and same_order
                if not (same_cards and same_order):
                    bad.append(g)
        known_jobs += knew
    return {'jobs': jobs, 'violations': len(set(bad)), 'named_piles_order_changed_share':
            None if not multi else round(changed / multi, 4),
            'jobs_seat_knew_a_position_share': None if not jobs else round(known_jobs / jobs, 4),
            'ok': jobs > 0 and not bad}


def exactness(lines, planned):
    exact = sum(1 for line in lines.values() if line['end'] is not None and not line['mismatch'])
    return {'planned': planned, 'ended': sum(1 for line in lines.values() if line['end'] is not None),
            'exact': exact, 'exact_share': round(exact / planned, 4) if planned else None}


# ---------------------------------------------------------------- stage C: luck

def preferences(pi):
    """D's own option preferences: log pi, pi-centred (the centred logits)."""
    pi = np.clip(np.asarray(pi, float), 1e-12, None)
    pi = pi / pi.sum()
    return np.log(pi) - float(pi @ np.log(pi))


def truth_by_option(gt):
    """{position: {option: [(rollout, won)]}} of a ground-truth folder (outcome states give the side, as E2)."""
    _, rows = rollout_outcomes(gt)
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, position, option, rollout, won in rows:
        table[position][option].append((rollout, won))
    return table


def luck_units(listed, rows_by_name, truth_a, truth_b, extra):
    """Per position with >= 2 options having rollouts in both truths: A*, halves per truth, and every signal."""
    units = []
    by_id = {name: {row['id']: row for row in rows} for name, rows in rows_by_name.items()}
    for r in listed:
        p = r['position']
        options = [o for o in range(r['options']) if truth_a[p].get(o) and truth_b[p].get(o)
                   and all(by_id[n].get(r['id']) and by_id[n][r['id']]['adv'][o] is not None for n in by_id)
                   and all(extra[n].get(r['id']) is not None for n in extra)]
        if len(options) < 2:
            continue
        pi = [r['pi'][o] for o in options]
        unit = {'id': r['id'], 'template': r.get('template'), 'options': options}
        for key, truth in (('a', truth_a), ('b', truth_b)):
            wins = [[w for _, w in truth[p][o]] for o in options]
            unit[f'a_star_{key}'] = true_advantages(pi, wins)[1]
            halves = []
            for parity in (0, 1):
                part = [[w for k, w in truth[p][o] if k % 2 == parity] for o in options]
                halves.append(None if any(not w for w in part) else true_advantages(pi, part)[1])
            unit[f'h1_{key}'], unit[f'h2_{key}'] = halves
        weights = np.asarray(pi, float) / np.sum(pi)
        signals = {}
        for name in by_id:
            adv = np.array([by_id[name][r['id']]['adv'][o] for o in options], float)
            signals[name] = adv - float(weights @ adv)
        for name, values in extra.items():
            vector = np.array([values[r['id']][o] for o in options], float)
            signals[name] = vector - float(weights @ vector)
        unit['signals'] = signals
        units.append(unit)
    return units


def luck_statistic(units, names):
    out = {}
    a = np.concatenate([u['a_star_a'] for u in units])
    b = np.concatenate([u['a_star_b'] for u in units])
    reliability = {}
    for key in ('a', 'b'):
        split = [u for u in units if u[f'h1_{key}'] is not None and u[f'h2_{key}'] is not None]
        reliability[key] = split_half(np.concatenate([u[f'h1_{key}'] for u in split]),
                                      np.concatenate([u[f'h2_{key}'] for u in split])) if split else None
        out[f'reliability_{key}'] = reliability[key]
    r_ab = pearson(a, b)
    out['corr_a_b'] = r_ab
    both = None if reliability['a'] is None or reliability['b'] is None or reliability['a'] * reliability['b'] <= 0 \
        else math.sqrt(reliability['a'] * reliability['b'])
    out['corr_a_b_disattenuated'] = None if r_ab is None or both is None else r_ab / both
    out['luck_share'] = None if out['corr_a_b_disattenuated'] is None else 1 - out['corr_a_b_disattenuated']
    out['mean_abs_a_minus_b'] = float(np.mean(np.abs(a - b)))
    for name in names:
        signal = np.concatenate([u['signals'][name] for u in units])
        ra = disattenuated(pearson(signal, a), reliability['a'])
        rb = disattenuated(pearson(signal, b), reliability['b'])
        out[f'{name}_a'], out[f'{name}_b'] = ra, rb
        out[f'{name}_a_minus_b'] = None if ra is None or rb is None else ra - rb
    return out


def cmd_luck(args):
    import torch
    spec = json.loads(Path(args.a, 'positions.json').read_text())
    listed = spec['list']
    rows = {}
    for path in args.rows:
        rows.update(json.loads(Path(path).read_text()))
    truth_a, truth_b = truth_by_option(args.a), truth_by_option(args.b)
    extra = {'D-pref': {r['id']: preferences(r['pi']) for r in listed}}
    if args.qnet:
        import critic_probes as cp
        from swccg_policy.qnet import load_qnet, option_values
        from swccg_policy.v4_checkpoint import load
        from swccg_policy.v4_data import stack
        base = load(args.checkpoint, 'cuda')
        qnet, _ = load_qnet(args.qnet, base, 'cuda')
        qnet.packed_attention = False
        states = load_states(spec['states'], base[2], {r['id'] for r in listed})
        order = [r['id'] for r in listed if r['id'] in states]
        values = option_values(qnet, [states[i][0] for i in order], cp.batches, stack)
        extra['Qnet'] = {i: np.asarray(v, float) for i, v in zip(order, values)}
        del qnet, base
        torch.cuda.empty_cache()
    units = luck_units(listed, rows, truth_a, truth_b, extra)
    names = [*rows, *extra]
    lines_a, lines_b = report_lines(args.a), report_lines(args.b)
    planned = spec['rollouts']
    out = {'positions': len(listed), 'positions_scored': len(units), 'pairs': int(sum(len(u['options']) for u in units)),
           'templates': dict(collections.Counter(u['template'] for u in units)), 'names': names, 'boot': args.boot,
           'exactness_a': exactness(lines_a, planned), 'exactness_b': exactness(lines_b, planned),
           'reshuffle_check': reshuffle_check(lines_b),
           'readout': bootstrap(lambda chosen: luck_statistic(chosen, names), units, args.boot, args.seed),
           'rule': '"graded on hindsight" if a signal correlates more with (a) than with (b): paired interval above 0'}
    out['graded_on_hindsight'] = {name: bool(out['readout'][f'{name}_a_minus_b']['ci95'] is not None
                                             and out['readout'][f'{name}_a_minus_b']['ci95'][0] > 0) for name in names}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({key: out[key] for key in ('positions_scored', 'exactness_a', 'exactness_b', 'reshuffle_check',
                                                'graded_on_hindsight')}))


# ---------------------------------------------------------------- stage D2: positions and readout

def turn_start_candidates(rows, turns):
    """Own-turn position states per seat of a game: (starts, mids) within the turn range. A start is the first
    recorded own-turn decision of each turn number (as credit.turn_anchors on the seat's decisions)."""
    seats = collections.defaultdict(list)
    for r in rows:
        if r['own']:
            seats[(r['game'], r['side'])].append(r)
    starts, mids = collections.defaultdict(list), collections.defaultdict(list)
    for (game, _), seat in seats.items():
        last = None
        for r in sorted(seat, key=lambda row: row['index']):
            is_start = last is None or r['turn'] > last
            if is_start:
                last = r['turn']
            if turns[0] <= r['turn'] <= turns[1]:
                (starts if is_start else mids)[game].append(r)
    return starts, mids


def spread_pick(pool, games, count, per_game, rng):
    """Round robin over games, at most per_game per game, preferring turn numbers not yet taken in that game."""
    chosen, taken = [], collections.defaultdict(set)
    for _ in range(per_game):
        for g in games:
            if len(chosen) >= count:
                return chosen
            fresh = [r for r in pool[g] if r['turn'] not in taken[g]]
            options = fresh or pool[g]
            if options:
                r = options[rng.randrange(len(options))]
                pool[g].remove(r)
                taken[g].add(r['turn'])
                chosen.append(r)
    return chosen


def cmd_d2_positions(args):
    schedule = Path(args.source, 'schedule.tsv').read_text().splitlines()
    seed = int(json.loads(Path(args.source, 'results.json').read_text())['seed'])
    records = {int(Path(f).stem.rsplit('-', 1)[1]): os.path.abspath(f) for f in glob.glob(os.path.join(args.record, '*.tsv'))}
    rows = [json.loads(line) for line in Path(args.states, 'states.jsonl').read_text().splitlines() if line]
    rows = [r for r in rows if r['game'] in records]
    starts, mids = turn_start_candidates(rows, args.turns)
    rng = random.Random(args.seed)
    games = sorted(g for g in starts if starts[g])
    rng.shuffle(games)
    chosen_starts = spread_pick(starts, games, args.starts, args.per_game, rng)
    used = sorted({r['game'] for r in chosen_starts})
    rng.shuffle(used)
    chosen_mids = spread_pick(mids, used, args.mid, args.per_game, rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    listed, lines, sched, meta = [], [], [], []
    for group, chosen in (('start', chosen_starts), ('mid', chosen_mids)):
        for r in chosen:
            p = len(listed)
            entry = {**r, 'group': group, 'position': p, 'seed': seed + r['game'], 'record': records[r['game']],
                     'schedule': schedule[r['game']]}
            listed.append(entry)
            for rollout in range(args.rollouts):
                lines.append(f"{entry['seed']}\t{entry['record']}\t{r['index']}\t-2\t0")
                sched.append(entry['schedule'])
                meta.append({'position': p, 'option': -2, 'rollout': rollout})
    (out / 'jobs-gt.tsv').write_text('\n'.join(lines) + '\n')
    (out / 'sched-gt.tsv').write_text('\n'.join(sched) + '\n')
    (out / 'meta-gt.json').write_text(json.dumps(meta))
    summary = {'source': str(args.source), 'states': str(args.states), 'starts': len(chosen_starts),
               'mid': len(chosen_mids), 'games': len(used), 'rollouts_per_position': args.rollouts,
               'rollouts': len(lines), 'selection': {'turns': args.turns, 'per_game': args.per_game, 'seed': args.seed},
               'start_candidates': sum(len(v) for v in starts.values()) + len(chosen_starts),
               'mid_candidates': sum(len(v) for v in mids.values()) + len(chosen_mids)}
    (out / 'positions.json').write_text(json.dumps({**summary, 'list': listed}, indent=1))
    print(json.dumps(summary))


def position_outcomes(gt, listed):
    """{position: [(rollout, won)]} from the report's end lines (the side is the position's: no state needed)."""
    meta = json.loads(Path(gt, 'meta-gt.json').read_text())
    side = {r['position']: r['side'] for r in listed}
    out = collections.defaultdict(list)
    for g, line in report_lines(gt).items():
        if line['end'] is None or line['mismatch'] or g >= len(meta):
            continue
        status, winner = line['end']
        p = meta[g]['position']
        out[p].append((meta[g]['rollout'], int(status == 'finished' and winner == side[p])))
    return out


def turnstart_statistic(units, names):
    """Per group (start / mid) and truth (a / b): reliability, disattenuated Pearson(V, V*), calibration; the
    start - mid differences."""
    out = {}
    for truth in ('a', 'b'):
        for group in ('start', 'mid'):
            chosen = [u for u in units if u['group'] == group and u[f'v_{truth}'] is not None]
            if len(chosen) < 3:
                continue
            target = np.array([u[f'v_{truth}'] for u in chosen])
            split = [u for u in chosen if u[f'h_{truth}'] is not None]
            reliability = split_half([u[f'h_{truth}'][0] for u in split], [u[f'h_{truth}'][1] for u in split])
            out[f'reliability_{group}_{truth}'] = reliability
            for name in names:
                value = np.array([u['values'][name] for u in chosen])
                out[f'{name}_{group}_{truth}'] = disattenuated(pearson(value, target), reliability)
                out[f'{name}_calibration_{group}_{truth}'] = float((value - target).mean())
        for name in names:
            s, m = out.get(f'{name}_start_{truth}'), out.get(f'{name}_mid_{truth}')
            out[f'{name}_start_minus_mid_{truth}'] = None if s is None or m is None else s - m
    return out


def cmd_turnstart(args):
    spec = json.loads(Path(args.a, 'positions.json').read_text())
    listed = spec['list']
    outcomes = {'a': position_outcomes(args.a, listed), 'b': position_outcomes(args.b, listed)}
    critics = [parse_critic(item) for item in args.critic]
    from swccg_policy.v4_checkpoint import load
    _, _, tables, _ = load(critics[0][1], 'cpu')
    states = load_states(spec['states'], tables, {r['id'] for r in listed})
    present = [r for r in listed if r['id'] in states]
    values = {}
    for name, path, view in critics:
        items = [states[r['id']][0 if view == 'actor' else 1] for r in present]
        values[name] = dict(zip([r['id'] for r in present], critic_values(path, {view: items})[view]))
    units = []
    for r in present:
        unit = {'id': r['id'], 'game': r['game'], 'group': r['group'], 'turn': r['turn'],
                'values': {name: float(values[name][r['id']]) for name in values}}
        for truth in ('a', 'b'):
            results = outcomes[truth].get(r['position'], [])
            unit[f'v_{truth}'] = 2 * float(np.mean([w for _, w in results])) - 1 if results else None
            halves = [[w for k, w in results if k % 2 == parity] for parity in (0, 1)]
            unit[f'h_{truth}'] = None if any(not h for h in halves) else (2 * np.mean(halves[0]) - 1,
                                                                            2 * np.mean(halves[1]) - 1)
        units.append(unit)
    names = list(values)
    by_game = collections.defaultdict(list)  # bootstrap over games: starts and mids of a game stay together
    for unit in units:
        by_game[unit['game']].append(unit)
    games = list(by_game.values())
    readout = bootstrap(lambda chosen: turnstart_statistic([u for g in chosen for u in g], names), games, args.boot,
                        args.seed)
    calibration_by_turn = {}
    for name in names:
        for group in ('start', 'mid'):
            for lo, hi in ((3, 5), (6, 8), (9, 12)):
                chosen = [u for u in units if u['group'] == group and lo <= u['turn'] <= hi and u['v_b'] is not None]
                if chosen:
                    calibration_by_turn[f'{name}_{group}_{lo}_{hi}'] = round(
                        float(np.mean([u['values'][name] - u['v_b'] for u in chosen])), 4)
    lines_a, lines_b = report_lines(args.a), report_lines(args.b)
    planned = spec['rollouts']
    out = {'positions': len(listed), 'scored': len(units), 'games': len(games), 'names': names, 'boot': args.boot,
           'exactness_a': exactness(lines_a, planned), 'exactness_b': exactness(lines_b, planned),
           'reshuffle_check': reshuffle_check(lines_b), 'readout': readout,
           'calibration_by_turn_b': calibration_by_turn,
           'rule': '"the critic misreads turn starts" if its turn-start correlation is lower than its mid-turn one '
                   'with the paired interval below 0 (truth b, reshuffled)'}
    out['misreads_turn_starts'] = {name: bool(readout.get(f'{name}_start_minus_mid_b', {}).get('ci95') is not None
                                              and readout[f'{name}_start_minus_mid_b']['ci95'][1] < 0)
                                   for name in names}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({key: out[key] for key in ('scored', 'exactness_a', 'exactness_b', 'reshuffle_check',
                                                'misreads_turn_starts')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    outcome = sub.add_parser('outcome')
    outcome.add_argument('--split', required=True)
    outcome.add_argument('--critic', action='append', required=True, help='NAME=CKPT:VIEW')
    outcome.add_argument('--pair', action='append', help='A-B paired differences')
    option = sub.add_parser('option')
    option.add_argument('--rows', action='append', required=True, help='[PREFIX=]FILE; one prefix per set')
    option.add_argument('--pair', action='append', default=[])
    option.add_argument('--pref', help='add NAME-pref, the preferences of the policy behind NAME\'s rows')
    verdict_parser = sub.add_parser('verdict')
    verdict_parser.add_argument('--fit', action='append', required=True)
    verdict_parser.add_argument('--option', required=True)
    verdict_parser.add_argument('--outcome')
    luck = sub.add_parser('luck')
    luck.add_argument('--a', required=True, help='the fixed-order ground-truth folder (positions.json, signals)')
    luck.add_argument('--b', required=True, help='the reshuffled ground-truth folder')
    luck.add_argument('--rows', action='append', required=True, help='advantage_readout.py power --rows of (a)')
    luck.add_argument('--qnet')
    luck.add_argument('--checkpoint')
    d2 = sub.add_parser('d2-positions')
    d2.add_argument('--source', required=True)
    d2.add_argument('--record', required=True)
    d2.add_argument('--states', required=True)
    d2.add_argument('--out', required=True)
    d2.add_argument('--starts', type=int, default=60)
    d2.add_argument('--mid', type=int, default=60)
    d2.add_argument('--per-game', type=int, default=2)
    d2.add_argument('--rollouts', type=int, default=48)
    d2.add_argument('--turns', type=int, nargs=2, default=[3, 12])
    d2.add_argument('--seed', type=int, default=20261020)
    v2 = sub.add_parser('verdict2')
    v2.add_argument('--outcome', required=True)
    v2.add_argument('--pooled', required=True)
    v2.add_argument('--e3')
    turnstart = sub.add_parser('turnstart')
    turnstart.add_argument('--a', required=True)
    turnstart.add_argument('--b', required=True)
    turnstart.add_argument('--critic', action='append', required=True, help='NAME=CKPT:VIEW')
    for command in (outcome, option, verdict_parser, luck, turnstart, v2):
        command.add_argument('--out', required=True)
        command.add_argument('--boot', type=int, default=2000)
        command.add_argument('--seed', type=int, default=20261020)
    args = parser.parse_args()
    {'outcome': cmd_outcome, 'option': cmd_option, 'verdict': cmd_verdict, 'luck': cmd_luck,
     'd2-positions': cmd_d2_positions, 'turnstart': cmd_turnstart, 'verdict2': cmd_verdict2}[args.command](args)


if __name__ == '__main__':
    main()
