"""Plan probe-economics: what has D learnt (stage 1), how much signal do averaged playouts carry (stage 2), and the
signal per unit of compute (stage 4).

    probe_readout.py learnt --set e2=ROWS:POSITIONS:RECORDS --set e3=... --checkpoint D --out J
        Per E2-like position, cross-fitted over the two rollout halves: the best option on one half, evaluated on the
        other. D's regret (win-rate points it gives up against that option, under pi) and top-1, by the half-1 gap
        (best minus second best: < 5, 5-15, >= 15 points) and by decision kind; H1 and H2 (plan §1).
    probe_readout.py playouts --rows STAGE_A_ROWS --truth ROWS:POSITIONS:RECORDS --checkpoint D --out J
        Per (position, option): the mean of N stage-A rollouts' advantage per horizon (gae-0, tc-r1..3, gae-1, and
        gae-0.8), pi-centred, against E2's truth (disattenuated), N = 1..24 by random subsets; the intra-option
        correlation rho per horizon and the ceiling r1 / sqrt(rho); S1 and S2.
    probe_readout.py economics --playouts J --costs J --out J
        Signal per game-equivalent of compute: plain PPO against probing at each (horizon, N); go / no-go.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advantage_readout import decision_texts, load_states, pearson, spearman_brown  # noqa: E402

GAP_BINS = ((0.0, 0.05), (0.05, 0.15), (0.15, 9.0))
HORIZONS = ('gae-0', 'tc-r1', 'tc-r2', 'tc-r3', 'gae-1', 'gae-0.8')
NS = (1, 2, 4, 8, 12, 24)
PPO_R, PPO_DECISIONS = 0.06, 260  # plain PPO: per-sample signal at lambda 0.95 (turn-credit stage A), decisions/game
S1_MIN, S1_LOWER, GO_R, GO_DECISIONS = 0.40, 0.25, 0.40, 100000


# ---------------------------------------------------------------- decision kinds

def clean(text):
    return re.sub(r"<div class='cardHint'[^>]*>([^<]*)</div>", '[card]', text or '').strip()


def kind_of(text, decision_type):
    """(kind, phase): act-or-pass decisions ('Choose X action or Pass', phase X), deploy-location choices, other card
    selections, other."""
    text = clean(text)
    match = re.match(r'Choose (\w+) action or Pass', text) or re.match(r'Choose (\w+) segment action to play or Pass', text)
    if match:
        return 'act-or-pass', match.group(1).lower()
    if 'deploy' in text.lower() and decision_type in ('CARD_SELECTION', 'MULTIPLE_CHOICE'):
        return 'deploy-where', None
    if decision_type == 'CARD_SELECTION':
        return 'card-selection', None
    return 'other', None


def load_set(spec, tables):
    """NAME=ROWS:POSITIONS:RECORDS -> positions with pi, outcomes per option, kind, phase and the options' pass flags.
    ROWS: advantage_readout.py power --rows (any name: the outcomes are the same), POSITIONS: positions.json, RECORDS:
    the record folder (the positions' record paths are re-rooted there)."""
    name, _, rest = spec.partition('=')
    rows_path, positions_path, records = rest.split(':')
    rows = next(iter(json.loads(Path(rows_path).read_text()).values()))
    listed = {r['id']: r for r in json.loads(Path(positions_path).read_text())['list']}
    states_dir = json.loads(Path(positions_path).read_text())['states']
    states = load_states(states_dir, tables, set(listed))
    texts = {}
    out = []
    for row in rows:
        r = listed[row['id']]
        record = os.path.join(records, r['record'].rsplit('/', 1)[1])
        if record not in texts:
            texts[record] = decision_texts(record)
        text = texts[record].get(r['index'], '')
        kind, phase = kind_of(text, r['type'])
        passes = [int(x) for x in states[row['id']][0]['opt'][:, 5]] if row['id'] in states else [0] * r['options']
        out.append({'set': name, 'id': row['id'], 'position': r['position'], 'pi': row['pi'],
                    'outcomes': row['outcomes'], 'kind': kind, 'phase': phase, 'text': clean(text), 'pass': passes})
    return out


# ---------------------------------------------------------------- stage 1

def halves(outcomes):
    """Per option: (win rate on even rollouts, on odd rollouts) or None where a half is empty."""
    out = []
    for results in outcomes:
        even = [w for k, w in results if k % 2 == 0]
        odd = [w for k, w in results if k % 2 == 1]
        out.append((np.mean(even), np.mean(odd)) if even and odd else None)
    return out


def folds(position):
    """Two cross-fitted units of one position: best and gap on one half, regret and top-1 on the other."""
    rates = halves(position['outcomes'])
    usable = [o for o, r in enumerate(rates) if r is not None]
    if len(usable) < 2:
        return []
    pi = np.array([position['pi'][o] for o in usable], float)
    pi = pi / pi.sum()
    choice = usable[int(np.argmax(pi))]
    units = []
    for pick, judge in ((0, 1), (1, 0)):
        pick_rates = np.array([rates[o][pick] for o in usable])
        judge_rates = np.array([rates[o][judge] for o in usable])
        order = np.argsort(-pick_rates, kind='stable')
        best = usable[order[0]]
        gap = float(pick_rates[order[0]] - pick_rates[order[1]])
        regret = float(judge_rates[order[0]] - pi @ judge_rates)
        units.append({'gap': gap, 'regret': regret, 'top1': int(choice == best),
                      'best_is_pass': bool(position['pass'][best]) if best < len(position['pass']) else False})
    return units


def bin_of(gap):
    return next(i for i, (lo, hi) in enumerate(GAP_BINS) if lo <= gap < hi)


def summary(units, boot, seed, key=None):
    """Mean regret (points) and top-1 with bootstrap intervals over positions (a position's two folds together)."""
    if not units:
        return {'positions': 0}
    groups = collections.defaultdict(list)
    for u in units:
        groups[u['pid']].append(u)
    positions = list(groups.values())
    def stat(chosen):
        flat = [u for g in chosen for u in g]
        return np.mean([u['regret'] for u in flat]) * 100, np.mean([u['top1'] for u in flat])
    main = stat(positions)
    rng = np.random.default_rng(seed)
    samples = [stat([positions[i] for i in rng.integers(0, len(positions), len(positions))]) for _ in range(boot)]
    def ci(i):
        values = [s[i] for s in samples]
        return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]
    return {'positions': len(positions), 'folds': sum(len(g) for g in positions),
            'regret_points': round(float(main[0]), 3), 'regret_ci95': ci(0),
            'top1': round(float(main[1]), 3), 'top1_ci95': ci(1)}


def difference(a, b, boot, seed):
    """Mean regret of group a minus group b (points), bootstrap over positions within each group."""
    def by_position(units):
        groups = collections.defaultdict(list)
        for u in units:
            groups[u['pid']].append(u['regret'])
        return [np.mean(v) for v in groups.values()]
    pa, pb = np.array(by_position(a)), np.array(by_position(b))
    if not len(pa) or not len(pb):
        return None
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(pa, len(pa)).mean() - rng.choice(pb, len(pb)).mean() for _ in range(boot)]
    return {'value': round(float((pa.mean() - pb.mean()) * 100), 3),
            'ci95': [round(float(np.percentile(diffs, 2.5) * 100), 3), round(float(np.percentile(diffs, 97.5) * 100), 3)],
            'positions': [int(len(pa)), int(len(pb))]}


def noise_ceiling(positions, draws, seed):
    """Information: the top-1 a perfect chooser would score under this measurement (it always picks the option with the
    highest full-48 win rate; the halves are redrawn as binomial(24, that rate)), per gap bin of the redrawn half."""
    rng = np.random.default_rng(seed)
    hits = collections.defaultdict(list)
    for p in positions:
        usable = [o for o, results in enumerate(p['outcomes']) if len(results) >= 2]
        if len(usable) < 2:
            continue
        rates = np.array([np.mean([w for _, w in p['outcomes'][o]]) for o in usable])
        n = np.array([len(p['outcomes'][o]) // 2 for o in usable])
        truth_best = int(np.argmax(rates))
        for _ in range(draws):
            half = rng.binomial(n, rates) / n
            order = np.argsort(-half, kind='stable')
            hits[bin_of(float(half[order[0]] - half[order[1]]))].append(int(order[0] == truth_best))
    return {f'{lo * 100:g}-{hi * 100:g}': round(float(np.mean(hits[i])), 3) if hits[i] else None
            for i, (lo, hi) in enumerate(GAP_BINS)}


def cmd_learnt(args):
    from swccg_policy.v4_checkpoint import load
    _, _, tables, _ = load(args.checkpoint, 'cpu')
    positions = [p for spec in args.set for p in load_set(spec, tables)]
    units = []
    for p in positions:
        for u in folds(p):
            units.append({**u, 'pid': f"{p['set']}:{p['id']}", 'kind': p['kind'], 'phase': p['phase']})
    out = {'positions': len(positions), 'units': len(units), 'all': summary(units, args.boot, args.seed),
           'by_gap': {f'{lo * 100:g}-{hi * 100:g}': summary([u for u in units if bin_of(u['gap']) == i], args.boot, args.seed)
                      for i, (lo, hi) in enumerate(GAP_BINS)},
           'by_kind': {}, 'by_phase': {}}
    for kind in ('act-or-pass', 'deploy-where', 'card-selection', 'other'):
        out['by_kind'][kind] = summary([u for u in units if u['kind'] == kind], args.boot, args.seed)
    out['by_kind']['act-or-pass, best is Pass'] = summary(
        [u for u in units if u['kind'] == 'act-or-pass' and u['best_is_pass']], args.boot, args.seed)
    out['by_kind']['act-or-pass, best is an action'] = summary(
        [u for u in units if u['kind'] == 'act-or-pass' and not u['best_is_pass']], args.boot, args.seed)
    for phase in sorted({u['phase'] for u in units if u['phase']}):
        out['by_phase'][phase] = summary([u for u in units if u['phase'] == phase], args.boot, args.seed)
    top = out['by_gap']['15-900']
    out['H1'] = {'holds': bool(top.get('positions') and top['top1'] >= 0.6 and top['regret_points'] <= 5),
                 'rule': 'gap >= 15 points: top-1 >= 0.6 and mean regret <= 5 points', 'bin': top}
    waiting = [u for u in units if u['kind'] == 'act-or-pass' and u['best_is_pass']]
    rest = [u for u in units if not (u['kind'] == 'act-or-pass' and u['best_is_pass'])]
    d = difference(waiting, rest, args.boot, args.seed + 1)
    out['H2'] = {'holds': bool(d and d['ci95'][0] > 0), 'difference': d,
                 'rule': 'regret where Pass is best minus elsewhere, bootstrap interval above 0'}
    out['top1_noise_ceiling_by_gap'] = noise_ceiling(positions, 200, args.seed + 2)
    out['regret_note'] = ('regret = V(best on the other half) - V(pi), an unbiased estimate of a lower bound on the '
                          'regret against the true best option')
    out['examples'] = [{'set': p['set'], 'id': p['id'], 'kind': p['kind'], 'phase': p['phase'], 'text': p['text'][:90],
                        'pass_flags': p['pass']} for p in positions[:20]]
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('positions', 'all', 'H1', 'H2')}, indent=1))


# ---------------------------------------------------------------- stage 2

def truth_table(positions):
    """{position: (options, A* per option, half-1 and half-2 A*)} from E2-like outcomes (as E2's power readout)."""
    from advantage_readout import true_advantages
    out = {}
    for p in positions:
        usable = [o for o, results in enumerate(p['outcomes']) if results]
        if len(usable) < 2:
            continue
        pi = [p['pi'][o] for o in usable]
        _, a_star, _ = true_advantages(pi, [[w for _, w in p['outcomes'][o]] for o in usable])
        parts = []
        for parity in (0, 1):
            wins = [[w for k, w in p['outcomes'][o] if k % 2 == parity] for o in usable]
            parts.append(None if any(not w for w in wins) else true_advantages(pi, wins)[1])
        out[p['position']] = {'options': usable, 'a_star': dict(zip(usable, a_star)), 'pi': dict(zip(usable, pi)),
                              'h': None if parts[0] is None else (dict(zip(usable, parts[0])), dict(zip(usable, parts[1]))),
                              'kind': p['kind'], 'phase': p['phase'], 'pass': p['pass']}
    return out


def reliability(truth, positions):
    h1 = [truth[p]['h'][0][o] for p in positions if truth[p]['h'] for o in truth[p]['options']]
    h2 = [truth[p]['h'][1][o] for p in positions if truth[p]['h'] for o in truth[p]['options']]
    r = pearson(h1, h2)
    return None if r is None else spearman_brown(r)


def signal(estimates, truth, positions, rel):
    """Disattenuated Pearson of pi-centred estimates with A* over the positions' options."""
    xs, ys = [], []
    for p in positions:
        t = truth[p]
        options = [o for o in t['options'] if o in estimates.get(p, {})]
        if len(options) < 2:
            continue
        w = np.array([t['pi'][o] for o in options], float)
        w = w / w.sum()
        e = np.array([estimates[p][o] for o in options])
        xs.extend(e - w @ e)
        ys.extend(t['a_star'][o] for o in options)
    r = pearson(xs, ys)
    return None if r is None or not rel or rel <= 0 else r / math.sqrt(rel)


def icc(samples_by_option):
    """One-way intra-class correlation of single samples within options (values centred per position first)."""
    groups = [np.asarray(v, float) for v in samples_by_option if len(v) >= 2]
    k = min(len(g) for g in groups)
    groups = [g[:k] for g in groups]
    means = np.array([g.mean() for g in groups])
    grand = means.mean()
    msb = k * ((means - grand) ** 2).sum() / (len(groups) - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(groups) * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def cmd_playouts(args):
    from swccg_policy.v4_checkpoint import load
    _, _, tables, _ = load(args.checkpoint, 'cpu')
    truth = truth_table(load_set(args.truth, tables))
    samples = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for line in Path(args.rows).read_text().splitlines():
        row = json.loads(line)
        for h in HORIZONS:
            if row.get(h) is not None:
                samples[h][row['position']][row['option']].append(row[h])
    positions = sorted(p for p in truth if p in samples['tc-r1'])
    groups = {'all': positions,
              'act-or-pass': [p for p in positions if truth[p]['kind'] == 'act-or-pass'],
              'other kinds': [p for p in positions if truth[p]['kind'] != 'act-or-pass']}
    rng = np.random.default_rng(args.seed)
    out = {'positions': len(positions), 'groups': {g: len(v) for g, v in groups.items()}, 'boot': args.boot,
           'reliability_truth': {g: reliability(truth, v) for g, v in groups.items()}, 'horizons': {}}
    for h in HORIZONS:
        per_option = samples[h]
        result = {}
        for g, members in groups.items():
            rel = out['reliability_truth'][g]
            curve = {}
            for n in NS:
                values = []
                for _ in range(1 if n == 24 else args.draws):
                    est = {p: {o: float(np.mean(rng.choice(v, n, replace=False) if n < len(v) else v))
                               for o, v in per_option[p].items()} for p in members}
                    values.append(signal(est, truth, members, rel))
                curve[n] = round(float(np.mean([v for v in values if v is not None])), 4)
            full = {p: {o: float(np.mean(v)) for o, v in per_option[p].items()} for p in members}
            boot = []
            for _ in range(args.boot):
                chosen = list(rng.choice(members, len(members)))
                relabelled = {i: full[p] for i, p in enumerate(chosen)}
                t = {i: truth[p] for i, p in enumerate(chosen)}
                value = signal(relabelled, t, list(t), reliability(t, list(t)))
                if value is not None:
                    boot.append(value)
            centred = []  # samples centred per position, for the intra-option correlation
            for p in members:
                pooled = np.mean([x for v in per_option[p].values() for x in v])
                centred.extend([np.asarray(v) - pooled for v in per_option[p].values()])
            rho = icc(centred)
            r1 = curve[1]
            result[g] = {'curve': curve, 'r24_ci95': [round(float(np.percentile(boot, 2.5)), 4),
                                                       round(float(np.percentile(boot, 97.5)), 4)] if boot else None,
                         'rho': round(rho, 4), 'ceiling': round(r1 / math.sqrt(rho), 4) if rho > 0 and r1 else None,
                         '_boot': boot}
        out['horizons'][h] = result
    s1 = out['horizons']['tc-r1']['all']
    out['S1'] = {'holds': bool(s1['curve'][24] >= S1_MIN and s1['r24_ci95'] and s1['r24_ci95'][0] >= S1_LOWER),
                 'rule': f'tc-r1 at N = 24: disattenuated r >= {S1_MIN} with lower bound >= {S1_LOWER}',
                 'value': s1['curve'][24], 'ci95': s1['r24_ci95']}
    paired = {}
    for g in ('act-or-pass', 'other kinds'):
        a, b = out['horizons']['gae-1'][g]['_boot'], out['horizons']['tc-r1'][g]['_boot']
        k = min(len(a), len(b))
        diffs = np.array(a[:k]) - np.array(b[:k])
        paired[g] = {'value': round(out['horizons']['gae-1'][g]['curve'][24] - out['horizons']['tc-r1'][g]['curve'][24], 4),
                     'ci95': [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}
    out['S2'] = {'holds': bool(paired['act-or-pass']['ci95'][0] > 0 and not paired['other kinds']['ci95'][0] > 0),
                 'rule': 'act-or-pass: gae-1 minus tc-r1 at N = 24 above 0 (paired), other kinds not',
                 'paired_gae1_minus_tcr1': paired,
                 'caveat': 'stage A and E2 share the deck order at each position, which flatters longer horizons'}
    for h in out['horizons'].values():
        for g in h.values():
            g.pop('_boot')
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({'S1': out['S1'], 'S2': out['S2'],
                      'curves': {h: {g: (v['curve'], v['rho'], v['ceiling']) for g, v in r.items()}
                                 for h, r in out['horizons'].items()}}, indent=1))


# ---------------------------------------------------------------- stage 4

def snr(r):
    return r * r / (1 - r * r) if r is not None and abs(r) < 1 else None


def cmd_economics(args):
    playouts = json.loads(Path(args.playouts).read_text())
    costs = json.loads(Path(args.costs).read_text())
    options = costs.get('options_per_decision', 3.5)
    ppo = snr(PPO_R) * PPO_DECISIONS
    rows = []
    for h, per_group in playouts['horizons'].items():
        cost = costs['playout_game_equivalents'].get(h)
        if cost is None:
            continue
        for n, r in per_group['all']['curve'].items():
            n = int(n)
            per_decision = n * options * cost
            rows.append({'horizon': h, 'N': int(n), 'r': r, 'game_equivalents_per_decision': round(per_decision, 4),
                         'snr_per_game_equivalent': round(snr(r) / per_decision, 5) if snr(r) else None,
                         'game_equivalents_for_100k': round(per_decision * GO_DECISIONS)})
    hours = costs.get('game_equivalents_per_gpu_hour')
    usable = [row for row in rows if row['r'] is not None and row['r'] >= GO_R]
    best = min(usable, key=lambda row: row['game_equivalents_for_100k']) if usable else None
    go = bool(best and hours and best['game_equivalents_for_100k'] / hours <= 20)
    out = {'ppo_snr_per_game': round(ppo, 4), 'rows': rows, 'best_usable': best,
           'gpu_hours_for_100k': round(best['game_equivalents_for_100k'] / hours, 1) if best and hours else None,
           'go': go, 'rule': f'go if a setting with r >= {GO_R} probes {GO_DECISIONS} decisions within 20 GPU-hours',
           'costs': costs}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('ppo_snr_per_game', 'best_usable', 'gpu_hours_for_100k', 'go')}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    learnt = sub.add_parser('learnt')
    learnt.add_argument('--set', action='append', required=True)
    learnt.add_argument('--checkpoint', default='runs/critic-noise/D.pt')
    playouts = sub.add_parser('playouts')
    playouts.add_argument('--rows', required=True)
    playouts.add_argument('--truth', required=True)
    playouts.add_argument('--checkpoint', default='runs/critic-noise/D.pt')
    playouts.add_argument('--draws', type=int, default=100)
    economics = sub.add_parser('economics')
    economics.add_argument('--playouts', required=True)
    economics.add_argument('--costs', required=True)
    for command in (learnt, playouts, economics):
        command.add_argument('--out', required=True)
        command.add_argument('--boot', type=int, default=2000)
        command.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    {'learnt': cmd_learnt, 'playouts': cmd_playouts, 'economics': cmd_economics}[args.command](args)


if __name__ == '__main__':
    main()
