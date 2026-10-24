"""Plan probe-economics stage 6: life force and damage as a shaped reward.

    shaping_readout.py advantage --rows runs/q-critic/e2/readout.rows.jsonl --globals globals.jsonl \
        --truth runs/cn-replay/gt --out J
        6a. Per traced rollout: the forced option's shaped advantage = stage A's gae-lambda + beta * sum_l lambda^l
        (Phi_{l+1} - Phi_l) (+ the terminal -Phi_T in the potential form), for every potential x form x beta x
        lambda; the per-sample disattenuated signal against E2's truth with turn-credit stage A's statistics, paired
        against gae-0.95 and gae-0.8; R1, R2, R3; and, as information, Phi's change over 1 and 2 rounds averaged per
        option next to the critic-scored tc-r1 / tc-r2.
    shaping_readout.py value --d2 runs/critic-ceiling/d2 --d2b runs/critic-ceiling/d2-b --checkpoint D --out J
        6b. Each Phi at D2's 120 positions against the true value V* (truth b), and its partial correlation with V*
        given D's critic value (bootstrap over games).

Phi is in cards, from the deciding seat's view of the globals head[4:22] (see KnowledgeStateEncoder): index 4 / 5 own /
opponent life force, 12 / 13 Lost Piles, 14 / 15 hands (offsets into the 18 globals).
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import turn_credit_readout as tcr  # noqa: E402
from swccg_policy.credit import turn_anchors  # noqa: E402

LIFE, LIFE_OPP, LOST, LOST_OPP, HAND, HAND_OPP = 4, 5, 12, 13, 14, 15
POTENTIALS = {
    'damage': lambda g: g[LOST_OPP] - g[LOST],
    'life': lambda g: g[LIFE] - g[LIFE_OPP],
    'life-hand': lambda g: (g[LIFE] + g[HAND]) - (g[LIFE_OPP] + g[HAND_OPP]),
}
BETAS = (0.005, 0.01, 0.02, 0.05)
FORMS = ('potential', 'plain')
LAMBDAS = (0.95, 0.9)


def shaping_term(phi, lam, form):
    """sum_l lam^l (Phi_{l+1} - Phi_l) over a trace's steps; the potential form adds the terminal step -Phi_T
    (Phi = 0 at the game end), so the sum telescopes towards -Phi_0."""
    phi = np.asarray(phi, float)
    deltas = np.diff(phi)
    if form == 'potential':
        deltas = np.append(deltas, -phi[-1])
    weights = lam ** np.arange(len(deltas))
    return float(weights @ deltas) if len(deltas) else 0.0


def round_change(globals_, phi, rounds):
    """Phi at the rounds-th own-turn start after step 0 minus Phi at step 0 (the last step's Phi if the game ended)."""
    heads = [[0, 0, 0, 0, *g] for g in globals_]
    anchors = [a for a in turn_anchors(heads) if a > 0]
    index = anchors[rounds - 1] if len(anchors) >= rounds else len(phi) - 1
    return float(phi[index] - phi[0])


def variant_name(potential, form, beta, lam):
    return f'{potential}|{form}|{beta:g}|{lam:g}'


def cmd_advantage(args):
    spec, truth, _ = tcr.truth_table(args.truth, Path(args.truth, 'positions.json'))
    traces = {}
    for line in Path(args.globals).read_text().splitlines():
        row = json.loads(line)
        traces[row['job']] = row['g']
    names = ['gae-0.95', 'gae-0.9', 'gae-0.8', 'tc-r1', 'tc-r2']
    variants = [variant_name(p, f, b, lam) for p in POTENTIALS for f in FORMS for b in BETAS for lam in LAMBDAS]
    playout = [f'{p}|r{n}' for p in POTENTIALS for n in (1, 2)]
    samples, missing, steps_mismatch = [], 0, 0
    for line in Path(args.rows).read_text().splitlines():
        row = json.loads(line)
        g = traces.get(row['job'])
        if g is None:
            missing += 1
            continue
        if len(g) != row['steps']:
            steps_mismatch += 1
        values = {name: row[name] for name in names}
        for p, fn in POTENTIALS.items():
            phi = [fn(x) for x in g]
            for f in FORMS:
                for lam in LAMBDAS:
                    term = shaping_term(phi, lam, f)
                    for b in BETAS:
                        values[variant_name(p, f, b, lam)] = row[f'gae-{lam:g}'] + b * term
            for n in (1, 2):
                values[f'{p}|r{n}'] = round_change(g, phi, n)
        samples.append((row['position'], row['option'], values))
    all_names = names + variants + playout
    tcr.NAMES, tcr.REFERENCE = all_names, 'gae-0.95'
    data = tcr.build_pairs(truth, samples)
    count = len(data['positions'])
    draw = draw_positions(spec, args.truth)
    groups = {'all': np.ones(count, bool), 'draw': np.array([p in draw for p in data['positions']]),
              'other': np.array([p not in draw for p in data['positions']])}
    rng = np.random.default_rng(args.seed)
    out = {'rollouts': len(samples), 'rollouts_without_globals': missing, 'step_count_mismatches': steps_mismatch,
           'positions': count, 'groups': {k: int(v.sum()) for k, v in groups.items()}, 'boot': args.boot, 'groups_out': {}}
    keys = ('signal_dis', 'bias_centred_dis')
    for group, mask in groups.items():
        base = mask.astype(int)
        main = tcr.core(data, base)
        collected = collections.defaultdict(list)
        for _ in range(args.boot):
            weights = np.bincount(rng.choice(np.where(mask)[0], int(mask.sum())), minlength=count)
            result = tcr.core(data, weights)
            for name in all_names:
                for key in keys:
                    if result[name][key] is not None:
                        collected[(name, key)].append(result[name][key])
                for reference in ('gae-0.95', 'gae-0.8'):
                    a, b = result[name]['signal_dis'], result[reference]['signal_dis']
                    if a is not None and b is not None:
                        collected[(name, 'minus ' + reference)].append(a - b)
        def ci(key):
            values = collected[key]
            return None if len(values) < args.boot // 2 else [round(float(np.percentile(values, 2.5)), 4),
                                                               round(float(np.percentile(values, 97.5)), 4)]
        table = {}
        for name in all_names:
            m = main[name]
            table[name] = {'signal_dis': None if m['signal_dis'] is None else round(m['signal_dis'], 4),
                           'signal_dis_ci95': ci((name, 'signal_dis')),
                           'averaged_centred_dis': None if m['bias_centred_dis'] is None else round(m['bias_centred_dis'], 4),
                           'averaged_centred_dis_ci95': ci((name, 'bias_centred_dis')),
                           'noise_share': None if m['noise_share'] is None else round(m['noise_share'], 4),
                           'minus_gae095': ci((name, 'minus gae-0.95')), 'minus_gae08': ci((name, 'minus gae-0.8'))}
        out['groups_out'][group] = {'reliability': main['reliability'], 'table': table}
    table = out['groups_out']['all']['table']
    best = max(variants, key=lambda n: -9 if table[n]['signal_dis'] is None else table[n]['signal_dis'])
    best_095 = max((n for n in variants if n.endswith('|0.95')),
                   key=lambda n: -9 if table[n]['signal_dis'] is None else table[n]['signal_dis'])
    out['best'] = {'variant': best, **table[best]}
    out['R1'] = {'holds': bool(table[best]['minus_gae095'] and table[best]['minus_gae095'][0] > 0),
                 'rule': 'best variant minus gae-0.95, paired interval above 0 (best of 48 variants)'}
    out['R2'] = {'holds': bool(table[best_095]['minus_gae08'] and table[best_095]['minus_gae08'][0] > 0),
                 'variant': best_095, 'rule': 'best variant at lambda 0.95 minus gae-0.8, paired interval above 0'}
    out['R3'] = r3(data, groups, rng, args.boot, table)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('rollouts', 'rollouts_without_globals', 'step_count_mismatches', 'groups',
                                          'best', 'R1', 'R2', 'R3')}, indent=1))


def r3(data, groups, rng, boot, table):
    """Damage minus life at each potential's best beta (potential form, lambda 0.95), paired, overall and on draws."""
    def best_of(potential):
        names = [variant_name(potential, 'potential', b, 0.95) for b in BETAS]
        return max(names, key=lambda n: -9 if table[n]['signal_dis'] is None else table[n]['signal_dis'])
    damage, life = best_of('damage'), best_of('life')
    out = {'damage': damage, 'life': life}
    count = len(data['positions'])
    for group in ('all', 'draw'):
        mask = groups[group]
        diffs = []
        for _ in range(boot):
            weights = np.bincount(rng.choice(np.where(mask)[0], int(mask.sum())), minlength=count)
            result = tcr.core(data, weights)
            a, b = result[damage]['signal_dis'], result[life]['signal_dis']
            if a is not None and b is not None:
                diffs.append(a - b)
        main = tcr.core(data, mask.astype(int))
        value = None if main[damage]['signal_dis'] is None or main[life]['signal_dis'] is None \
            else round(main[damage]['signal_dis'] - main[life]['signal_dis'], 4)
        out[group] = {'damage_minus_life': value, 'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                                                            round(float(np.percentile(diffs, 97.5)), 4)] if diffs else None}
    out['damage_is_needed'] = bool(out['draw']['ci95'] and out['draw']['ci95'][0] > 0)
    return out


def draw_positions(spec, truth_dir):
    from advantage_readout import decision_texts
    out = set()
    for r in spec['list']:
        text = decision_texts(Path(truth_dir, 'record', r['record'].rsplit('/', 1)[1])).get(r['index'], '')
        if text.strip().startswith('Choose Draw action or Pass'):
            out.add(r['position'])
    return out


def partial(x, y, z):
    """Partial correlation of x and y given z."""
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return (rxy - rxz * ryz) / math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))


def cmd_value(args):
    from advantage_readout import load_states
    from critic_ceiling_readout import critic_values, position_outcomes
    from swccg_policy.v4_checkpoint import load
    spec = json.loads(Path(args.d2, 'positions.json').read_text())
    listed = spec['list']
    _, _, tables, _ = load(args.checkpoint, 'cpu')
    states = load_states(spec['states'], tables, {r['id'] for r in listed})
    present = [r for r in listed if r['id'] in states]
    v = critic_values(args.checkpoint, {'critic': [states[r['id']][1] for r in present]})['critic']
    outcomes = position_outcomes(args.d2b, listed)
    units = []
    for r, value in zip(present, v):
        results = outcomes.get(r['position'], [])
        if not results:
            continue
        g = [int(x) for x in states[r['id']][0]['head'][4:22]]
        units.append({'game': r['game'], 'v_star': 2 * np.mean([w for _, w in results]) - 1, 'v': float(value),
                      **{p: fn(g) for p, fn in POTENTIALS.items()}})
    by_game = collections.defaultdict(list)
    for u in units:
        by_game[u['game']].append(u)
    games = list(by_game.values())
    def stats(chosen):
        flat = [u for g in chosen for u in g]
        vs, v_ = np.array([u['v_star'] for u in flat]), np.array([u['v'] for u in flat])
        out = {'critic': float(np.corrcoef(v_, vs)[0, 1])}
        for p in POTENTIALS:
            x = np.array([u[p] for u in flat], float)
            out[p] = float(np.corrcoef(x, vs)[0, 1]) if x.std() > 0 else None
            out[p + '|partial'] = float(partial(x, vs, v_)) if x.std() > 0 else None
        return out
    main = stats(games)
    rng = np.random.default_rng(args.seed)
    samples = [stats([games[i] for i in rng.integers(0, len(games), len(games))]) for _ in range(args.boot)]
    out = {'positions': len(units), 'games': len(games), 'boot': args.boot,
           'note': 'raw Pearson (truth b reliability 0.99, so disattenuation changes < 1%)', 'readout': {}}
    for key, value in main.items():
        values = [s[key] for s in samples if s[key] is not None]
        out['readout'][key] = {'value': None if value is None else round(value, 4),
                               'ci95': [round(float(np.percentile(values, 2.5)), 4),
                                        round(float(np.percentile(values, 97.5)), 4)] if values else None}
    out['adds_to_critic'] = {p: bool(out['readout'][p + '|partial']['ci95'] and out['readout'][p + '|partial']['ci95'][0] > 0)
                             for p in POTENTIALS}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    adv = sub.add_parser('advantage')
    adv.add_argument('--rows', required=True)
    adv.add_argument('--globals', required=True)
    adv.add_argument('--truth', required=True)
    value = sub.add_parser('value')
    value.add_argument('--d2', required=True)
    value.add_argument('--d2b', required=True)
    value.add_argument('--checkpoint', default='runs/critic-noise/D.pt')
    for command in (adv, value):
        command.add_argument('--out', required=True)
        command.add_argument('--boot', type=int, default=1000)
        command.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    {'advantage': cmd_advantage, 'value': cmd_value}[args.command](args)


if __name__ == '__main__':
    main()
