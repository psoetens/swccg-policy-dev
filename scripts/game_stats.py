"""Aggregate TrainGen GameStats CSVs into report tables (plan game-stats-objectives).

Writes reports/data/knowledge-v3.1/game-stats.json and prints Markdown tables.
Roles: "agent" is the seat TrainGen's --agent played (alternating seats), "opponent" the other one.
Force numbers are Force a side LOST (to the other side's drains / battle damage).
"""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'runs/knowledge-v3.1/evaluations'
RUNS = [  # label, agent, opponent, csv
    ('n1 vs AdvancedAI', 'n1', 'AdvancedAI', E / 'stats-n1/advanced.csv'),
    ('n3 vs AdvancedAI', 'n3', 'AdvancedAI', E / 'stats-n3/advanced.csv'),
    ('n1 vs random', 'n1', 'random', E / 'stats-n1/random.csv'),
    ('n3 vs random', 'n3', 'random', E / 'stats-n3/random.csv'),
    ('n1 vs n3', 'n1', 'n3', E / 'stats-n1/initial.csv'),
    ('AdvancedAI vs AdvancedAI', 'AdvancedAI', 'AdvancedAI', E / 'stats-advanced/advanced.csv'),
]
TITLES = dict(line.rstrip('\n').split('\t', 1) for line in open(ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv') if '\t' in line)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h)), round(100 * (c + h))]


def rate(k, n):
    return {'k': k, 'n': n, 'pct': round(100 * k / n, 1) if n else None, 'ci95': wilson(k, n)}


def spread(values):
    if not values:
        return None
    return {'n': len(values), 'min': min(values), 'max': max(values), 'mean': round(st.mean(values), 1),
            'median': st.median(values)}


def games(path):
    rows = list(csv.DictReader(open(path)))
    for index, row in enumerate(rows):
        g = int(row['game'].rsplit('-', 1)[1])
        agent = 'ds' if g % 2 == 0 else 'ls'  # TrainGen --agentSide alternate: even games agent = DS
        other = 'ls' if agent == 'ds' else 'ds'
        s = json.loads(row['stats'])
        yield row, agent, other, s


def side_stats(s, side):
    v = s[side]
    events = [e for e in s['flip_events'] if e['side'] == side and e['objective']]
    flipped = any(e['to_back'] for e in events)
    return {
        'has_objective': v['objective'] is not None,
        'objective': v['objective'],
        'flipped': flipped,
        'first_flip_turn': next((e['turn'] for e in events if e['to_back']), None),
        'flip_back': any(not e['to_back'] for e in events),
        'objective_lost': v['objective'] is not None and not v['objective_in_play_at_end'],
        'drain_lost': v['force_lost_drain'], 'battle_lost': v['force_lost_battle'], 'other_lost': v['force_lost_other'],
        'drains': v['drains'], 'forfeits': v['forfeits'], 'life_force': v['life_force'],
    }


def summarize(label, agent_name, opponent_name, path):
    out = {'label': label, 'agent_name': agent_name, 'opponent_name': opponent_name, 'csv': str(path.relative_to(ROOT))}
    roles = {'agent': defaultdict(list), 'opponent': defaultdict(list)}
    winners = {'agent': [], 'opponent': []}
    win_by_flip = defaultdict(lambda: [0, 0])
    per_objective = defaultdict(lambda: {'agent': [0, 0], 'opponent': [0, 0]})
    n = finished = agent_wins = 0
    for row, a, o, s in games(path):
        n += 1
        sides = {'agent': side_stats(s, a), 'opponent': side_stats(s, o)}
        for role, v in sides.items():
            for key, value in v.items():
                roles[role][key].append(value)
            if v['has_objective']:
                cell = per_objective[v['objective']][role]
                cell[0] += v['flipped']
                cell[1] += 1
        if row['outcome'] == 'finished':
            finished += 1
            won = 'agent' if row['winner'] == a else 'opponent'
            agent_wins += won == 'agent'
            winners[won].append(s[a if won == 'agent' else o]['life_force'])
            if sides['agent']['has_objective']:
                cell = win_by_flip['agent flipped' if sides['agent']['flipped'] else 'agent did not flip']
                cell[0] += won == 'agent'
                cell[1] += 1
    out['games'], out['finished'] = n, finished
    out['agent_win'] = rate(agent_wins, n)
    for role, v in roles.items():
        with_obj = [i for i, h in enumerate(v['has_objective']) if h]
        flipped = [i for i in with_obj if v['flipped'][i]]
        out[role] = {
            'objective_games': len(with_obj),
            'flip': rate(len(flipped), len(with_obj)),
            'first_flip_turn': spread([v['first_flip_turn'][i] for i in flipped]),
            'flip_back_of_flipped': rate(sum(v['flip_back'][i] for i in flipped), len(flipped)),
            'objective_lost_of_flipped': rate(sum(v['objective_lost'][i] for i in flipped), len(flipped)),
            'force_lost_to_drains': spread(v['drain_lost']),
            'force_lost_to_battle_damage': spread(v['battle_lost']),
            'force_lost_other': spread(v['other_lost']),
            'drains_made': spread(v['drains']),
            'battle_forfeits': spread(v['forfeits']),
        }
    out['winner_life_force'] = {'agent_won': spread(winners['agent']), 'opponent_won': spread(winners['opponent']),
                                'all': spread(winners['agent'] + winners['opponent'])}
    out['agent_win_by_flip'] = {k: rate(*v) for k, v in win_by_flip.items()}
    out['per_objective'] = {TITLES.get(k, k): v for k, v in sorted(per_objective.items(), key=lambda kv: TITLES.get(kv[0], kv[0]))}
    return out


def pct(r):
    if not r or r['n'] == 0:
        return '—'
    return f"{r['pct']:.0f}% [{r['ci95'][0]}, {r['ci95'][1]}] ({r['k']}/{r['n']})"


def mean(sp):
    return '—' if not sp else f"{sp['mean']:.1f}"


def main():
    results = [summarize(*run) for run in RUNS]
    out = ROOT / 'reports/data/knowledge-v3.1/game-stats.json'
    out.write_text(json.dumps(results, indent=1) + '\n')
    print('| Pairing | agent win | agent flips | opponent flips | agent flip-backs | opponent flip-backs |'
          ' agent obj. lost | opponent obj. lost |')
    print('|---|---|---|---|---|---|---|---|')
    for r in results:
        print(f"| {r['label']} | {pct(r['agent_win'])} | {pct(r['agent']['flip'])} | {pct(r['opponent']['flip'])} | "
              f"{pct(r['agent']['flip_back_of_flipped'])} | {pct(r['opponent']['flip_back_of_flipped'])} | "
              f"{pct(r['agent']['objective_lost_of_flipped'])} | {pct(r['opponent']['objective_lost_of_flipped'])} |")
    print()
    print('| Pairing | winner life force: agent won (n, min/max/mean) | opponent won (n, min/max/mean) |'
          ' Force lost to drains: agent / opponent | battle damage: agent / opponent | forfeits: agent / opponent |'
          ' other Force loss: agent / opponent |')
    print('|---|---|---|---|---|---|---|')
    for r in results:
        w = r['winner_life_force']
        f = lambda sp: '—' if not sp else f"{sp['n']}, {sp['min']}/{sp['max']}/{sp['mean']:.1f}"
        print(f"| {r['label']} | {f(w['agent_won'])} | {f(w['opponent_won'])} | "
              f"{mean(r['agent']['force_lost_to_drains'])} / {mean(r['opponent']['force_lost_to_drains'])} | "
              f"{mean(r['agent']['force_lost_to_battle_damage'])} / {mean(r['opponent']['force_lost_to_battle_damage'])} | "
              f"{mean(r['agent']['battle_forfeits'])} / {mean(r['opponent']['battle_forfeits'])} | "
              f"{mean(r['agent']['force_lost_other'])} / {mean(r['opponent']['force_lost_other'])} |")
    print()
    print('| Pairing | agent win when it flipped (finished games) | when it did not flip | first flip turn: agent / opponent (median) |')
    print('|---|---|---|---|')
    for r in results:
        b = r['agent_win_by_flip']
        t = lambda sp: '—' if not sp else f"{sp['median']}"
        print(f"| {r['label']} | {pct(b.get('agent flipped'))} | {pct(b.get('agent did not flip'))} | "
              f"{t(r['agent']['first_flip_turn'])} / {t(r['opponent']['first_flip_turn'])} |")
    print()
    # Per objective, pooled over the n1/n3 policy runs vs AdvancedAI and the AdvancedAI mirror.
    pooled = defaultdict(lambda: {'policy': [0, 0], 'AdvancedAI': [0, 0]})
    for r in results:
        for title, v in r['per_objective'].items():
            for role in ('agent', 'opponent'):
                name = r[role + '_name']
                who = 'AdvancedAI' if name == 'AdvancedAI' else 'policy' if name in ('n1', 'n3') and r['label'] != 'n1 vs n3' else None
                if who:
                    pooled[title][who][0] += v[role][0]
                    pooled[title][who][1] += v[role][1]
    print('| Objective | policy (n1+n3, vs AdvancedAI and random) flipped | AdvancedAI (all its games) flipped |')
    print('|---|---|---|')
    for title, v in sorted(pooled.items(), key=lambda kv: -(kv[1]['policy'][1] + kv[1]['AdvancedAI'][1])):
        print(f"| {title} | {pct(rate(*v['policy']))} | {pct(rate(*v['AdvancedAI']))} |")


if __name__ == '__main__':
    main()
