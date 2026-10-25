"""Plan dojo-v-cards §1.3: per-item readout of record-only Dojo games (the CSVs of evaluate_v31.py, stats column).

    PYTHONPATH=src:scripts .venv/bin/python scripts/dojo_item_readout.py DIR [DIR ...] --items 225_32,201_39 [--out JSON] [--table MD]

Per item, over the games where it was active for the agent's side (the CSV's agent seat) and per opponent:
games, success rate (earned flip / Epic Event success / Jedi Test completion; 0-valued successes counted too), the share
of successes before own turn 3 (the plan's "free reward" check fails when > 90% of all games succeed before own
turn 3), per-milestone hit rate, and the total
Dojo value paid to the side (P items must pay 0). Items never active (e.g. excluded_by their objective in every deck)
are listed as such.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from dojo_lib import stats_rows

ROOT = Path(__file__).resolve().parents[1]
REWARDS = json.loads((ROOT / 'data/dojo/rewards.json').read_text())['items']
EARLY_TURN = 3
FREE_SHARE = 0.9


def agent_side(row):
    for side in ('ds', 'ls'):
        if row.get(f'{side}_ai', '').startswith('policy:ck:') and 'agent' in row.get(f'{side}_ai', ''):
            return side
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--table', type=Path)
    a = ap.parse_args()
    wanted = a.items.split(',')
    paths = sorted(p for d in a.dirs for p in glob.glob(str(Path(d) / '*.csv')))
    acc = defaultdict(lambda: {'games': 0, 'success': 0, 'early': 0, 'paid': 0.0, 'milestones': defaultdict(int)})
    for row in stats_rows(paths):
        stats = row['stats']
        side = agent_side(row)
        if not stats or not side or 'dojo' not in stats:
            continue
        opponent = row[('ls' if side == 'ds' else 'ds') + '_ai'].split(':')[0]
        opponent = 'self' if opponent == 'policy' else opponent
        events = [e for e in stats['dojo']['events'] if e['side'] == side]
        for item in stats['dojo'][side]['items']:
            if item not in wanted:
                continue
            a_ = acc[(item, opponent)]
            a_['games'] += 1
            mine = [e for e in events if e['item'] == item]
            success = [e for e in mine if e['m'] == 'success' and e['ev'] == 'success']
            if success:
                a_['success'] += 1
                a_['early'] += success[0]['own_turn'] < EARLY_TURN
            a_['paid'] += sum(e['v'] for e in mine)
            for name in {e['m'] for e in mine if e['ev'] == 'gain'}:
                a_['milestones'][name] += 1
    result, lines = {}, ['| Item | Title | Opponent | Games | Success | Early share | Paid / game | Milestone hit rates | Check |',
                         '|---|---|---|---:|---:|---:|---:|---|---|']
    for item in wanted:
        spec = REWARDS[item]
        rows = [(o, v) for (i, o), v in sorted(acc.items()) if i == item]
        if not rows:
            result[item] = {'active': False}
            lines.append(f"| `{item}` | {spec['title']} | — | 0 | — | — | — | — | never active |")
            continue
        for opponent, v in rows:
            rate = v['success'] / v['games']
            early = v['early'] / v['success'] if v['success'] else 0.0
            passive = spec['success'] == 0 and not spec['milestones']
            check = 'FAIL: pays' if passive and abs(v['paid']) > 1e-9 else \
                'FAIL: free' if spec['success'] > 0 and v['early'] / v['games'] > FREE_SHARE else 'ok'
            hits = ', '.join(f"{m['name']} {v['milestones'][m['name']] / v['games']:.0%}" for m in spec['milestones'])
            result.setdefault(item, {})[opponent] = {'games': v['games'], 'success_rate': rate, 'early_share': early,
                                                     'paid_per_game': v['paid'] / v['games'], 'check': check,
                                                     'milestones': {m: n / v['games'] for m, n in v['milestones'].items()}}
            lines.append(f"| `{item}` | {spec['title']} | {opponent} | {v['games']} | {rate:.0%} | {early:.0%} | "
                         f"{v['paid'] / v['games']:.3f} | {hits or '—'} | {check} |")
    if a.out:
        a.out.write_text(json.dumps(result, indent=1) + '\n')
    table = '\n'.join(lines)
    if a.table:
        a.table.write_text(table + '\n')
    print(table)
    return 1 if any('FAIL' in line for line in lines) else 0


if __name__ == '__main__':
    raise SystemExit(main())
