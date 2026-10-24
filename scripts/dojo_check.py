"""Plan objective-dojo end-goal A-3: Dojo ladder check.

    dojo_check.py --necessity DIR [DIR ...] [--accounting DIR [DIR ...]] [--out JSON] [--table MD]

--necessity DIRs: run or eval folders whose CSVs carry the stats column (every game records the Dojo events).
  * Necessity: in every game where an item succeeded (earned objective flip, Epic Event success, Jedi Test
    completed), each milestone marked necessary in data/dojo/rewards.json was gained before the success. 0 allowed.
  * Hit table: per item, games where it was active, per-milestone hit rate and median own turn of the first gain,
    success rate; milestones with 0 hits are listed as warnings (not failures). Plan dojo-three-objectives: a repeating
    event ("repeat": n) also reports its gains per game; a "when": "flipped" milestone is never necessary (it can only
    follow the flip), so necessity skips it.
--accounting DIRs: folders with learner trajectories (traj/ or trajectories/) and the CSVs of the same games.
  * Per game side: sum of the trajectory's dojo_rewards == the side's recorded Dojo total == the sum of its events
    == the value held at game end (held milestones at their paid values plus a held success). 0 mismatches allowed.
  * Standard Gameplay: the trajectory reward is +1 / -1 for a finished game with a winner, else 0.
Exit 0 only if necessity and accounting both pass.
"""
import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from dojo_lib import folder_metas, stats_rows

ROOT = Path(__file__).resolve().parents[1]
REWARDS = json.loads((ROOT / 'data/dojo/rewards.json').read_text())['items']
TOLERANCE = 1e-4


def csvs(folder):
    folder = Path(folder)
    return sorted(set(glob.glob(str(folder / '*.csv')) + glob.glob(str(folder / 'rollouts/*.csv'))))


def side_events(stats, side):
    return [event for event in stats['dojo']['events'] if event['side'] == side]


def held_value(events):
    """Value held at game end, replaying gain/loss/success/refund events in order. A repeating event milestone gains
    several times and is never lost, so gains add up."""
    held = {}
    for event in events:
        key = (event['item'], event['m'])
        if event['ev'] in ('gain', 'success'):
            held[key] = held.get(key, 0.0) + event['v']
        elif event['ev'] in ('loss', 'flip_back', 'lost'):
            held.pop(key, None)
    return sum(held.values())


def necessity(rows, select=lambda row, side: True):
    violations, table = [], {}
    games = defaultdict(int)
    hits = defaultdict(lambda: defaultdict(list))
    gains = defaultdict(lambda: defaultdict(int))
    successes = defaultdict(int)
    for row in rows:
        stats = row['stats']
        if not stats or 'dojo' not in stats:
            continue
        for side in ('ds', 'ls'):
            if not select(row, side):
                continue
            events = side_events(stats, side)
            for key in stats['dojo'][side]['items']:
                games[key] += 1
                first = {}
                for index, event in enumerate(events):
                    if event['item'] == key and event['ev'] == 'gain':
                        first.setdefault(event['m'], (index, event['own_turn']))
                        gains[key][event['m']] += 1
                for name, (_, turn) in first.items():
                    hits[key][name].append(turn)
                success = next((index for index, event in enumerate(events)
                                if event['item'] == key and event['ev'] == 'success'), None)
                if success is None:
                    continue
                successes[key] += 1
                for milestone in REWARDS[key]['milestones']:
                    if milestone.get('when') == 'flipped':
                        continue
                    if milestone['necessary'] and not (milestone['name'] in first and first[milestone['name']][0] < success):
                        violations.append({'game': row['game'], 'side': side, 'item': key, 'milestone': milestone['name']})
    warnings = []
    for key, item in REWARDS.items():
        entry = {'title': item['title'], 'kind': item['kind'], 'games': games[key], 'successes': successes[key],
                 'success_rate': successes[key] / games[key] if games[key] else None, 'milestones': {}}
        for milestone in item['milestones']:
            turns = hits[key][milestone['name']]
            entry['milestones'][milestone['name']] = {
                'necessary': milestone['necessary'], 'games_hit': len(turns),
                'hit_rate': len(turns) / games[key] if games[key] else None,
                'median_own_turn': statistics.median(turns) if turns else None,
                'when': milestone.get('when', 'front'), 'repeat': milestone.get('repeat', 1),
                'gains_per_game': gains[key][milestone['name']] / games[key] if games[key] else None}
            if games[key] and not turns:
                warnings.append(f'{key} ({item["title"]}): milestone {milestone["name"]} never reached in {games[key]} games')
        if not games[key]:
            warnings.append(f'{key} ({item["title"]}): not active in any checked game')
        table[key] = entry
    return violations, table, warnings


def accounting(folders):
    mismatches, sides = [], 0
    for folder in folders:
        folder = Path(folder)
        rows = {row['game']: row for row in stats_rows(csvs(folder))}
        trajectories = folder / 'traj' if (folder / 'traj').is_dir() else folder / 'trajectories'
        for meta in folder_metas(trajectories):
            row = rows.get(meta['g'])
            if row is None:
                mismatches.append({'game': meta['g'], 'side': meta['s'], 'problem': 'no stats row'})
                continue
            sides += 1
            side = meta['s']
            events = side_events(row['stats'], side)
            paid = sum(entry[1] for entry in meta.get('dojo_rewards', ()))
            recorded = row['stats']['dojo'][side]['total']
            summed = sum(event['v'] for event in events)
            held = held_value(events)
            standard = (1.0 if meta['w'] == side else -1.0) if meta['o'] == 'finished' and meta['w'] in ('ds', 'ls') else 0.0
            values = {'trajectory': paid, 'recorded_total': recorded, 'event_sum': summed, 'held_at_end': held}
            if max(values.values()) - min(values.values()) > TOLERANCE:
                mismatches.append({'game': meta['g'], 'side': side, 'problem': 'dojo totals differ', **values})
            if abs(meta['reward'] - standard) > 1e-9:
                mismatches.append({'game': meta['g'], 'side': side, 'problem': 'reward is not Standard Gameplay',
                                   'reward': meta['reward'], 'expected': standard})
            if any(not 0 <= entry[0] < meta['n_steps'] for entry in meta.get('dojo_rewards', ())):
                mismatches.append({'game': meta['g'], 'side': side, 'problem': 'dojo reward index outside the game'})
    return mismatches, sides


def markdown(table):
    lines = ['| Item | Kind | Games | Success | Milestones: hit rate (median own turn) |', '|---|---|---:|---:|---|']
    for key, entry in table.items():
        milestones = '; '.join(
            f'{name}{" (N)" if value["necessary"] else ""}{" (after flip)" if value.get("when") == "flipped" else ""}: '
            + (f'{value["hit_rate"]:.0%} ({value["median_own_turn"]})' if value['games_hit'] else '**0**')
            + (f', {value["gains_per_game"]:.2f} gains/game (max {value["repeat"]})'
               if value.get('repeat', 1) > 1 and value['games_hit'] else '')
            for name, value in entry['milestones'].items()) or '—'
        rate = f'{entry["success_rate"]:.0%}' if entry['success_rate'] is not None else '—'
        lines.append(f'| {entry["title"]} (`{key}`) | {entry["kind"]} | {entry["games"]} | {rate} | {milestones} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--necessity', nargs='*', default=[])
    parser.add_argument('--accounting', nargs='*', default=[])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--table', type=Path)
    args = parser.parse_args()
    rows = stats_rows([path for folder in args.necessity for path in csvs(folder)])
    violations, table, warnings = necessity(rows)
    mismatches, sides = accounting(args.accounting)
    report = {'necessity_games': len(rows), 'necessity_violations': violations, 'accounting_sides': sides,
              'accounting_mismatches': mismatches, 'warnings': warnings, 'items': table,
              'necessity_dirs': args.necessity, 'accounting_dirs': args.accounting}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + '\n')
    if args.table:
        args.table.write_text(markdown(table))
    print(json.dumps({'necessity_games': len(rows), 'violations': len(violations), 'accounting_sides': sides,
                      'mismatches': len(mismatches), 'warnings': len(warnings)}))
    for violation in violations[:20]:
        print('VIOLATION', violation)
    for mismatch in mismatches[:20]:
        print('MISMATCH', mismatch)
    ok = not violations and not mismatches and (not args.accounting or sides > 0)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
