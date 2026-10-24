"""Plan night-pool-v3: per-item Dojo rates from evaluation folders (the stats column of each <opponent>.csv).

For each registry item and each player of it (the policy or the opponent AI): games in which the item was tracked, and
in how many of them each milestone and the success fired at least once. Objectives also report the share of games that
kept the objective in play at the end (a start-up failure places it out of play).

    PYTHONPATH=scripts .venv/bin/python scripts/dojo_item_rates.py FOLDER [FOLDER ...] [--decks] [--out FILE.json]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARDS = json.loads((ROOT / 'data/dojo/rewards.json').read_text())['items']
csv.field_size_limit(sys.maxsize)


def player(row, side):
    return 'policy' if row[f'{side}_ai'].startswith('policy') else row[f'{side}_ai']


def rates(folders):
    games = defaultdict(int)
    fired = defaultdict(lambda: defaultdict(int))
    decks = defaultdict(lambda: {'games': 0, 'objective_in_play': 0, 'objective': None})
    for folder in folders:
        for path in sorted(Path(folder).glob('*.csv')):
            for row in csv.DictReader(path.open()):
                if not row.get('stats'):
                    continue
                stats = json.loads(row['stats'])
                for side in ('ds', 'ls'):
                    who = player(row, side)
                    deck = decks[row[f'{side}_deck']]
                    deck['games'] += 1
                    deck['objective'] = stats[side].get('objective')
                    deck['objective_in_play'] += bool(stats[side].get('objective_in_play_at_end'))
                    seen = {(e['item'], e['m'] if e['ev'] != 'success' else 'success')
                            for e in stats['dojo']['events'] if e['side'] == side and e['ev'] in ('gain', 'success')}
                    for item in stats['dojo'][side]['items']:
                        games[(item, who)] += 1
                        for item_seen, milestone in seen:
                            if item_seen == item:
                                fired[(item, who)][milestone] += 1
    items = {}
    for (item, who), n in sorted(games.items()):
        names = [m['name'] for m in REWARDS.get(item, {}).get('milestones', [])] + ['success']
        items.setdefault(item, {'title': REWARDS.get(item, {}).get('title', item)})[who] = {
            'games': n, **{name: fired[(item, who)].get(name, 0) for name in names}}
    return {'items': items, 'decks': dict(decks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--items', nargs='*', help='only these registry keys')
    parser.add_argument('--decks', action='store_true', help='also print each deck\'s objective-in-play share')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = rates(args.folders)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + '\n')
    for item, entry in result['items'].items():
        if args.items and item not in args.items:
            continue
        for who, counts in entry.items():
            if who == 'title':
                continue
            cells = ', '.join(f'{name} {count}' for name, count in counts.items() if name != 'games')
            print(f'{item} {entry["title"]} [{who}] games {counts["games"]}: {cells}')
    if args.decks:
        for deck, entry in sorted(result['decks'].items()):
            print(f'{entry["objective_in_play"]}/{entry["games"]} objective {entry["objective"]} in play: {deck}')


if __name__ == '__main__':
    main()
