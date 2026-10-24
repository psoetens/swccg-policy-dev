"""Plan dojo-three-objectives: 128-game eval schedules for Dojo items whose decks exist only as staged decks.

The decks come from data/decks/objective-coverage/staged/ (plan objective-deck-coverage), which are in no pool, so
make_dojo_pool.py cannot build these schedules. As there: the item's decks cycled against the held-out decks of the
opposite side, 128 rows, target seat = the item's side. Writes data/schedules/dojo/<item>-eval.tsv and adds the items to
data/schedules/dojo/summary.json (existing entries are kept as they are; pool share is 0, as no pool holds the decks).

Usage: python3 scripts/make_dojo_staged_schedules.py 12_180 12_89 13_73
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGED = Path('data/decks/objective-coverage')  # project-relative, like the pools
REWARDS = ROOT / 'data/dojo/rewards.json'
GAMES = 128


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('items', nargs='+')
    args = parser.parse_args()
    items = json.loads(REWARDS.read_text())['items']
    manifest = list(csv.DictReader(open(ROOT / STAGED / 'manifest.csv')))
    held = [line.split('\t') for line in (ROOT / 'data/schedules/eval_decipher_heldout.tsv').read_text().splitlines()
            if line and not line.startswith('#')]
    heldout = {'DS': sorted({row[0] for row in held}), 'LS': sorted({row[1] for row in held})}
    schedules = ROOT / 'data/schedules/dojo'
    summary_path = schedules / 'summary.json'
    summary = json.loads(summary_path.read_text())
    for key in args.items:
        item = items[key]
        side = item['side']
        holding = [str(STAGED / 'staged' / f'{row["deck_id"]}.xml') for row in manifest if row['objective'] == item['title']]
        if not holding:
            raise SystemExit(f'Dojo item {key} ({item["title"]}) has no staged deck')
        opponents = heldout['LS' if side == 'DS' else 'DS']
        games = [f'{deck}\t{opponent}' if side == 'DS' else f'{opponent}\t{deck}'
                 for deck, opponent in itertools.islice(zip(itertools.cycle(holding), itertools.cycle(opponents)), GAMES)]
        (schedules / f'{key}-eval.tsv').write_text('\n'.join(games) + '\n')
        summary['items'][key] = {'title': item['title'], 'kind': item['kind'], 'card_side': side,
                                 'target_seat': side.lower(), 'decks': len(holding), 'weak': False,
                                 'pool_line_share_of_side': 0.0, 'staged': True,
                                 'schedule': f'data/schedules/dojo/{key}-eval.tsv'}
        print(f'{key} {item["title"]}: {len(holding)} staged decks, {len(games)} games')
    summary_path.write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
