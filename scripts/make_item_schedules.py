"""Plan dojo-v-cards §1.3: per-item check schedules. For each Dojo item card (objective, Epic Event, Jedi Test) held by
the decks of the given schedules, ROWS games with its decks on the agent's side (cycled) against opponent decks drawn
(seeded) from the same schedules' decks of the other side. DS items go to OUT-ds.tsv, LS items to OUT-ls.tsv (one
`ds_deck<TAB>ls_deck` row per game, for evaluate_v31.py --agent-side ds / ls); OUT.json records the rows per item.

    PYTHONPATH=src:scripts .venv/bin/python scripts/make_item_schedules.py data/schedules/v-cards/{gather-ds,gather-ls,heldout}.tsv \
        --card-db data/cards/card_blueprint_database-7eee2c8fc8a2221a0425617c5cc4451ca7b88d81.json \
        --items 225_32,201_39 --rows 128 --min-decks 4 --out data/schedules/v-cards/items
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {'OBJECTIVE', 'EPIC_EVENT', 'JEDI_TEST'}


def decks_of(schedules):
    decks = set()
    for schedule in schedules:
        for line in Path(schedule).read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            decks.update(part.strip() for part in line.split('\t') if part.strip().endswith('.xml'))
    return sorted(decks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('schedules', nargs='+')
    ap.add_argument('--card-db', type=Path, required=True)
    ap.add_argument('--items', help='comma-separated card ids (default: every item card in the decks)')
    ap.add_argument('--rows', type=int, default=128)
    ap.add_argument('--min-decks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    db = {r['cardId']: r for r in json.loads(a.card_db.read_text())}
    side_of, holders = {}, defaultdict(list)
    for deck in decks_of(a.schedules):
        ids = set(re.findall(r'blueprintId="([^"]+)"', (ROOT / deck).read_text()))
        side = None
        for i in ids:
            card = db.get(i, {})
            if card.get('cardCategory') in CATEGORIES:
                holders[i].append(deck)
                side = card['side']
        side_of[deck] = side or next((db[i]['side'] for i in ids if i in db and db[i].get('cardCategory') == 'LOCATION'), None)
    wanted = a.items.split(',') if a.items else sorted(holders)
    rng = random.Random(a.seed)
    rows = {'DARK': [], 'LIGHT': []}
    record = {}
    pool = {s: sorted(d for d, ds in side_of.items() if ds == s) for s in ('DARK', 'LIGHT')}
    for item in wanted:
        decks = sorted(holders.get(item, []))
        if len(decks) < a.min_decks:
            record[item] = {'title': db[item]['title'], 'decks': len(decks), 'rows': 0, 'skipped': f'< {a.min_decks} decks'}
            continue
        side = db[item]['side']
        other = pool['LIGHT' if side == 'DARK' else 'DARK']
        start = len(rows[side])
        for n in range(a.rows):
            deck, opponent = decks[n % len(decks)], rng.choice(other)
            rows[side].append((deck, opponent) if side == 'DARK' else (opponent, deck))
        record[item] = {'title': db[item]['title'], 'side': side, 'decks': len(decks), 'rows': a.rows,
                        'file': f'{a.out.name}-{"ds" if side == "DARK" else "ls"}.tsv', 'first_row': start}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for side, suffix in (('DARK', 'ds'), ('LIGHT', 'ls')):
        Path(f'{a.out}-{suffix}.tsv').write_text(''.join(f'{ds}\t{ls}\n' for ds, ls in rows[side]))
    Path(f'{a.out}.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f"DS rows {len(rows['DARK'])}, LS rows {len(rows['LIGHT'])}; "
          f"{sum(1 for r in record.values() if r['rows'])} items, {sum(1 for r in record.values() if not r['rows'])} skipped")


if __name__ == '__main__':
    main()
