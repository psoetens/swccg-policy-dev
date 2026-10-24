"""Plan objective-dojo: the n2 training pool and one 128-game eval schedule per Dojo item.

Pool (runs/dojo/n2-pool.txt), per side:
- the Decipher RL pool's distinct decks, plus the Carbon Chamber Testing / Jabba's Prize decks (objective-learnability
  Amendment 5); minus the Endor Operations deck without Establish Secret Base (Amendment 6, it can never flip);
- every Dojo item of that side gets the same line budget, spread equally over its decks; the 12 weak objectives
  (plan §2 rows 1-12) get twice that budget;
- decks without any Dojo item (Standard Gameplay) together get 10% of the side's lines.

Eval schedules (data/schedules/dojo/<item>-eval.tsv): the item's decks cycled against the held-out decks of the
opposite side, 128 rows, target seat fixed (summary.json records the seat). For a counter item the target seat is the
opponent of the decks holding the card.
"""
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from dojo_coverage import CARD_DB, REWARDS, schedule_decks

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path('data/decks/v4/corpus')  # project-relative, like the pools (plan host-migration Amendment 3)
EXTRA = {'DS': [CORPUS / name for name in ('swdb-217-Jabba_s_prize.xml', 'web-cctjp-oakes-unh-elimination.xml',
                                           'web-cctjp-ford-dark-deal.xml', 'web-cctjp-wilcox-bastardized.xml')]}
DROPPED = {CORPUS / 'pcz-PREF2_2021_Top8_Event_Decklists-8_PREF2_Eops_Gogolen_2021.xml'}
WEAK = ['7_138', '7_296', '8_167', '111_6', '110_6', '109_12', '7_137', '7_298', '8_78', '109_4', '9_61', '9_151']
BUDGET = 24        # pool lines per Dojo item (weak objectives: 2x)
STANDARD_SHARE = 0.10
GAMES = 128


def deck_cards(deck):
    return set(re.findall(r'blueprintId="([^"]+)"', (ROOT / deck).read_text()))


def main():
    items = json.loads(REWARDS.read_text())['items']
    sides = {card['cardId']: card['side'] for card in json.loads(CARD_DB.read_text())}
    pool = [line.split('\t') for line in (ROOT / 'data/schedules/rl_pool_decipher.txt').read_text().splitlines()
            if '\t' in line and not line.startswith('#')]
    decks = {'DS': set(), 'LS': set()}
    for side, deck in pool:
        decks[side.strip()].add(Path(deck.strip()))
    for side, extra in EXTRA.items():
        decks[side] |= set(extra)
    for side in decks:
        decks[side] -= DROPPED
    lines = {'DS': defaultdict(float), 'LS': defaultdict(float)}
    holders = {}
    summary = {'items': {}, 'sides': {}}
    for side in ('DS', 'LS'):
        cards = {deck: deck_cards(deck) for deck in sorted(decks[side])}
        standard = [deck for deck, ids in cards.items() if not any(key.split('~')[0] in ids for key in items)]
        for key, item in items.items():
            blueprint = key.split('~')[0]
            holding = [deck for deck, ids in cards.items() if blueprint in ids]
            if not holding:
                continue
            holders[key] = (side, holding)
            if item.get('counter'):
                continue
            budget = BUDGET * (2 if key in WEAK else 1)
            # A deck holding several items takes the largest of their budgets (no stacking): every item still gets at
            # least its own share, and decks with many items (e.g. Jedi Tests) do not crowd out the rest.
            for deck in holding:
                lines[side][deck] = max(lines[side][deck], budget / len(holding))
        item_lines = sum(lines[side].values())
        per_standard = STANDARD_SHARE / (1 - STANDARD_SHARE) * item_lines / max(1, len(standard))
        for deck in standard:
            lines[side][deck] += per_standard
        summary['sides'][side] = {'decks': len(cards), 'standard_decks': len(standard)}
    out = ROOT / 'runs/dojo'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for side in ('DS', 'LS'):
        for deck, weight in sorted(lines[side].items()):
            rows.extend([f'{side}\t{deck}'] * max(1, round(weight)))
        total = sum(1 for row in rows if row.startswith(side + '\t'))
        standard = sum(max(1, round(lines[side][deck])) for deck in lines[side]
                       if not any(key.split('~')[0] in deck_cards(deck) for key in items))
        summary['sides'][side].update({'lines': total, 'standard_line_share': round(standard / total, 4)})
    (out / 'n2-pool.txt').write_text('\n'.join(rows) + '\n')
    held = [line.split('\t') for line in (ROOT / 'data/schedules/eval_decipher_heldout.tsv').read_text().splitlines()
            if line and not line.startswith('#')]
    heldout = {'DS': sorted({row[0] for row in held}), 'LS': sorted({row[1] for row in held})}
    schedules = ROOT / 'data/schedules/dojo'
    schedules.mkdir(parents=True, exist_ok=True)
    pool_lines = {'DS': defaultdict(int), 'LS': defaultdict(int)}
    for row in rows:
        side, deck = row.split('\t')
        pool_lines[side][deck] += 1
    for key, item in items.items():
        if key not in holders:
            raise SystemExit(f'Dojo item {key} ({item["title"]}) has no deck in the pool')
        side, holding = holders[key]
        target = ('LS' if side == 'DS' else 'DS') if item.get('counter') else side
        opponents = heldout['LS' if side == 'DS' else 'DS']
        games = [f'{deck}\t{opponent}' if side == 'DS' else f'{opponent}\t{deck}'
                 for deck, opponent in itertools.islice(zip(itertools.cycle(holding), itertools.cycle(opponents)), GAMES)]
        name = key.replace('~', '-')
        (schedules / f'{name}-eval.tsv').write_text('\n'.join(games) + '\n')
        share = sum(pool_lines[side][str(deck)] for deck in holding) / sum(pool_lines[side].values())
        summary['items'][key] = {'title': item['title'], 'kind': item['kind'], 'card_side': side, 'target_seat': target.lower(),
                                 'decks': len(holding), 'weak': key in WEAK, 'pool_line_share_of_side': round(share, 4),
                                 'schedule': str((schedules / f'{name}-eval.tsv').relative_to(ROOT))}
    (schedules / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary['sides'], indent=1))
    print(f'{len(summary["items"])} item schedules; pool lines {len(rows)}')


if __name__ == '__main__':
    main()
