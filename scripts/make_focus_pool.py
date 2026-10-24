"""Plan dojo-focus: the focus training pool, the n2 pool's decks that hold one of the focus items.

    make_focus_pool.py [--out runs/dojo/focus-pool.txt]

Every focus item gets 48 lines (the weak-objective budget of make_dojo_pool.py), spread equally over its decks; a deck
holding several focus items takes the largest share (no stacking). No Standard Gameplay decks. The opponent seat draws
from runs/dojo/n2-pool.txt (TrainGen --opponentPool), so 14_52 We Have A Plan stays an opponent only (user 2026-10-06).

The pools of the focus runs (rev. 3 / 3b drop the SYCFA decks that hold no CPI / Superlaser, user 2026-10-06):
    make_focus_pool.py --n2-pool runs/dojo/n2-pool.orig.txt --out runs/dojo/focus-pool.txt          (s1)
    make_focus_pool.py --filter-only runs/dojo/focus-pool.txt --out runs/dojo/focus-pool-v2.txt     (s1b, s2)
    make_focus_pool.py --filter-only runs/dojo/n2-pool.orig.txt --out runs/dojo/opponent-pool-v2.txt
The running chain read the opponent pool from runs/dojo/n2-pool.txt, swapped to opponent-pool-v2 at 19:55 on
2026-10-06; runs/dojo/n2-pool.orig.txt is n2's pool, byte for byte.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from make_dojo_pool import deck_cards


def cpi_infeasible(side, cards):
    """Plan dojo-focus rev. 3: a DS deck holding Set Your Course For Alderaan (111_6) without Commence Primary Ignition
    (2_130) and Superlaser (2_161) can never flip it."""
    return side == 'DS' and '111_6' in cards and not ('2_130' in cards and '2_161' in cards)


def write_without_cpi_infeasible(source, out):
    """Copy a pool, dropping the lines of CPI-infeasible SYCFA decks (rev. 3b: also the opponent pool)."""
    kept, dropped = [], set()
    for line in Path(source).read_text().splitlines():
        side, deck = line.split('\t')
        if cpi_infeasible(side, deck_cards(deck)):
            dropped.add(deck)
            continue
        kept.append(line)
    Path(out).write_text('\n'.join(kept) + '\n')
    return {'out': str(out), 'lines': len(kept), 'dropped_decks': len(dropped)}


ROOT = Path(__file__).resolve().parents[1]
FOCUS = {'LS': ['9_61', '7_139', '8_43'], 'DS': ['10_29', '111_6', '9_151', '2_130']}
BUDGET = 48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n2-pool', type=Path, default=ROOT / 'runs/dojo/n2-pool.txt')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/dojo/focus-pool.txt')
    parser.add_argument('--filter-only', type=Path,
                        help='rev. 3/3b: only copy this pool to --out without the CPI-infeasible SYCFA decks')
    args = parser.parse_args()
    if args.filter_only:
        print(json.dumps(write_without_cpi_infeasible(args.filter_only, args.out), indent=1))
        return
    decks = {'DS': set(), 'LS': set()}
    for line in args.n2_pool.read_text().splitlines():
        side, deck = line.split('\t')
        decks[side].add(deck)
    rows, summary = [], {}
    for side, keys in FOCUS.items():
        cards = {deck: deck_cards(deck) for deck in sorted(decks[side])}
        lines = defaultdict(float)
        for key in keys:
            holding = [deck for deck, ids in cards.items() if key in ids]
            if not holding:
                raise SystemExit(f'focus item {key} has no deck in {args.n2_pool}')
            for deck in holding:
                lines[deck] = max(lines[deck], BUDGET / len(holding))
            summary[key] = {'side': side, 'decks': len(holding)}
        for deck, weight in sorted(lines.items()):
            rows.extend([f'{side}\t{deck}'] * max(1, round(weight)))
        for key in keys:
            summary[key]['lines'] = sum(1 for row in rows if row.startswith(side + '\t') and key in cards[row.split('\t')[1]])
    args.out.write_text('\n'.join(rows) + '\n')
    print(json.dumps({'out': str(args.out), 'lines': len(rows), 'items': summary}, indent=1))


if __name__ == '__main__':
    main()
