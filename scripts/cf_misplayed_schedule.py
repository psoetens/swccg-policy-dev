"""Plan critic-followups, BL-48: an evaluation schedule for the decks W-bal reweighted.

Misplayed decks (runs/dojo/lr2-deck-weights.tsv, factor 1.0 = 2x): each deck in GAMES games (GAMES / 2 pairings with
random pool-v3b opponents of the other side), every pairing twice in a row. With evaluate_v31's alternating seats
(even game: agent Dark Side) the agent holds the listed deck in one game of each pair and the reference in the other.
Control: the same number of decks drawn from the pool-v3b decks that are not reweighted, same construction.
Writes PREFIX-{misplayed,control}-{0..PARTS-1}.tsv (an even number of rows each) and PREFIX-decks.json.

    cf_misplayed_schedule.py --out reports/data/critic-followups/bl48 [--games 24 --parts 2]
"""
import argparse
import json
import random
from pathlib import Path

POOL = 'data/schedules/rl_pool_decipher_v3b.txt'
WEIGHTS = 'runs/dojo/lr2-deck-weights.tsv'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--games', type=int, default=24)
    parser.add_argument('--parts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    if args.games % 2:
        raise ValueError('--games must be even')
    side, lines = {}, {'DS': [], 'LS': []}
    for line in Path(POOL).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            s, path = (x.strip() for x in line.split('\t'))
            side[path] = s.upper()
            lines[s.upper()].append(path)
    weighted = {}
    for line in Path(WEIGHTS).read_text().splitlines():
        if line.strip() and not line.startswith(('#', '*')):
            deck, factor = line.split('\t')
            weighted[deck] = float(factor)
    misplayed = sorted(d for d, f in weighted.items() if f == 1.0 and d in side)
    rng = random.Random(args.seed)
    others = sorted(d for d in side if d not in weighted)
    control = sorted(rng.sample(others, len(misplayed)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, decks in (('misplayed', misplayed), ('control', control)):
        rows = []
        for deck in decks:
            opposite = 'LS' if side[deck] == 'DS' else 'DS'
            for _ in range(args.games // 2):
                opponent = rng.choice(lines[opposite])
                row = f'{deck}\t{opponent}' if side[deck] == 'DS' else f'{opponent}\t{deck}'
                rows += [row, row]
        pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
        rng.shuffle(pairs)
        size = len(pairs) // args.parts
        for part in range(args.parts):
            chunk = pairs[part * size:(part + 1) * size] if part < args.parts - 1 else pairs[part * size:]
            Path(f'{out}-{name}-{part}.tsv').write_text(''.join(r + '\n' for pair in chunk for r in pair))
    Path(f'{out}-decks.json').write_text(json.dumps({'misplayed': misplayed, 'control': control,
                                                     'games_per_deck': args.games, 'seed': args.seed}, indent=1))
    print(len(misplayed), 'misplayed decks;', len(control), 'control decks')


if __name__ == '__main__':
    main()
