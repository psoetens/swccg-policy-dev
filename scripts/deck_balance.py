"""Deck and matchup balance of policy-vs-policy games (reports/deck-balance.md).

Reads rollout CSVs (training games; both seats a policy: learner or snapshot), keeps finished games, and reports:
- per deck: games and win rate (either seat), the spread against binomial noise, a histogram, the extremes;
- per matchup (DS deck, LS deck): games and DS win rate; the share of matchups with >= MIN games decided >= 90% one way,
  against the share expected by chance at p = 0.5;
- per objective: mean deck win rate;
- the matchup table (pairs with >= MIN games) for schedule weighting (plan links-round2, phase 2).

    deck_balance.py --rollouts DIR [DIR ...] --out reports/data/deck-balance [--min-pair 6]
"""
import argparse
import collections
import csv
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / 'data/cards/card_blueprint_database-5f2f57d1336527ecf9505db0052f2c54ca54a7ed.json'


def objective_of(deck, cards):
    try:
        for element in ElementTree.parse(ROOT / deck).getroot().iter('card'):
            card = cards.get(element.get('blueprintId'))
            if card and card.get('cardCategory') == 'OBJECTIVE':
                return card['title']
    except (OSError, ElementTree.ParseError):
        pass
    return '(none)'


def chance_lopsided(n):
    """P(<= 10% or >= 90% DS wins) for n games at p = 0.5."""
    return sum(math.comb(n, k) for k in range(n + 1) if k <= 0.1 * n or k >= 0.9 * n) / 2 ** n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rollouts', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-pair', type=int, default=6)
    args = parser.parse_args()
    cards = {c['cardId']: c for c in json.loads(CARDS.read_text())}
    games, wins, seat_games, seat_wins = (collections.Counter() for _ in range(4))
    pairs = collections.defaultdict(lambda: [0, 0])
    sides = collections.Counter()
    total = skipped = 0
    for folder in args.rollouts:
        for path in sorted(Path(folder).glob('*.csv')):
            for row in csv.DictReader(path.open()):
                if not (row['ds_ai'].startswith('policy:') and row['ls_ai'].startswith('policy:')):
                    continue
                if row['outcome'] != 'finished':
                    skipped += 1
                    continue
                total += 1
                sides[row['winner']] += 1
                for side, deck in (('ds', row['ds_deck']), ('ls', row['ls_deck'])):
                    games[deck] += 1
                    wins[deck] += row['winner'] == side
                    seat_games[side] += 1
                pair = pairs[(row['ds_deck'], row['ls_deck'])]
                pair[0] += 1
                pair[1] += row['winner'] == 'ds'
    decks = sorted(((wins[d] / games[d], games[d], d) for d in games if games[d] >= 30))
    rates = [r for r, _, _ in decks]
    noise = statistics.mean(math.sqrt(0.25 / g) for _, g, _ in decks)
    histogram = [sum(1 for r in rates if i / 10 <= r < (i + 1) / 10 or (i == 9 and r == 1.0)) for i in range(10)]
    eligible = [(k, p) for k, p in pairs.items() if p[0] >= args.min_pair]
    lopsided = [(k, p) for k, p in eligible if p[1] / p[0] <= 0.1 or p[1] / p[0] >= 0.9]
    expected = statistics.mean(chance_lopsided(p[0]) for _, p in eligible) if eligible else None
    by_objective = collections.defaultdict(list)
    for rate, n, deck in decks:
        by_objective[objective_of(deck, cards)].append(rate)
    objectives = sorted(((statistics.mean(v), len(v), k) for k, v in by_objective.items() if len(v) >= 3))
    summary = {
        'policy_vs_policy_finished_games': total, 'unfinished_skipped': skipped,
        'ds_win_share': round(sides['ds'] / total, 4),
        'decks': len(games), 'decks_with_30_games': len(decks),
        'deck_win_rate_sd': round(statistics.pstdev(rates), 4), 'binomial_noise_sd': round(noise, 4),
        'histogram_0_to_1_by_tenths': histogram,
        'share_below_0.2': round(sum(r < 0.2 for r in rates) / len(rates), 4),
        'share_above_0.8': round(sum(r > 0.8 for r in rates) / len(rates), 4),
        'share_0.35_to_0.65': round(sum(0.35 <= r <= 0.65 for r in rates) / len(rates), 4),
        'matchups_with_min_games': len(eligible), 'min_games_per_matchup': args.min_pair,
        'lopsided_share': round(len(lopsided) / len(eligible), 4) if eligible else None,
        'lopsided_share_expected_by_chance': round(expected, 4) if expected is not None else None,
        'games_in_lopsided_matchups_share': round(sum(p[0] for _, p in lopsided) / max(sum(p[0] for _, p in eligible), 1), 4),
        'weakest_decks': [{'win_rate': round(r, 3), 'games': n, 'deck': d} for r, n, d in decks[:15]],
        'strongest_decks': [{'win_rate': round(r, 3), 'games': n, 'deck': d} for r, n, d in decks[-15:][::-1]],
        'objectives_weakest': [{'objective': k, 'decks': n, 'mean_deck_win_rate': round(m, 3)} for m, n, k in objectives[:10]],
        'objectives_strongest': [{'objective': k, 'decks': n, 'mean_deck_win_rate': round(m, 3)} for m, n, k in objectives[-10:][::-1]],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    with (args.out / 'decks.tsv').open('w') as stream:
        stream.write('deck\tgames\twins\twin_rate\n')
        for rate, n, deck in decks:
            stream.write(f'{deck}\t{n}\t{wins[deck]}\t{rate:.4f}\n')
    with (args.out / 'matchups.tsv').open('w') as stream:
        stream.write('ds_deck\tls_deck\tgames\tds_wins\tds_win_rate\n')
        for (ds, ls), (n, w) in sorted(eligible):
            stream.write(f'{ds}\t{ls}\t{n}\t{w}\t{w / n:.4f}\n')
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=1))


if __name__ == '__main__':
    main()
