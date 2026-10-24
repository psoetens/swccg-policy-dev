"""Plan decision-cap-loops: find the repeating decision cycle in each game that hit the decision cap.

Reads an evaluation folder run with --export (every decision with 2+ options: game, index, side, type, question) and
its <opponent>.csv (outcome, seats, decks). For each capped game:
- the tail (the last TAIL exported decisions) and its shortest repeating cycle of (side, type, question);
- where the repetition starts (the first decision of the uninterrupted run of that cycle) and its length;
- which seat drives it (the policy or the opponent AI) and the decks.
Cycles are grouped into families by their (side-agnostic) question texts.

    python3 scripts/cap_loop_analysis.py EVAL_FOLDER [--opponent advanced] [--out JSON]
    python3 scripts/cap_loop_analysis.py --self-test
"""
import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
TAIL = 600


def shortest_cycle(sequence, max_period=12):
    """The shortest period p such that the sequence repeats with period p (None if none up to max_period)."""
    for period in range(1, max_period + 1):
        if len(sequence) >= 2 * period and all(sequence[i] == sequence[i - period] for i in range(period, len(sequence))):
            return period
    return None


def run_start(sequence, period):
    """Index of the first element of the uninterrupted periodic run that ends the sequence."""
    start = len(sequence) - period
    while start - 1 >= 0 and sequence[start - 1] == sequence[start - 1 + period]:
        start -= 1
    return start


def load(folder, opponent):
    rows = {r['game']: r for r in csv.DictReader((folder / f'{opponent}.csv').open())}
    decisions = collections.defaultdict(list)
    for path in sorted((folder / f'{opponent}-export').glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                record = json.loads(line)
                game = record[1]
                if rows.get(game, {}).get('outcome') in ('cap', 'loop'):
                    decisions[game].append((record[2], record[3], record[4], record[5]))
    return rows, decisions


def analyse(rows, decisions):
    games = []
    for game, records in sorted(decisions.items()):
        records.sort()
        row = rows[game]
        tail = [(side, kind, question) for _, side, kind, question in records[-TAIL:]]
        period = shortest_cycle(tail)
        entry = {'game': game, 'decisions': int(row['decisions']), 'turns': int(row['turns']), 'exported': len(records),
                 'ds_deck': row['ds_deck'].split('/')[-1], 'ls_deck': row['ls_deck'].split('/')[-1],
                 'ds_ai': 'policy' if row['ds_ai'].startswith('policy') else row['ds_ai'],
                 'ls_ai': 'policy' if row['ls_ai'].startswith('policy') else row['ls_ai']}
        if period is None:
            entry['cycle'] = None
            entry['tail_top'] = collections.Counter(tail).most_common(4)
        else:
            sequence = [(side, kind, question) for _, side, kind, question in records]
            start = run_start(sequence, period)
            cycle = sequence[-period:]
            entry.update({'period': period, 'cycle': cycle, 'run_start_index': records[start][0],
                          'run_length': len(sequence) - start,
                          'drivers': sorted({entry['ds_ai'] if side == 'DS' else entry['ls_ai'] for side, _, _ in cycle})})
        games.append(entry)
    families = collections.defaultdict(list)
    for entry in games:
        key = ' | '.join(sorted({q for _, _, q in entry['cycle']})) if entry.get('cycle') else 'no short cycle'
        families[key].append(entry)
    return games, {key: {'games': len(v), 'drivers': collections.Counter(d for e in v for d in e.get('drivers', [])),
                         'turns': sorted(e['turns'] for e in v), 'examples': [e['game'] for e in v[:3]],
                         'decks': collections.Counter(e['ds_deck'] if any(s == 'DS' for s, _, _ in (e.get('cycle') or []))
                                                      else e['ls_deck'] for e in v).most_common(5)}
                   for key, v in sorted(families.items(), key=lambda kv: -len(kv[1]))}


def guard_repeats(records, max_period=4):
    """Plan decision-cap-fixes (b): replay TrainGen's LoopGuard over one game's exported decisions (index, side, type,
    text, turn key) and return the most repetitions of any cycle of length 1-4 within a turn (the guard fires at 200).
    The export holds only decisions with 2+ options, so this approximates what the guard sees."""
    best, turn, history, run = 1, None, [], [0] * (max_period + 1)
    for _, side, kind, question, turn_key in records:
        key = (side, kind, question)
        if turn_key != turn:
            turn, history, run = turn_key, [], [0] * (max_period + 1)
        for p in range(1, max_period + 1):
            run[p] = run[p] + 1 if len(history) >= p and history[-p] == key else 0
            best = max(best, run[p] // p + 1)
        history.append(key)
    return best


def replay(folder, opponent):
    games = collections.defaultdict(list)
    for path in sorted((folder / f'{opponent}-export').glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                games[r[1]].append((r[2], r[3], r[4], r[5], f'{r[7][0]}/{r[7][1]}'))
    return {game: guard_repeats(sorted(records)) for game, records in games.items()}


def self_test():
    sequence = [('DS', 'X', 'a')] * 5 + [('DS', 'C', 'set'), ('DS', 'C', 'clone')] * 50
    assert shortest_cycle(sequence[-40:]) == 2
    assert run_start(sequence, 2) == 5
    assert shortest_cycle([('DS', 'X', str(i)) for i in range(40)]) is None
    loop = [(i, 'DS', 'C', 'set' if i % 2 else 'clone', '3/2') for i in range(500)]
    assert guard_repeats(loop) == 250
    assert guard_repeats([(i, 'DS', 'C', 'set', f'{i}/0') for i in range(50)]) == 1  # a new turn each time
    print('self-test ok')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, nargs='?')
    parser.add_argument('--opponent', default='advanced')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--guard-replay', action='store_true', help='plan decision-cap-fixes (b): max cycle repeats per game')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.guard_replay:
        repeats = replay(args.folder, args.opponent)
        top = sorted(repeats.items(), key=lambda kv: -kv[1])
        summary = {'games': len(repeats), 'max_repeats': top[0][1] if top else None, 'top': top[:5],
                   'games_over_100': sum(v > 100 for v in repeats.values())}
        if args.out:
            args.out.write_text(json.dumps(summary, indent=1) + '\n')
        print(json.dumps(summary))
        return
    rows, decisions = load(args.folder, args.opponent)
    games, families = analyse(rows, decisions)
    if args.out:
        args.out.write_text(json.dumps({'games': games, 'families': families}, indent=1, default=list) + '\n')
    for key, family in families.items():
        print(f"{family['games']:3} games  drivers {dict(family['drivers'])}  turns {family['turns']}\n      {key}\n      decks {family['decks']}")


if __name__ == '__main__':
    main()
