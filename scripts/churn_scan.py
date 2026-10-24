"""Plan churn-and-drain-cancel: find back-and-forth / repeated actions in evaluation game logs (evaluate_v31.py
--game-log), the metric whose absence let the deploy-phase capacity-slot swap go unnoticed (reports/capacity-slot-rule.md).

Per turn (the logs mark turns, not phases) and acting player:
- repeats: one exact log line more than REPEATS times (default 4: normal play repeats some actions 3-4 times a turn);
- back_and_forth: one card moved between the same two places (from A to B and from B to A), or one pair of starships
  ship-docked, more than THRESHOLD times (default 2);
- self_cancelled_drains: a Force drain canceled by the player who initiated it.
Patterns replace card names by <card> and numbers by #; patterns of normal game mechanics (drawing cards one by one,
shuffling, paying Force loss, ...) are listed under "benign" and left out of "suspicious".

    python3 scripts/churn_scan.py LOGDIR [--csv RUN.csv ...] [--threshold 2] [--repeats 4] [--out churn.json]

With --csv (the run's per-game CSV), actors are split into "policy" (a seat whose *_ai starts with "policy") and "other".
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

TURN = re.compile(r"^Start of (\S+)'s turn #(\d+)")
ACTOR = re.compile(r"^(ls|ds) ")
CARD = re.compile(r"•+[^•,]+?(?= from | to | at | using | as | for | of | and | with | on | in | into |, |$)")
# Plan forfeit-reset-and-scan: named movement verbs, so non-unique cards (no •) count too; several copies of one
# non-unique card moving between the same two places add up (a possible over-count, listed with an example).
FROM_TO = re.compile(r"^(ls|ds) (moves|transfers|relocates|shuttles|embarks|disembarks) (.+?) from (.+?) to (.+?)(?= using| as|$)")
DOCK = re.compile(r"^(ls|ds) ship-docks (•.+?) and (•.+)$")
DRAIN = re.compile(r"^(ls|ds) initiates Force drain")
CANCEL_DRAIN = re.compile(r"^(ls|ds) cancels Force drain")
BENIGN = (
    'draws a card into hand from', 'draws a card into sabacc hand', 'shuffles ', 'Force loss remaining to satisfy',
    'verifies ', 'uses # Force', 'activated # Force', 'puts a card at random', 'loses a Force', 'loses # Force',
    'has # battle damage remaining', 'draws <card> as a # for', 'places a card from', 'retrieves ',
)


def pattern(line):
    return re.sub(r'\d+', '#', CARD.sub('<card>', line))


def benign(pat):
    return any(b in pat for b in BENIGN)


def policy_sides(csv_paths):
    """game log file name -> the side ('ls' / 'ds') the policy played; empty without CSVs."""
    sides = {}
    for path in csv_paths or ():
        with open(path) as stream:
            for row in csv.DictReader(stream):
                side = 'ls' if row.get('ls_ai', '').startswith('policy') else 'ds' if row.get('ds_ai', '').startswith('policy') else None
                if side:
                    sides[row['game'] + '.log'] = side
    return sides


def scan(logdir, sides=None, threshold=2, repeats_over=4):
    sides = sides or {}
    repeats = {}  # (actor, pattern) -> {turns, max, lines, example}
    back = {}
    drains = collections.Counter()
    self_cancel = collections.Counter()
    games = 0

    def who(game, side):
        if not sides:
            return side
        return 'policy' if sides.get(game) == side else 'other'

    def add(table, key, n, example):
        entry = table.setdefault(key, {'turns': 0, 'max': 0, 'lines': 0, 'example': example})
        entry['turns'] += 1
        entry['lines'] += n
        entry['max'] = max(entry['max'], n)

    for path in sorted(Path(logdir).glob('*.log')):
        games += 1
        turn, lines, drainer = None, [], None

        def flush():
            for line, n in collections.Counter(l for l in lines if ACTOR.match(l)).items():
                if n > repeats_over:
                    add(repeats, (who(path.name, line[:2]), pattern(line)), n, f'{path.name} {turn}: x{n} {line}')
            moves = collections.Counter()
            for line in lines:
                m = FROM_TO.match(line)
                if m:
                    moves[(m.group(1), m.group(2), m.group(3), frozenset((m.group(4), m.group(5))))] += 1
            for line in lines:
                m = DOCK.match(line)
                if m:
                    moves[(m.group(1), 'ship-docks', m.group(2), frozenset((m.group(2), m.group(3), 'docked')))] += 1
            for (side, verb, card, places), n in moves.items():
                if verb == 'ship-docks' and n > threshold:
                    add(back, (who(path.name, side), f'{side} ship-docks <card> and <card> again'), n,
                        f'{path.name} {turn}: x{n} {side} ship-docks {" and ".join(sorted(p for p in places if p != "docked"))}')
                    continue
                if n > threshold and len(places) == 2:
                    add(back, (who(path.name, side), f'{side} {verb} <card> between two places'), n,
                        f'{path.name} {turn}: x{n} {side} {verb} {card} {" <-> ".join(sorted(places))}')

        for raw in path.read_text(errors='replace').splitlines():
            line = raw.lstrip('~')
            m = TURN.match(line)
            if m:
                if turn is not None:
                    flush()
                turn, lines = line, []
                continue
            m = DRAIN.match(line)
            if m:
                drainer = m.group(1)
                drains[who(path.name, drainer)] += 1
            m = CANCEL_DRAIN.match(line)
            if m and drainer and m.group(1) == drainer:
                self_cancel[who(path.name, drainer)] += 1
            lines.append(line)
        if turn is not None:
            flush()

    def table(entries, keep):
        rows = [{'actor': actor, 'pattern': pat, **v} for (actor, pat), v in entries.items() if keep(pat)]
        return sorted(rows, key=lambda r: (-r['turns'], -r['max']))

    return {
        'games': games,
        'threshold': threshold,
        'repeats_over': repeats_over,
        'suspicious': table(repeats, lambda p: not benign(p)),
        'back_and_forth': table(back, lambda p: True),
        'benign': table(repeats, benign),
        'drains_initiated': dict(drains),
        'self_cancelled_drains': dict(self_cancel),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs')
    parser.add_argument('--csv', action='append', help="the run's per-game CSV (repeatable): splits policy / other")
    parser.add_argument('--threshold', type=int, default=2, help='back-and-forth: more than this many per turn')
    parser.add_argument('--repeats', type=int, default=4, help='exact repeats: more than this many per turn')
    parser.add_argument('--out')
    args = parser.parse_args()
    result = scan(args.logs, policy_sides(args.csv), args.threshold, args.repeats)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
