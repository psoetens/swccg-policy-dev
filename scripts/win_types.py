"""Plan card-identity-combos, M1: how games are won.

For each evaluation folder (with game logs): every game's Force losses per side, by source, are rebuilt from the engine
message log and checked against the per-game stats (force_lost_drain / _battle / _other). Each won game of the
evaluated seat is then classified by what took the loser's life force (the largest of drain, battle, other), with:
- rounds to the win (player turns / 2) and whether it came within 10 rounds;
- the decisive turn: the first player turn from which the winner had lost less Force than the loser for the rest of
  the game;
- the objective: flipped at the end or not (deck-intent split, project rule).

    win_types.py validate FOLDER [--games 20]
    win_types.py classify --eval LABEL=FOLDER ... --out JSON
"""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

TURN = re.compile(r"^Start of (ds|ls)'s turn #(\d+)")
LOSS = re.compile(r'^(ds|ls) loses (?:a|(\d+)) Force')
DRAIN = re.compile(r'^(ds|ls) Force drains (\d+)')
BATTLE_DEBT = re.compile(r'^(ds|ls) has .*battle damage remaining')
END_BATTLE = re.compile(r'attrition remaining to satisfy$|^(ds|ls) has no more')


def rebuild(log_path):
    """Per side: Force lost by source, and the cumulative losses at the start of every player turn."""
    lost = {side: collections.Counter() for side in ('ds', 'ls')}
    context = {'ds': 'other', 'ls': 'other'}
    timeline = []  # (player turn index, ds lost so far, ls lost so far)
    turns = 0
    for line in Path(log_path).read_text(errors='replace').splitlines():
        if TURN.match(line):
            turns += 1
            timeline.append((turns, sum(lost['ds'].values()), sum(lost['ls'].values())))
            context = {'ds': 'other', 'ls': 'other'}
            continue
        if match := DRAIN.match(line):
            context['ls' if match.group(1) == 'ds' else 'ds'] = 'drain'
            continue
        if match := BATTLE_DEBT.match(line):
            context[match.group(1)] = 'battle'
            continue
        if match := LOSS.match(line):
            side = match.group(1)
            lost[side][context[side]] += int(match.group(2) or 1)
            continue
        if line.startswith(('ds initiates', 'ls initiates', 'ds plays', 'ls plays', 'ds deploys', 'ls deploys')):
            context = {'ds': 'other', 'ls': 'other'}
    timeline.append((turns + 1, sum(lost['ds'].values()), sum(lost['ls'].values())))
    return lost, timeline, turns


def games(folder):
    folder = Path(folder)
    for csv_path in sorted(folder.glob('*.csv')):
        opponent = csv_path.stem
        for row in csv.DictReader(csv_path.open()):
            log = folder / 'game-logs' / f'{row["game"]}.log'
            if log.exists():
                yield opponent, row, log


def validate(args):
    checked = mismatches = 0
    rows = []
    for _, row, log in list(games(args.folder))[:args.games]:
        stats = json.loads(row['stats'])
        lost, _, _ = rebuild(log)
        for side in ('ds', 'ls'):
            mine = {source: lost[side][source] for source in ('drain', 'battle', 'other')}
            theirs = {source: stats[side][f'force_lost_{source}'] for source in ('drain', 'battle', 'other')}
            checked += 1
            mismatches += mine != theirs
            rows.append({'game': row['game'], 'side': side, 'log': mine, 'stats': theirs})
    print(json.dumps({'sides_checked': checked, 'mismatching': mismatches, 'rows': rows[:12]}, indent=1))


def decisive_turn(timeline, winner):
    loser = 'ls' if winner == 'ds' else 'ds'
    index = {'ds': 1, 'ls': 2}
    turn = 0
    for position, point in enumerate(timeline):
        if all(later[index[loser]] - later[index[winner]] > 0 for later in timeline[position:]):
            turn = point[0]
            break
    return turn


def classify(args):
    out = {}
    for item in args.eval:
        label, folder = item.split('=', 1)
        seat_ai = {}
        result = collections.defaultdict(list)
        for opponent, row, log in games(folder):
            evaluated = 'ds' if row['ds_ai'].startswith('policy:ck:') and 'agent' in row['ds_ai'] else 'ls'
            if row['ls_ai'].startswith('policy:ck:') and 'agent' in row['ls_ai']:
                evaluated = 'ls'
            for side in ('ds', 'ls'):
                ai = row[f'{side}_ai']
                if row['outcome'] != 'finished' or row['winner'] != side:
                    continue
                who = 'evaluated' if side == evaluated else ('advanced' if ai == 'advanced' else 'reference')
                stats = json.loads(row['stats'])
                loser = 'ls' if side == 'ds' else 'ds'
                losses = {source: stats[loser][f'force_lost_{source}'] for source in ('drain', 'battle', 'other')}
                _, timeline, turns = rebuild(log)
                result[(opponent, who)].append({
                    'source': max(losses, key=losses.get), 'losses': losses, 'turns': int(row['turns']),
                    'decisive_turn': decisive_turn(timeline, side),
                    'flipped': bool(stats[side].get('objective_flipped_at_end'))})
        summary = {}
        for (opponent, who), wins in sorted(result.items()):
            n = len(wins)
            share = lambda key, value: round(sum(1 for w in wins if w[key] == value) / n, 3)
            summary[f'{opponent}:{who}'] = {
                'wins': n,
                'won_by': {source: share('source', source) for source in ('drain', 'battle', 'other')},
                'mean_loser_losses': {source: round(statistics.mean(w['losses'][source] for w in wins), 2)
                                      for source in ('drain', 'battle', 'other')},
                'rounds_median': statistics.median(w['turns'] for w in wins) / 2,
                'within_10_rounds': round(sum(1 for w in wins if w['turns'] <= 20) / n, 3),
                'decisive_round_median': statistics.median(w['decisive_turn'] for w in wins) / 2,
                'by_intent': {intent: {'wins': len(sub),
                                       'won_by': {s: round(sum(1 for w in sub if w['source'] == s) / len(sub), 3)
                                                  for s in ('drain', 'battle', 'other')} if sub else {},
                                       'rounds_median': statistics.median(w['turns'] for w in sub) / 2 if sub else None}
                              for intent, sub in (('flipped', [w for w in wins if w['flipped']]),
                                                  ('not_flipped', [w for w in wins if not w['flipped']]))}}
        out[label] = summary
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validate')
    v.add_argument('folder', type=Path)
    v.add_argument('--games', type=int, default=20)
    c = sub.add_parser('classify')
    c.add_argument('--eval', action='append', required=True)
    c.add_argument('--out', required=True)
    args = parser.parse_args()
    {'validate': validate, 'classify': classify}[args.command](args)


if __name__ == '__main__':
    main()
