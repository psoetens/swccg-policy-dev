"""Sanity checks for TrainGen GameStats CSVs (plan game-stats-objectives §1.2).

Usage: game_stats_check.py CSV [CSV ...] [--equivalence OFF.csv ON.csv]
Exit 0 only if every check passes; prints one summary line per file.
"""
import argparse
import csv
import json
import math
import sys

SIDES = ('ds', 'ls')


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def check_file(path):
    problems, games, finished, flips = [], 0, 0, 0
    with open(path) as stream:
        for row in csv.DictReader(stream):
            games += 1
            gid = row['game']
            if not row.get('stats'):
                problems.append(f'{gid}: no stats ({row["outcome"]})')
                continue
            s = json.loads(row['stats'])
            if row['outcome'] == 'finished':
                finished += 1
                winner, loser = row['winner'], 'ls' if row['winner'] == 'ds' else 'ds'
                if s[loser]['life_force'] != 0 or s[winner]['life_force'] <= 0:
                    problems.append(f'{gid}: life force winner {s[winner]["life_force"]} loser {s[loser]["life_force"]}')
            state = {}
            for e in s['flip_events']:
                key = (e['side'], e['card'])
                if not e['to_back'] and not state.get(key):
                    problems.append(f'{gid}: flip-back of {e["title"]} without an earlier flip')
                state[key] = e['to_back']
                flips += 1
            for side in SIDES:
                v = s[side]
                if min(v['force_lost_drain'], v['force_lost_battle'], v['force_lost_other'], v['drains']) < 0:
                    problems.append(f'{gid}: negative count on {side}')
                if v['flips'] - v['flip_backs'] not in (0, 1) and v['objective'] is not None:
                    problems.append(f'{gid}: {side} flips {v["flips"]} flip-backs {v["flip_backs"]}')
                objective_events = [e for e in s['flip_events'] if e['side'] == side and e['card'] == v['objective']]
                # An objective that left the table (lost / out of play) is reset to its front side by the engine.
                on_table = v['objective_in_play_at_end']
                if (v['objective'] is not None and objective_events and on_table
                        and objective_events[-1]['to_back'] != v['objective_flipped_at_end']):
                    problems.append(f'{gid}: {side} objective end state disagrees with its last flip event')
            if 'check' in s:
                c = s['check']
                for i, side in enumerate(SIDES):
                    lost = sum(s[side][k] for k in ('force_lost_drain', 'force_lost_battle', 'force_lost_other'))
                    if lost != c[f'{side}_loses_msg']:
                        problems.append(f'{gid}: {side} force-loss events {lost} != messages {c[f"{side}_loses_msg"]}')
                if len(s['flip_events']) != c['flips_msg']:
                    problems.append(f'{gid}: flip events {len(s["flip_events"])} != messages {c["flips_msg"]}')
    print(json.dumps({'file': path, 'games': games, 'finished': finished, 'flip_events': flips,
                      'problems': len(problems), 'first': problems[:5]}))
    return not problems


def summary(path):
    rows = list(csv.DictReader(open(path)))
    agent_wins = sum(1 for r in rows if r['outcome'] == 'finished'
                     and (r['ds_ai'] if r['winner'] == 'ds' else r['ls_ai']) == 'advanced')
    turns = [int(r['turns']) for r in rows if r['outcome'] == 'finished']
    return len(rows), agent_wins, sum(turns) / len(turns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='*')
    parser.add_argument('--equivalence', nargs=2, metavar=('OFF', 'ON'))
    args = parser.parse_args()
    ok = all([check_file(path) for path in args.csv])
    if args.equivalence:
        (n0, w0, t0), (n1, w1, t1) = (summary(p) for p in args.equivalence)
        dw, dt = abs(w1 / n1 - w0 / n0), abs(t1 - t0) / t0
        passed = dw <= 0.15 and dt <= 0.15
        print(json.dumps({'equivalence': passed, 'off_win': round(w0 / n0, 3), 'off_ci95': [round(x, 3) for x in wilson(w0, n0)],
                          'on_win': round(w1 / n1, 3), 'on_ci95': [round(x, 3) for x in wilson(w1, n1)],
                          'win_diff_points': round(100 * dw, 1), 'off_mean_turns': round(t0, 1), 'on_mean_turns': round(t1, 1),
                          'turns_diff_rel': round(dt, 3), 'tolerance': '15 points / 15% (user)'}))
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
