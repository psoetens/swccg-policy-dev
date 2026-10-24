"""Plan critic-replay: where a seat's decisions go (learner seat, finished games).

    decision_breakdown.py TRAJECTORIES [--every 1] [--out JSON] [--top 6]

Per phase and side (own / opponent turn): decisions per seat and per own turn, the most frequent questions with the
option picked; overall: decisions per seat, the Activate phase per own turn, the "You have not activated Force" check,
and the Force the policy chose to activate per own turn (the answers to "Choose amount of Force to activate").
"""
import argparse
import collections
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from swccg_policy.trajectory import trajectory_files  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402

PHASES = ['START', 'ACTIVATE', 'CONTROL', 'DEPLOY', 'BATTLE', 'MOVE', 'DRAW', 'END_OF_TURN', 'BETWEEN']
READER = None


def per_file(path):
    out = []
    for game in READER.load(path):
        if game['o'] != 'finished':
            continue
        rows = []
        for item, sequence in zip(game['items'], game['seqs']):
            head = item['head']
            picks = [int(item['opt'][s, 0]) for s in sequence]
            values = [int(item['opt'][s, 4]) for s in sequence]
            rows.append((int(head[6]), int(head[7]), int(head[1]), len(item['opt']), picks, values, int(head[4])))
        out.append(rows)
    return out


def main():
    global READER
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trajectories', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/dojo/focus-s2/final.pt')
    parser.add_argument('--every', type=int, default=1, help='use every n-th trajectory file')
    parser.add_argument('--top', type=int, default=6)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    _, _, tables, _ = load(args.checkpoint, 'cpu')
    READER = TrajectoryReader(tables, [])
    names = {family: {v: k for k, v in tables.vocabulary.payload[family].items()} for family in ('q', 'o')}
    files = sorted(trajectory_files(args.trajectories))[::args.every]
    seats = own_turns = 0
    count = collections.Counter()
    by_phase = collections.defaultdict(collections.Counter)
    picks = collections.defaultdict(collections.Counter)
    activated = check = 0
    with multiprocessing.get_context('fork').Pool(8) as pool:
        for result in pool.imap(per_file, files):
            for rows in result:
                seats += 1
                own_turns += max(row[6] for row in rows)
                for phase, mine, question, _, chosen, values, _ in rows:
                    key = (PHASES[phase] if 0 <= phase < len(PHASES) else 'NONE', 'own' if mine else 'opp')
                    count[key] += 1
                    by_phase[key][question] += 1
                    if chosen:
                        picks[question][chosen[0]] += 1
                    text = str(names['q'].get(question, ''))
                    if text.startswith('Choose amount of Force to activate') and values:
                        activated += values[0]
                    check += text.startswith('You have not activated Force')
    turns = own_turns / max(seats, 1)
    total = sum(count.values())
    summary = {'files': len(files), 'seats': seats, 'own_turns_per_seat': turns, 'decisions_per_seat': total / seats,
               'activate_per_own_turn': count[('ACTIVATE', 'own')] / seats / turns,
               'not_activated_check_per_seat': check / seats, 'force_activated_per_own_turn': activated / seats / turns,
               'phases': {f'{p}:{s}': c / seats for (p, s), c in count.items()}}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != 'phases'}))
    for phase in PHASES:
        for side in ('own', 'opp'):
            key = (phase, side)
            if count[key] / seats < 1:
                continue
            print(f'\n=== {phase} ({side} turn): {count[key] / seats:.1f} per seat = {count[key] / seats / turns:.2f} per turn')
            for question, n in by_phase[key].most_common(args.top):
                top = picks[question].most_common(3)
                shown = sum(picks[question].values()) or 1
                desc = ', '.join(f'{str(names["o"].get(o, o))[:30]} {100 * k / shown:.0f}%' for o, k in top)
                print(f'  {n / seats / turns:5.2f}/turn  {str(names["q"].get(question, question))[:64]:64s} | {desc}')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
