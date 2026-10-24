"""Histogram of the decisions stored as PPO steps (plan activation-and-cancel).

Usage: decision_histogram.py RUN_DIR [RUN_DIR ...] [--top 16] [--json OUT] [--checkpoint CK]
Decodes RUN_DIR/trajectories/*.traj with n1's vocabulary (or --checkpoint's) and reports, as shares of all stored steps:
the top questions, activation steps (Activate action choices that picked "Activate Force", amount questions,
opponent "allow" questions), activation rounds per Activate phase, and chosen "Done/cancel" answers.
"""
import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

import torch

from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_trajectory import load_binary

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'runs/knowledge-v3.1/v31-train8h-n1/final.pt'
ACT, AMOUNT, ALLOW = 'Choose Activate action or Pass', 'Choose amount of Force to activate', 'Choose amount of Force to allow opponent'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--top', type=int, default=16)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT,
                        help='the vocabulary the trajectories were written with (default n1; a v6 run needs a v6 checkpoint)')
    args = parser.parse_args()
    _, _, tables, _ = load(args.checkpoint, 'cpu')
    vocab = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['vocab']
    question = {i: t for t, i in vocab['q'].items()}
    option = {i: t for t, i in vocab['o'].items()}
    questions = collections.Counter()
    total = activate_force = amount = allow = cancel = games = 0
    phases, amounts = [], collections.Counter()
    for run in args.runs:
        for path in sorted(glob.glob(f'{run}/trajectories/*.traj')):
            for game in load_binary(path, tables):
                games += 1
                phase = None
                for item, seq in zip(game['items'], game['seqs']):
                    total += 1
                    q = question.get(int(item['head'][1]), '?')
                    o = option.get(int(item['opt'][seq[0]][0]), '?') if seq else '<pass>'
                    questions[q] += 1
                    if q.startswith(AMOUNT):
                        amount += 1
                        amounts[int(item['opt'][seq[0]][4])] += 1
                    elif q.startswith(ALLOW):
                        allow += 1
                    elif q.startswith(ACT):
                        phase = phase or [0, 0]
                        phase[0] += 1
                        if o == 'Activate Force':
                            activate_force += 1
                        else:
                            phases.append(phase)
                            phase = None
                    if phase is not None:
                        phase[1] += 1
                    if "click 'Done' to cancel" in q and o == '<none>':
                        cancel += 1
    share = lambda n: round(100 * n / total, 1) if total else None
    result = {
        'runs': args.runs, 'games': games, 'steps': total, 'steps_per_game': round(total / games, 1) if games else None,
        'activation_share': share(activate_force + amount + allow),
        'activate_force_share': share(activate_force), 'amount_share': share(amount), 'allow_share': share(allow),
        'cancel_share': share(cancel),
        'activate_actions_per_phase': round(statistics.mean(p[0] for p in phases), 2) if phases else None,
        'steps_per_activate_phase': round(statistics.mean(p[1] for p in phases), 2) if phases else None,
        'amounts': dict(sorted(amounts.items())[:20]),
        'top_questions': [[q, share(n)] for q, n in questions.most_common(args.top)],
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'top_questions'}))
    for q, s in result['top_questions']:
        print(f'{s:5.1f}% {q}')


if __name__ == '__main__':
    main()
