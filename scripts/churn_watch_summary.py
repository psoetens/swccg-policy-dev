"""Plan dock-churn-and-loop-watch-run: one strength check's churn.json (scripts/churn_scan.py) -> one line of
runs/dojo/<tag>.churn.jsonl with the policy's patterns; prints a FINDING line for each policy pattern with more than 10
repeats in a turn (strength_watch.sh appends the output to its log).

    python3 scripts/churn_watch_summary.py CHURN_JSON ITERATION OUT_JSONL
"""
import json
import sys


def main():
    churn, iteration, out = sys.argv[1:]
    c = json.load(open(churn))
    mine = [r for k in ('suspicious', 'back_and_forth') for r in c[k] if r['actor'] == 'policy']
    row = {'iteration': int(iteration), 'games': c['games'],
           'policy_patterns': [{k: r[k] for k in ('pattern', 'turns', 'max', 'example')} for r in mine],
           'max_repeat': max((r['max'] for r in mine), default=0),
           'self_cancelled_drains': c['self_cancelled_drains'].get('policy', 0),
           'drains': c['drains_initiated'].get('policy', 0)}
    with open(out, 'a') as f:
        f.write(json.dumps(row) + '\n')
    for r in mine:
        if r['max'] > 10:
            print(f"FINDING iteration {iteration}: {r['pattern']} x{r['max']} ({r['turns']} turns) e.g. {r['example'][:160]}")


if __name__ == '__main__':
    main()
