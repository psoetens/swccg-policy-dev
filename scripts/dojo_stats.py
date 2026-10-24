"""Plan objective-dojo A4: per-item Dojo statistics of any run or eval folder (always recorded in the stats column).

    dojo_stats.py DIR [DIR ...] [--role agent|all] [--out JSON] [--table MD]

Per CSV group (an eval's opponent: random / advanced / initial; a stage's rollouts), per Dojo item: games where the
item was active for the selected seats, success rate, and per milestone the hit rate and median own turn of the first
gain. --role agent counts only the evaluated checkpoint's seat (policy:ck:...agent.ts, or the learner in rollouts).
Standard Gameplay games (no item for the seat) are counted separately.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from dojo_check import csvs, markdown, necessity
from dojo_lib import stats_rows


def agent_seat(row, side):
    ai = row[side + '_ai']
    return ai.endswith('agent.ts') or ai.startswith('policy:learner')


def summarize(folders, role):
    groups = defaultdict(list)
    for folder in folders:
        for path in csvs(folder):
            group = 'rollouts' if Path(path).parent.name == 'rollouts' else Path(path).stem
            groups[group].extend(stats_rows([path]))
    select = agent_seat if role == 'agent' else (lambda row, side: True)
    report = {}
    for group, rows in sorted(groups.items()):
        _, table, warnings = necessity(rows, select)
        standard = sum(1 for row in rows if row['stats'] for side in ('ds', 'ls')
                       if select(row, side) and not row['stats']['dojo'][side]['items'])
        report[group] = {'games': len(rows), 'standard_gameplay_seats': standard,
                         'items': {key: entry for key, entry in table.items() if entry['games']}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--role', choices=['agent', 'all'], default='agent')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--table', type=Path)
    args = parser.parse_args()
    report = summarize(args.folders, args.role)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + '\n')
    text = ''.join(f'### {group} ({value["games"]} games)\n\n' + markdown(value['items']) + '\n'
                   for group, value in report.items())
    if args.table:
        args.table.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
