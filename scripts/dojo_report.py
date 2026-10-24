"""Plan objective-dojo Part B: build reports/objective-dojo.md from the n2 stage, the per-item evals and the yardsticks.

    dojo_report.py [--n2-train runs/dojo/n2-train8h] [--pilot runs/dojo/n2-pilot30]

Per Dojo item (data/schedules/dojo/summary.json), four evals of 128 games on decks holding it, target seat fixed:
{n2, n1} x {AdvancedAI, n1} (evaluations dojo-n2-<item> / dojo-n1-<item>, opponents advanced + initial = n1).
Success = earned objective flip, Epic Event success or Jedi Test completed, for the evaluated seat. Verdict (plan §1
B-2, the round-2 rule) from the vs-n1 column; the same rule on the vs-AdvancedAI column is shown next to it.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import statistics

from dojo_check import REWARDS, necessity
from dojo_lib import stats_rows
from dojo_stats import agent_seat
from swccg_policy.rl import wilson

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / 'runs/knowledge-v3.1/evaluations'
SUMMARY = json.loads((ROOT / 'data/schedules/dojo/summary.json').read_text())
WEAK = [key for key, item in SUMMARY['items'].items() if item['weak']]


def item_name(key):
    return key.replace('~', '-').replace('_', '-')


def seat_stats(rows, key):
    """Games where the item was active for the evaluated seat; successes, flip-backs/losses, wins, milestone hits.

    Shares: flip-back and lost (the flipped objective left play) per successful game; refund per milestone = games
    where a reached state milestone was lost again (engine event 'loss', which refunds what it paid) / games it was hit.
    """
    games = successes = undone = flip_backs = lost = wins = 0
    refunded = defaultdict(int)
    for row in rows:
        stats = row['stats']
        for side in ('ds', 'ls'):
            if not agent_seat(row, side) or key not in stats['dojo'][side]['items']:
                continue
            games += 1
            events = [event for event in stats['dojo']['events'] if event['side'] == side and event['item'] == key]
            successes += any(event['ev'] == 'success' for event in events)
            undone += any(event['ev'] in ('flip_back', 'lost') for event in events)
            flip_backs += any(event['ev'] == 'flip_back' for event in events)
            lost += any(event['ev'] == 'lost' for event in events)
            for name in {event['m'] for event in events if event['ev'] == 'loss'}:
                refunded[name] += 1
            wins += row['outcome'] == 'finished' and row['winner'] == side
    _, table, _ = necessity(rows, agent_seat)
    milestones = {name: {**value, 'games_refunded': refunded[name],
                         'refund_share': refunded[name] / value['games_hit'] if value['games_hit'] else None}
                  for name, value in table[key]['milestones'].items()}
    return {'games': games, 'successes': successes, 'rate': successes / games if games else None,
            'ci95': wilson(successes, games) if games else None, 'undone': undone,
            'flip_backs': flip_backs, 'lost': lost,
            'flip_back_share': flip_backs / successes if successes else None,
            'lost_share': lost / successes if successes else None,
            'win_rate': wins / games if games else None, 'milestones': milestones}


def standard_stats(rows):
    """Standard Gameplay seats (no Dojo item for the evaluated seat): games, wins, win rate with 95% Wilson."""
    games = wins = 0
    for row in rows:
        for side in ('ds', 'ls'):
            if row['stats'] and agent_seat(row, side) and not row['stats']['dojo'][side]['items']:
                games += 1
                wins += row['outcome'] == 'finished' and row['winner'] == side
    return {'games': games, 'wins': wins, 'rate': wins / games if games else None,
            'ci95': wilson(wins, games) if games else None}


def load(tag, opponent):
    path = EVALS / tag / f'{opponent}.csv'
    return stats_rows([path]) if path.exists() else []


def verdict(n2, n1):
    if not n2['games'] or not n1['games']:
        return 'n/a'
    if n1['rate'] >= 0.5:
        return 'ALREADY (ok)' if n2['ci95'][1] >= n1['rate'] - 0.10 else 'ALREADY (regressed)'
    if n2['ci95'][0] > n1['ci95'][1]:
        return 'LEARNED' if n2['rate'] >= 0.30 else 'PARTIAL'
    return 'NOT LEARNED'


def pct(value):
    return '—' if value is None else f'{100 * value:.0f}%'


def cell(stats):
    if not stats['games']:
        return '—'
    low, high = stats['ci95']
    return f'{pct(stats["rate"])} [{100 * low:.0f}, {100 * high:.0f}] ({stats["successes"]}/{stats["games"]})'


def training_curve(train):
    """Per hour of the stage: learner-seat success rate per item and hit rate per milestone (rollout CSVs, chunk
    start times). Returns {'success': {item: {hour: rate}}, 'milestones': {item: {milestone: {hour: rate}}}}."""
    started = json.loads((train / 'started.json').read_text())['time']
    hours = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for path in sorted((train / 'rollouts').glob('*.csv')):
        chunk = path.stem
        start = train / 'rollouts' / f'{chunk}-started.json'
        if not start.exists():
            continue
        hour = int((json.loads(start.read_text())['time'] - started) // 3600)
        for row in stats_rows([path]):
            if not row['stats']:
                continue
            for side in ('ds', 'ls'):
                if not row[side + '_ai'].startswith('policy:learner'):
                    continue
                for key in row['stats']['dojo'][side]['items']:
                    success = any(event['side'] == side and event['item'] == key and event['ev'] == 'success'
                                  for event in row['stats']['dojo']['events'])
                    hours[key][hour][0] += success
                    hours[key][hour][1] += 1
                    for name in {event['m'] for event in row['stats']['dojo']['events']
                                 if event['side'] == side and event['item'] == key and event['ev'] == 'gain'}:
                        hits[key][name][hour] += 1
    success = {key: {hour: values[0] / values[1] for hour, values in sorted(per.items()) if values[1]}
               for key, per in hours.items()}
    milestones = {key: {name: {hour: hits[key][name][hour] / hours[key][hour][1] for hour in sorted(hours[key])
                               if hours[key][hour][1]}
                        for name in (m['name'] for m in REWARDS[key]['milestones'])}
                  for key in hours if key in REWARDS}
    return {'success': success, 'milestones': milestones}


def frozen(tag):
    path = EVALS / tag / 'results.json'
    if not path.exists():
        return None
    return {label: {'win_rate': result['win_rate_all_games'], 'ci95': result['ci95_all_games'], 'games': result['games']}
            for label, result in json.loads(path.read_text())['results'].items()}


FOCUS_ITEMS = ['9_61', '14_52', '7_139', '8_43', '10_29', '111_6', '9_151', '2_130']
# Milestones added or redefined by plan dojo-focus (the readout's "changed milestone"); 9_61 has none left.
FOCUS_CHANGED = {'14_52': ['amidala_at_theed_palace_site', 'amidala_at_interior_naboo_site', 'amidala_at_throne_room'],
                 '7_139': ['spy_at_detention_block', 'leia_not_captive', 'leia_at_docking_bay_327', 'leia_at_yavin_4_docking_bay'],
                 '8_43': ['explosive_charge_on_bunker_1', 'explosive_charge_on_bunker_2'],
                 '10_29': ['xizor_at_battleground_site'], '9_151': ['vader_at_lukes_system'],
                 '2_130': ['superlaser_on_death_star', 'death_star_orbits_planet', 'gunners_on_death_star_2', 'cpi_good_attempt'],
                 '111_6': ['superlaser_on_death_star', 'death_star_orbits_alderaan', 'gunners_on_death_star_2',
                           'cpi_good_attempt_at_alderaan']}
# Plan dojo-focus rev. 3: n1 re-evaluated under the CPI set-up milestones for these items.
FOCUS_N1B = {'111_6', '2_130'}


def cpi_feasible(deck):
    """A DS deck that can fire Commence Primary Ignition (holds CPI and the Superlaser)."""
    ids = set(re.findall(r'blueprintId="([^"]+)"', (ROOT / deck).read_text()))
    return '2_130' in ids and '2_161' in ids
FOCUS_MODELS = (('n1', 'focus-n1'), ('n2', 'dojo-n2'), ('s1', 'focus-s1'), ('s2', 'focus-s2'))


def focus_report():
    """Plan dojo-focus §1 F-3: per focus item, n1 / n2 / s1 / s2 x {vs n1, vs AdvancedAI}; training curves of s1 and s2."""
    out, data = ROOT / 'reports/objective-dojo-focus.md', ROOT / 'reports/data/objective-dojo/focus-results.json'
    items = {}
    for key in FOCUS_ITEMS:
        name = item_name(key)
        prefixes = {model: ('focus-n1b' if model == 'n1' and key in FOCUS_N1B else prefix) for model, prefix in FOCUS_MODELS}
        rows = {(model, label): load(f'{prefixes[model]}-{name}', label)
                for model, _ in FOCUS_MODELS for label in ('initial', 'advanced')}
        variants = [(key, SUMMARY['items'][key]['title'], lambda row: True)]
        if key == '111_6':  # rev. 3: 68 of 128 eval rows hold no CPI / Superlaser and can never flip
            variants += [('111_6~feasible', SUMMARY['items'][key]['title'] + ' (CPI decks)', lambda row: cpi_feasible(row['ds_deck'])),
                         ('111_6~infeasible', SUMMARY['items'][key]['title'] + ' (no CPI)', lambda row: not cpi_feasible(row['ds_deck']))]
        for variant, title, keep in variants:
            items[variant] = {'title': title, **{
                f'{model}_vs_{opponent}': seat_stats([row for row in rows[(model, label)] if keep(row)], key)
                for model, _ in FOCUS_MODELS for opponent, label in (('n1', 'initial'), ('advanced', 'advanced'))}}

    def wilson_hit(stats, milestone):
        value = stats['milestones'].get(milestone)
        return None if not value or not stats['games'] else wilson(value['games_hit'], stats['games'])

    readout = {}
    for key, item in items.items():
        readout[key] = {}
        for model in ('s1', 's2'):
            for opponent in ('n1', 'advanced'):
                mine, base = item[f'{model}_vs_{opponent}'], item[f'n1_vs_{opponent}']
                if not mine['games'] or not base['games']:
                    continue
                improved = [m for m in FOCUS_CHANGED.get(key.split('~')[0], [])
                            if (wilson_hit(mine, m) or [0, 0])[0] > (wilson_hit(base, m) or [0, 0])[1]]
                success_ok = mine['ci95'][1] >= base['ci95'][0]
                readout[key][f'{model}_vs_{opponent}'] = {
                    'improved_milestones': improved, 'success_not_below_n1': success_ok,
                    'milestones_improved': bool(improved) and success_ok if key.split('~')[0] in FOCUS_CHANGED else None}
    curves = {tag: training_curve(ROOT / 'runs/dojo' / tag) for tag in ('focus-s1', 'focus-s1b', 'focus-s2')
              if (ROOT / 'runs/dojo' / tag / 'started.json').exists()}
    yardstick = {tag: frozen(f'{tag}-frozen') for tag in ('focus-s1', 'focus-s2')}
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({'items': items, 'readout': readout, 'training_curve': curves, 'frozen': yardstick},
                               indent=1) + '\n')
    lines = ['# Objective dojo, focus runs (plan dojo-focus)', '',
             'Plan: `.claude/plans/dojo-focus.md`. Data: `reports/data/objective-dojo/focus-results.json`. '
             'n1 is re-evaluated under the new milestone definitions; n2 rows reuse the Part B evals, so for changed '
             'milestones only success and win rate are comparable there. s1 = n1 + 42,772 focus games; s2 = s1 + 42,772 more. '
             '14_52 is not in the focus pool (opponent pool only); 7_139 has a single deck. '
             'Rev. 3: s1 was stopped part-way and resumed as focus-s1b with the CPI set-up milestones (2_130, 111_6) '
             'and focus pool v2 (without the 11 SYCFA decks that hold no CPI / Superlaser); the s1 evals use the '
             'focus-s1b checkpoint; n1 is re-evaluated for 111_6 and 2_130 (focus-n1b).', '',
             '## Success (evaluated seat, 128 games per cell, 95% Wilson)', '',
             '| Item | vs | n1 | n2 | s1 | s2 | win n1 / n2 / s1 / s2 |', '|---|---|---|---|---|---|---|']
    for key, item in items.items():
        for opponent, label in (('n1', 'n1'), ('advanced', 'AdvancedAI')):
            row = [item[f'{model}_vs_{opponent}'] for model, _ in FOCUS_MODELS]
            lines.append(f'| {item["title"]} (`{key}`) | {label} | ' + ' | '.join(cell(v) for v in row) + ' | '
                         + ' / '.join(pct(v['win_rate']) for v in row) + ' |')
    lines += ['', '## Milestones: hit rate (median own turn), refund share', '',
              '| Item | Milestone | vs | n1 | n2 | s1 | s2 |', '|---|---|---|---|---|---|---|']
    for key, item in items.items():
        for milestone in item['n1_vs_n1']['milestones']:
            for opponent, label in (('n1', 'n1'), ('advanced', 'AdvancedAI')):
                cells = []
                for model, _ in FOCUS_MODELS:
                    value = item[f'{model}_vs_{opponent}']['milestones'].get(milestone)
                    cells.append('—' if not value or not value['games_hit'] else
                                 f'{pct(value["hit_rate"])} ({value["median_own_turn"]}), refund {pct(value["refund_share"])}')
                lines.append(f'| {item["title"]} | {milestone} | {label} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Readout (pre-registered, not a gate)', '',
              'Milestones improved = a changed milestone\'s hit rate above n1 with non-overlapping Wilson intervals, '
              'and success not below n1 (upper bound ≥ n1\'s lower bound).', '',
              '| Item | s1 vs n1 | s1 vs AI | s2 vs n1 | s2 vs AI |', '|---|---|---|---|---|']
    for key, per in readout.items():
        def verdict_cell(entry):
            if entry is None:
                return '—'
            head = {True: '**improved**', False: 'not improved', None: 'no changed milestone'}[entry['milestones_improved']]
            return f'{head}; success {"ok" if entry["success_not_below_n1"] else "BELOW n1"}' + \
                (f' ({", ".join(entry["improved_milestones"])})' if entry['improved_milestones'] else '')
        lines.append(f'| {items[key]["title"]} | ' + ' | '.join(verdict_cell(per.get(k)) for k in
                     ('s1_vs_n1', 's1_vs_advanced', 's2_vs_n1', 's2_vs_advanced')) + ' |')
    for tag, curve in curves.items():
        hours = sorted({int(h) for per in curve['success'].values() for h in per}) or [0]
        header = ' | '.join(f'h{h}' for h in hours)
        lines += ['', f'## Training curve {tag}: learner-seat success and milestone hit rate per hour', '',
                  f'| Item | Milestone | {header} |', '|---|---|' + '---:|' * len(hours)]
        for key in FOCUS_ITEMS:
            if key not in curve['success']:
                continue
            lines.append(f'| {SUMMARY["items"][key]["title"]} | success | ' + ' | '.join(pct(curve['success'][key].get(h)) for h in hours) + ' |')
            for milestone, hourly in curve['milestones'].get(key, {}).items():
                lines.append(f'| | {milestone} | ' + ' | '.join(pct(hourly.get(h)) for h in hours) + ' |')
    lines += ['', '## General strength (frozen schedule, 128 games per opponent)', '']
    for tag, result in yardstick.items():
        lines.append(f'- {tag}: {json.dumps(result)}')
    out.write_text('\n'.join(lines) + '\n')
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n2-train', type=Path, default=ROOT / 'runs/dojo/n2-train8h')
    parser.add_argument('--pilot', type=Path, default=ROOT / 'runs/dojo/n2-pilot30')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/objective-dojo.md')
    parser.add_argument('--data', type=Path, default=ROOT / 'reports/data/objective-dojo/n2-results.json')
    parser.add_argument('--focus', action='store_true', help='Plan dojo-focus report (reports/objective-dojo-focus.md)')
    args = parser.parse_args()
    if args.focus:
        return focus_report()
    items = {}
    for key, info in SUMMARY['items'].items():
        name = item_name(key)
        settings = {f'{model}_vs_{opponent}': seat_stats(load(f'dojo-{model}-{name}', label), key)
                    for model in ('n2', 'n1') for opponent, label in (('advanced', 'advanced'), ('n1', 'initial'))}
        items[key] = {'title': info['title'], 'kind': info['kind'], 'weak': info['weak'], **settings,
                      'verdict_vs_n1': verdict(settings['n2_vs_n1'], settings['n1_vs_n1']),
                      'verdict_vs_advanced': verdict(settings['n2_vs_advanced'], settings['n1_vs_advanced'])}
    curve = training_curve(args.n2_train) if (args.n2_train / 'started.json').exists() else {'success': {}, 'milestones': {}}
    standard = {f'{model}_vs_{opponent}': standard_stats(load(f'dojo-{model}-frozen', label))
                for model in ('n2', 'n1') for opponent, label in (('advanced', 'advanced'), ('n1', 'initial'))}
    yardstick = {'frozen_n2': frozen('dojo-n2-frozen'), 'pilot_vs_n1': frozen('dojo-n2-pilot-h2h')}
    rr_path = ROOT / 'reports/data/objective-dojo/round-robin-n2.json'
    round_robin = json.loads(rr_path.read_text()) if rr_path.exists() else None
    weak_partial = [key for key in WEAK if items[key]['verdict_vs_n1'] in ('PARTIAL', 'LEARNED')]
    regressed = [key for key, item in items.items() if item['verdict_vs_n1'] == 'ALREADY (regressed)']
    h2h = (yardstick['frozen_n2'] or {}).get('initial', {}).get('win_rate')
    success = len(weak_partial) >= 6 and not regressed and h2h is not None and h2h >= 0.45
    report = {'items': items, 'standard_gameplay': standard, 'training_curve': curve, 'yardstick': yardstick, 'round_robin': round_robin,
              'readout': {'weak_partial_or_better': weak_partial, 'already_regressed': regressed,
                          'n2_vs_n1_frozen': h2h, 'success': success}}
    args.data.parent.mkdir(parents=True, exist_ok=True)
    args.data.write_text(json.dumps(report, indent=1) + '\n')
    lines = ['# Objective dojo, Part B: n2', '',
             'Plan: `.claude/plans/objective-dojo.md`. Harness: `reports/objective-dojo-harness.md`. '
             f'Data: `{args.data.relative_to(ROOT)}`.', '',
             '## Success per Dojo item (evaluated seat, decks holding the item, 128 games per cell, 95% Wilson)', '',
             '| Item | Kind | n2 vs n1 | n1 vs n1 | n2 vs AdvancedAI | n1 vs AdvancedAI | Win n2/n1 (vs n1) | Verdict (vs n1) | Verdict (vs AI) |',
             '|---|---|---|---|---|---|---|---|---|']
    for key, item in items.items():
        lines.append(f'| {item["title"]}{" (weak)" if item["weak"] else ""} (`{key}`) | {item["kind"]} | {cell(item["n2_vs_n1"])} | '
                     f'{cell(item["n1_vs_n1"])} | {cell(item["n2_vs_advanced"])} | {cell(item["n1_vs_advanced"])} | '
                     f'{pct(item["n2_vs_n1"]["win_rate"])} / {pct(item["n1_vs_n1"]["win_rate"])} | '
                     f'**{item["verdict_vs_n1"]}** | {item["verdict_vs_advanced"]} |')
    lines += ['', '## Standard Gameplay decks (frozen schedule, evaluated seat holds no Dojo item): win rate, 95% Wilson', '',
              '| Row | n2 vs n1 | n1 vs n1 | n2 vs AdvancedAI | n1 vs AdvancedAI |', '|---|---|---|---|---|']
    win_cell = lambda v: '—' if not v['games'] else f'{pct(v["rate"])} [{100 * v["ci95"][0]:.0f}, {100 * v["ci95"][1]:.0f}] ({v["wins"]}/{v["games"]})'  # noqa: E731
    lines.append('| Standard Gameplay | ' + ' | '.join(win_cell(standard[k]) for k in ('n2_vs_n1', 'n1_vs_n1', 'n2_vs_advanced', 'n1_vs_advanced')) + ' |')
    lines += ['', '## Success undone, vs n1: flip-backs and lost objectives per successful game', '',
              '| Item | n2 successes | n2 flip-back | n2 lost | n1 successes | n1 flip-back | n1 lost |', '|---|---:|---:|---:|---:|---:|---:|']
    share = lambda v, k: '—' if not v['successes'] else f'{pct(v[k + "_share"])} ({v[k + "s" if k == "flip_back" else k]})'  # noqa: E731
    for key, item in items.items():
        n2, n1 = item['n2_vs_n1'], item['n1_vs_n1']
        if n2['successes'] or n1['successes']:
            lines.append(f'| {item["title"]} | {n2["successes"]} | {share(n2, "flip_back")} | {share(n2, "lost")} | '
                         f'{n1["successes"]} | {share(n1, "flip_back")} | {share(n1, "lost")} |')
    lines += ['', '## Milestones, n2 vs n1 (vs n1): hit rate (median own turn), refund share', '',
              'Refund share: games where the reached state milestone was lost again (its reward refunded) / games it was hit.', '',
              '| Item | Milestone | n2 | n2 refund | n1 | n1 refund |', '|---|---|---|---:|---|---:|']
    for key, item in items.items():
        for name, value in item['n2_vs_n1']['milestones'].items():
            other = item['n1_vs_n1']['milestones'][name]
            fmt = lambda v: '—' if not v['games_hit'] else f'{pct(v["hit_rate"])} ({v["median_own_turn"]})'  # noqa: E731
            lines.append(f'| {item["title"]} | {name} | {fmt(value)} | {pct(value["refund_share"])} | {fmt(other)} | {pct(other["refund_share"])} |')
    by_title = lambda kv: SUMMARY['items'].get(kv[0], {}).get('title', kv[0])  # noqa: E731
    header = ' | '.join(f'h{h}' for h in range(9))
    lines += ['', '## Training curve: learner-seat success rate per hour', '', f'| Item | {header} |', '|---|' + '---:|' * 9]
    for key, per in sorted(curve['success'].items(), key=by_title):
        lines.append(f'| {SUMMARY["items"].get(key, {}).get("title", key)} | ' + ' | '.join(pct(per.get(h)) for h in range(9)) + ' |')
    lines += ['', '## Training curve: learner-seat milestone hit rate per hour', '', f'| Item | Milestone | {header} |',
              '|---|---|' + '---:|' * 9]
    for key, per in sorted(curve['milestones'].items(), key=by_title):
        for name, hourly in per.items():
            lines.append(f'| {SUMMARY["items"].get(key, {}).get("title", key)} | {name} | ' + ' | '.join(pct(hourly.get(h)) for h in range(9)) + ' |')
    lines += ['', '## Yardsticks', '', f'- Frozen eval (`dojo-n2-frozen`): {json.dumps(yardstick["frozen_n2"])}',
              f'- Pilot head-to-head vs n1: {json.dumps(yardstick["pilot_vs_n1"])}',
              f'- Round-robin row (new rules): {json.dumps((round_robin or {}).get("pairings"))}', '',
              '## Pre-registered readout', '',
              f'- Weak objectives at PARTIAL or better vs n1: {len(weak_partial)} of {len(WEAK)} ({", ".join(weak_partial) or "none"}); target ≥ 6.',
              f'- ALREADY items that regressed: {", ".join(regressed) or "none"}; target none.',
              f'- n2 vs n1 on the frozen schedule: {pct(h2h)}; target ≥ 45%.',
              f'- **Readout: {"SUCCESS" if success else "NOT A SUCCESS"}** (pre-registered rule, plan §1).', '']
    args.out.write_text('\n'.join(lines))
    print(args.out)


if __name__ == '__main__':
    main()
