"""Summarize completed self-play stages without inventing learning-quality thresholds."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from game_length_metrics import opponent, summarize
from swccg_policy.data import DTYPES


def distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {'minimum': min(values), 'median': float(np.median(values)),
            'maximum': max(values), 'last': values[-1]}


def nonfinite(value, path=''):
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in nonfinite(child, path + '/' + str(key))]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in nonfinite(child, path + '/' + str(index))]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def action_summary(updates):
    counts = defaultdict(Counter)
    for row in updates:
        for kind, values in row['action_counts'].items():
            counts[DTYPES[int(kind)]].update(values)
    return {kind: {**dict(row), 'first_rate': row['first'] / row['decisions'],
                   'pass_rate': row['pass_choice'] / row['decisions']} for kind, row in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads((args.run / 'report.json').read_text())
    configuration = json.loads((args.run / 'configuration.json').read_text())
    updates = stage['updates']
    if not updates:
        raise ValueError('No completed learner updates')
    rows = []
    chunks = []
    counters = Counter()
    failure_logs = []
    for path in sorted((args.run / 'rollouts').glob('*.csv')):
        with path.open() as stream:
            chunk = [row for row in csv.DictReader(stream) if row.get('error') is not None]
        rows.extend(chunk)
        log = path.with_suffix('.log').read_text()
        complete = re.search(r'64 games on \d+ threads in', log) is not None
        chunks.append({'file': str(path), 'games_recorded': len(chunk), 'summary_present': complete})
        for label, pattern in {
            'fallbacks': r'fallbacks (\d+)', 'rejected': r'rejected (\d+)',
            'loop_guard': r'loop-guard (\d+)', 'real_loops': r'real-loops (\d+)',
            'native_failures': r'(\d+) failures',
        }.items():
            match = re.search(pattern, log)
            if match:
                counters[label] += int(match[1])
        if 'native batch failed' in log or 'Exception in thread' in log or 'Traceback (most recent' in log:
            failure_logs.append(str(path.with_suffix('.log')))
    exceptions = [row for row in rows if row['outcome'] not in ('finished', 'cap', 'timeout') or row['error']]
    by_opponent = defaultdict(list)
    for row in rows:
        by_opponent[opponent(row)].append(row)
    metrics = ('max_abs_logit', 'entropy', 'approx_kl_old', 'approx_kl_nonnegative', 'clipfrac',
               'vloss', 'value_explained_variance', 'actor_grad_norm', 'critic_grad_norm',
               'policy_lag_mean', 'policy_lag_max', 'queue_wait_seconds')
    samples = json.loads((args.run / 'resources.json').read_text())
    sustained = [row for row in samples if row['elapsed'] >= 180]
    checks = {
        'duration_complete': stage['result']['reason'] == 'duration_complete',
        'learner_exit_zero': stage['result']['learner_returncode'] == 0,
        'finite_updates': not nonfinite(updates),
        'policy_updates_present': any(row['policy_on'] for row in updates),
        'accepted_policy_lag_at_most_three': all(row['policy_lag_max'] <= 3 for row in updates),
        'no_recorded_game_exceptions': not exceptions,
        'no_recorded_native_failures': not failure_logs and not counters['native_failures'],
        'gpu_below_28_gib': all(row['gpu_used_mib'] < 28 * 1024 for row in samples),
        'ram_above_2_gib': all(row['memory_available'] >= 2 * 2 ** 30 for row in samples),
        'sustained_ram_at_least_4_gib': bool(sustained)
            and all(row['memory_available'] >= 4 * 2 ** 30 for row in sustained),
        'disk_at_least_20_gib': all(row['disk_free'] >= 20 * 2 ** 30 for row in samples),
    }
    report = {
        'run': str(args.run), 'configuration': configuration['args'], 'result': stage['result'],
        'checks': checks, 'operational_checks_passed': all(checks.values()),
        'learning_quality_decision': 'Inspect the reported trends; no automatic KL or accuracy threshold.',
        'updates': len(updates), 'accepted_decisions': sum(row['steps'] for row in updates),
        'accepted_player_trajectories': sum(row['games'] for row in updates),
        'stale_decisions_discarded': sum(row['discarded_stale_steps'] for row in updates),
        'metrics': {name: distribution([row.get(name) for row in updates]) for name in metrics},
        'first_five_updates': {name: distribution([row.get(name) for row in updates[:5]]) for name in metrics},
        'last_five_updates': {name: distribution([row.get(name) for row in updates[-5:]]) for name in metrics},
        'action_counts': action_summary(updates), 'last_five_action_counts': action_summary(updates[-5:]),
        'export_seconds': distribution([row['publication']['export_seconds'] for row in updates]),
        'cached_export_seconds': distribution([row['publication']['export_seconds'] for row in updates
                                               if not row['publication']['retraced']]),
        'resources': stage['resources'],
        'accepted_decisions_per_second': stage['accepted_decisions_per_second'],
        'rollout_counters_from_completed_summaries': dict(counters), 'chunks': chunks,
        'failure_logs': failure_logs, 'exceptions': exceptions, 'nonfinite_paths': nonfinite(updates),
        'games': summarize(rows),
        'games_by_opponent': {key: summarize(group) for key, group in by_opponent.items()},
        'censoring_note': 'Only flushed CSV rows are counted. The final in-flight chunk is stopped at the deadline; '
                          'its absent games and absent native summary are unverified.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                         for name in ('report.json', 'configuration.json', 'resources.json')},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.out), 'checks': checks, 'updates': len(updates),
                      'accepted_decisions': report['accepted_decisions']}))


if __name__ == '__main__':
    main()
