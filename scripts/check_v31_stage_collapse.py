"""Amendment 12 collapse check for a running or finished V3.1 RL stage (reads learner.jsonl only)."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


LIMITS = {'entropy_min': 0.3, 'entropy_drop_10': 0.15, 'kl_p95_10': 0.05, 'max_abs_logit': 150.0,
          'random_win_min_20': 0.5, 'random_games_min_20': 10}


def win_rate(rows, opponent):
    games = sum(row['win_by_opp'].get(opponent, {}).get('games', 0) for row in rows)
    wins = sum(value['games'] * value['win_rate'] for row in rows
               for name, value in row['win_by_opp'].items() if name == opponent)
    return (wins / games if games else None), games


A13_LIMITS = {'degradation_points': 0.15, 'window': 100, 'min_updates': 400, 'min_bot_games': 80}


def degradation(rows):
    """Amendment 13: win rate vs AdvancedAI over the last 100 updates against the best earlier window."""
    window = A13_LIMITS['window']
    if len(rows) < A13_LIMITS['min_updates']:
        return None, None, False
    now, now_games = win_rate(rows[-window:], 'bot')
    best = None
    for end in range(window, len(rows) - window + 1, 10):
        rate, games = win_rate(rows[end - window:end], 'bot')
        if rate is not None and games >= A13_LIMITS['min_bot_games'] and (best is None or rate > best):
            best = rate
    degraded = (now is not None and best is not None and now_games >= A13_LIMITS['min_bot_games']
                and now < best - A13_LIMITS['degradation_points'])
    return now, best, degraded


def check(rows, rules='a12'):
    """Return (verdict, details). Verdict is 'ok', 'warn' or 'abort'."""
    last10 = rows[-10:]
    last20 = rows[-20:]
    previous20 = rows[-40:-20]
    entropy = [row['entropy'] for row in rows]
    kl = [row['approx_kl_old'] for row in last10]
    logit = max(row['max_abs_logit'] for row in last10)
    finite = all(math.isfinite(row[key]) for row in rows for key in ('entropy', 'vloss', 'approx_kl_old', 'pg'))
    random_now, random_games = win_rate(last20, 'random')
    bot_now, _ = win_rate(last20, 'bot')
    bot_before, _ = win_rate(previous20, 'bot')
    entropy_drop = max(entropy[-10:]) - entropy[-1] if len(entropy) >= 10 else 0.0
    wins_falling = bot_now is not None and bot_before is not None and bot_now < bot_before - 0.15
    kl_p95 = float(np.percentile(kl, 95))
    details = {'iteration': rows[-1]['iteration'], 'elapsed_h': round(rows[-1]['elapsed'] / 3600, 2),
               'entropy': round(entropy[-1], 4), 'entropy_drop_10': round(entropy_drop, 4),
               'kl_p95_10': round(kl_p95, 4), 'kl_median_10': round(float(np.median(kl)), 4),
               'max_abs_logit_10': logit, 'win_random_20': random_now, 'random_games_20': random_games,
               'win_bot_20': bot_now, 'win_bot_prev20': bot_before, 'finite': finite}
    reasons = []
    if not finite:
        reasons.append('non-finite values')
    if entropy[-1] < LIMITS['entropy_min']:
        reasons.append('entropy below floor')
    # 20-update bot windows hold ~20 games; under a13 the 100-update degradation rule replaces them.
    if rules == 'a13':
        wins_falling = False
    if entropy_drop > LIMITS['entropy_drop_10'] and wins_falling:
        reasons.append('sharp entropy drop with falling win rate')
    if rules == 'a12' and logit > LIMITS['max_abs_logit']:
        reasons.append('max logit above bf16 ceiling')
    if rules == 'a13':
        details['raw_max_abs_logit_10'] = max((row.get('raw_max_abs_logit', 0.0) for row in last10), default=None)
        bot_now_100, bot_best_100, degraded = degradation(rows)
        details['win_bot_100'] = bot_now_100
        details['win_bot_best_100'] = bot_best_100
        if degraded:
            reasons.append('win rate vs AdvancedAI degraded by more than 15 points from its best window')
    if random_games >= LIMITS['random_games_min_20'] and random_now < LIMITS['random_win_min_20']:
        reasons.append('win rate vs random below 50%')
    # A single-update KL spike is not collapse (A10c had p95 0.10 at 2.8 h and kept improving).
    # Abort only when KL stays high (median) or spikes together with falling win rates.
    kl_high = float(np.median(kl)) > LIMITS['kl_p95_10'] or (kl_p95 > LIMITS['kl_p95_10'] and wins_falling)
    # a13 aborts only on collapse or play degradation; KL is reported (A10 had a KL-median spell at 5.7 h while
    # still improving against AdvancedAI).
    if kl_high and rules == 'a12':
        reasons.append('sustained KL above limit')
    warnings = []
    if kl_p95 > LIMITS['kl_p95_10']:
        warnings.append('sustained KL' if kl_high else 'KL spike')
    if logit > 50 or details.get('raw_max_abs_logit_10', 0) > 50:
        warnings.append('max logit above 50')
    verdict = 'abort' if reasons else ('warn' if warnings else 'ok')
    return verdict, {**details, 'reasons': reasons, 'warnings': warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--history', action='store_true', help='Replay the check at every iteration')
    parser.add_argument('--rules', choices=['a12', 'a13'], default='a12',
                        help='a12: logit ceiling aborts; a13: logit is reported, play degradation aborts')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.evidence / 'learner.jsonl').read_text().splitlines() if line]
    if len(rows) < 20:
        print(json.dumps({'verdict': 'ok', 'note': 'fewer than 20 iterations', 'iterations': len(rows)}))
        return
    if args.history:
        for end in range(20, len(rows) + 1):
            verdict, details = check(rows[:end], args.rules)
            if verdict != 'ok':
                print(json.dumps({'verdict': verdict, **details}))
        return
    verdict, details = check(rows, args.rules)
    print(json.dumps({'verdict': verdict, **details}))


if __name__ == '__main__':
    main()
