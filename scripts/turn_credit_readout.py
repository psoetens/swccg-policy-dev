"""Plan turn-credit stage A: which credit scheme carries the most true signal per sample, given D's critic (offline).

Input: the stage-A rollouts (scripts/tc_stage_a.sh: 24 rollouts per (position, option) of E2's positions, every option
forced at decision K and played to the end by D; ReplayControl replay.trace wrote the branching seat's decisions from K
to the game end as one pseudo-game rt|g per job) and E2's ground truth (runs/cn-replay/gt: 48 rollouts per option).

Per usable rollout: D's critic values V for all steps (fp32, as E2's readout), the game result as the only reward
(credit.step_rewards with the pseudo-game's reward, +1 / -1 / 0 from the branching seat's view; no Dojo rewards: E2's
truth is the win rate), the turn anchors (credit.turn_anchors of the steps' heads), and the advantage AT STEP 0 (decision
K, the forced option) under every scheme:
  - per-decision GAE, lambda in {0, 0.8, 0.9, 0.95, 0.985, 1} (credit.gae);
  - turn credit, n-round returns n in {1, 2, 3} and round-lambda in {0.5, 0.7} (credit.turn_credit).
Truth: A*(o) = winrate(o) - sum_o' pi(o') winrate(o') from E2's rollouts (pi from positions.json, renormalised over the
options with rollouts); its reliability = split half (even / odd rollouts), Spearman-Brown to the full count.
Readouts per scheme, with bootstrap 95% intervals over positions:
  1. per-sample signal: Pearson(scheme advantage of one rollout, A*(o)) over all (position, option, rollout),
     disattenuated by the truth's reliability; also pi-centred (minus the position's pi-weighted mean of the option
     means) and the paired difference against GAE(0.95);
  2. bias: Pearson(mean of the scheme over the option's rollouts, A*(o)), disattenuated likewise (and pi-centred);
  3. noise share: within-option variance / total variance of the scheme's advantage.
Choice (fixed by the plan): the two turn-credit variants with the highest per-sample signal, as learner flags.

    turn_credit_readout.py --stage-a runs/turn-credit/a --truth runs/cn-replay/gt --checkpoint runs/critic-noise/D.pt \
        --out runs/turn-credit/a/readout.json
"""
import argparse
import collections
import concurrent.futures
import json
import math
import time
from pathlib import Path

import numpy as np

from advantage_readout import rollout_outcomes, spearman_brown, true_advantages
from swccg_policy.credit import gae, step_rewards, turn_anchors, turn_credit

GAE_LAMBDAS = (0.0, 0.8, 0.9, 0.95, 0.985, 1.0)
TURN_ROUNDS = (1, 2, 3)
ROUND_LAMBDAS = (0.5, 0.7)
REFERENCE = 'gae-0.95'
# (name, kind, parameter, learner flags); the turn variants are the stage-B candidates.
SCHEMES = ([(f'gae-{lam:g}', 'gae', lam, None) for lam in GAE_LAMBDAS]
           + [(f'tc-r{n}', 'rounds', n, f'--credit turn --rounds {n}') for n in TURN_ROUNDS]
           + [('tc-rl' + f'{x:.1f}'.replace('.', ''), 'round_lambda', x, f'--credit turn --round-lambda {x:g}')
              for x in ROUND_LAMBDAS])
NAMES = [name for name, *_ in SCHEMES]
TURN_VARIANTS = [name for name, kind, *_ in SCHEMES if kind != 'gae']


def step0_advantages(values, reward, heads):
    """The advantage of step 0 (decision K, the forced option) under every scheme; the game result is the only reward.
    values: V of the trajectory's steps; reward: the pseudo-game's result; heads: the steps' head rows."""
    values = np.asarray(values, dtype=np.float64)
    rewards = step_rewards({'reward': reward}, len(values))
    anchors = turn_anchors(heads)
    out = {}
    for name, kind, parameter, _ in SCHEMES:
        if kind == 'gae':
            out[name] = float(gae(values, rewards, parameter)[0])
        elif kind == 'rounds':
            out[name] = float(turn_credit(values, rewards, anchors, rounds=parameter)[0])
        else:
            out[name] = float(turn_credit(values, rewards, anchors, round_lambda=parameter)[0])
    return out


def seat_reward(outcome, winner, side):
    """The branching seat's result as the trajectory writer books it: +1 won, -1 lost, 0 unfinished."""
    if outcome != 'finished' or winner not in ('ds', 'ls'):
        return 0.0
    return 1.0 if winner == side else -1.0


# ---------------------------------------------------------------- statistics (weights = position multiplicities)

def wpearson(x, y, w):
    """Pearson correlation with non-negative weights (integer weights = the rows repeated); None if undefined."""
    total = w.sum()
    if total <= 0 or np.count_nonzero(w) < 3:
        return None
    dx = x - (w @ x) / total
    dy = y - (w @ y) / total
    vx, vy = w @ (dx * dx), w @ (dy * dy)
    if vx <= 1e-18 or vy <= 1e-18:
        return None
    return float((w @ (dx * dy)) / math.sqrt(vx * vy))


def build_pairs(truth, samples):
    """Arrays of the readout. truth: {position: {'options', 'pi', 'a_star', 'h1', 'h2'}} (dicts by option; h1/h2 None if
    a half lacks an option); samples: [(position, option, {scheme: advantage})]. Only (position, option) pairs with
    samples and truth count; positions need >= 2 such options."""
    by_pair = collections.defaultdict(list)
    for position, option, advantages in samples:
        if position in truth and option in truth[position]['a_star']:
            by_pair[(position, option)].append(advantages)
    options_with = collections.defaultdict(list)
    for position, option in by_pair:
        options_with[position].append(option)
    positions = sorted(p for p, options in options_with.items() if len(options) >= 2)
    index_of = {p: i for i, p in enumerate(positions)}
    pair_rows, sample_rows = [], []
    for p in positions:
        t = truth[p]
        options = sorted(options_with[p])
        pi = np.array([t['pi'][o] for o in options], float)
        pi = pi / pi.sum() if pi.sum() > 0 else np.full(len(options), 1 / len(options))
        means = {name: np.array([np.mean([s[name] for s in by_pair[(p, o)]]) for o in options]) for name in NAMES}
        centre = {name: float(pi @ means[name]) for name in NAMES}
        for k, o in enumerate(options):
            pair_rows.append({'position': index_of[p], 'a_star': t['a_star'][o],
                              'h1': None if t['h1'] is None else t['h1'][o], 'h2': None if t['h2'] is None else t['h2'][o],
                              'n': len(by_pair[(p, o)]),
                              **{name: means[name][k] for name in NAMES},
                              **{name + '|c': means[name][k] - centre[name] for name in NAMES}})
            for s in by_pair[(p, o)]:
                sample_rows.append({'position': index_of[p], 'pair': len(pair_rows) - 1, 'a_star': t['a_star'][o],
                                    **{name: s[name] for name in NAMES},
                                    **{name + '|c': s[name] - centre[name] for name in NAMES}})
    def column(rows, key, dtype=float):
        return np.array([r[key] for r in rows], dtype=dtype)
    split = np.array([r['h1'] is not None for r in pair_rows], bool)
    data = {'positions': positions,
            'pair_position': column(pair_rows, 'position', int), 'pair_a_star': column(pair_rows, 'a_star'),
            'pair_split': split,
            'pair_h1': np.array([r['h1'] if r['h1'] is not None else 0.0 for r in pair_rows], float),
            'pair_h2': np.array([r['h2'] if r['h2'] is not None else 0.0 for r in pair_rows], float),
            'pair_n': column(pair_rows, 'n', int),
            'sample_position': column(sample_rows, 'position', int), 'sample_pair': column(sample_rows, 'pair', int),
            'sample_a_star': column(sample_rows, 'a_star')}
    for name in NAMES:
        data['pair|' + name] = column(pair_rows, name)
        data['pair|' + name + '|c'] = column(pair_rows, name + '|c')
        data['sample|' + name] = column(sample_rows, name)
        data['sample|' + name + '|c'] = column(sample_rows, name + '|c')
    return data


def core(data, weights):
    """All readouts for one weighting of the positions (ones = the data; bootstrap = multiplicities)."""
    pair_w = weights[data['pair_position']].astype(float)
    sample_w = weights[data['sample_position']].astype(float)
    split = data['pair_split']
    half = wpearson(data['pair_h1'][split], data['pair_h2'][split], pair_w[split]) if split.any() else None
    reliability = spearman_brown(half) if half is not None else None
    root = math.sqrt(reliability) if reliability is not None and reliability > 0 else None
    def dis(value):
        return None if value is None or root is None else value / root
    out = {'split_half_r': half, 'reliability': reliability}
    for name in NAMES:
        values = data['sample|' + name]
        r1 = wpearson(values, data['sample_a_star'], sample_w)
        r1c = wpearson(data['sample|' + name + '|c'], data['sample_a_star'], sample_w)
        r2 = wpearson(data['pair|' + name], data['pair_a_star'], pair_w)
        r2c = wpearson(data['pair|' + name + '|c'], data['pair_a_star'], pair_w)
        total_w = sample_w.sum()
        grand = (sample_w @ values) / total_w if total_w > 0 else 0.0
        within = sample_w @ (values - data['pair|' + name][data['sample_pair']]) ** 2
        total = sample_w @ (values - grand) ** 2
        out[name] = {'signal': r1, 'signal_dis': dis(r1), 'signal_centred': r1c, 'signal_centred_dis': dis(r1c),
                     'bias': r2, 'bias_dis': dis(r2), 'bias_centred': r2c, 'bias_centred_dis': dis(r2c),
                     'noise_share': float(within / total) if total > 0 else None,
                     'variance': float(total / total_w) if total_w > 0 else None}
    reference = out[REFERENCE]['signal_dis']
    for name in NAMES:
        value = out[name]['signal_dis']
        out[name]['signal_dis_minus_reference'] = None if value is None or reference is None else value - reference
    return out


KEYS = ('signal', 'signal_dis', 'signal_centred', 'signal_centred_dis', 'bias', 'bias_dis', 'bias_centred',
        'bias_centred_dis', 'noise_share', 'signal_dis_minus_reference')


def readout(truth, samples, boot=1000, seed=20261019):
    """The stage-A readout: point estimates, bootstrap 95% intervals over positions, and the choice."""
    data = build_pairs(truth, samples)
    count = len(data['positions'])
    if count < 3:
        return {'positions_scored': count, 'error': 'fewer than 3 positions with 2+ options scored'}
    main = core(data, np.ones(count, int))
    rng = np.random.default_rng(seed)
    collected = collections.defaultdict(list)
    for _ in range(boot):
        weights = np.bincount(rng.integers(0, count, count), minlength=count)
        result = core(data, weights)
        if result['reliability'] is not None:
            collected['reliability'].append(result['reliability'])
        for name in NAMES:
            for key in KEYS:
                if result[name][key] is not None:
                    collected[(name, key)].append(result[name][key])
    def interval(key):
        values = collected[key]
        if len(values) < boot // 2:
            return None
        return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]
    def rounded(value):
        return None if value is None else round(float(value), 4)
    schemes = {}
    for name, kind, parameter, flags in SCHEMES:
        row = {'kind': kind, 'parameter': parameter, 'flags': flags}
        for key in KEYS:
            row[key] = rounded(main[name][key])
            row[key + '_ci95'] = interval((name, key))
        row['variance'] = rounded(main[name]['variance'])
        schemes[name] = row
    # The disattenuation divides every scheme by the same factor: rank by the raw per-sample Pearson (the same order,
    # and defined even if the truth's reliability is not positive).
    ranked = sorted(TURN_VARIANTS, key=lambda name: -np.inf if main[name]['signal'] is None else main[name]['signal'],
                    reverse=True)
    choice = [{'name': name, 'flags': schemes[name]['flags'], 'signal': schemes[name]['signal'],
               'signal_dis': schemes[name]['signal_dis']} for name in ranked[:2]]
    return {'positions_scored': count, 'pairs': int(len(data['pair_a_star'])), 'samples': int(len(data['sample_a_star'])),
            'rollouts_per_pair': {'min': int(data['pair_n'].min()), 'median': float(np.median(data['pair_n'])),
                                  'max': int(data['pair_n'].max())},
            'positions_split_half': int(len(set(data['pair_position'][data['pair_split']].tolist()))),
            'truth_split_half_r': rounded(main['split_half_r']), 'truth_reliability': rounded(main['reliability']),
            'truth_reliability_ci95': interval('reliability'), 'boot': boot, 'schemes': schemes,
            'turn_ranking': ranked, 'choice': choice,
            'rule': 'choice = the two turn-credit variants (tc-r1/2/3, tc-rl05/07) with the highest per-sample signal '
                    '(point estimate; the disattenuation factor is common to all schemes)'}


# ---------------------------------------------------------------- inputs

def truth_table(truth_dir, positions_file):
    """E2's truth per position: options with rollouts, pi (as listed), A* and the two halves' A* (dicts by option)."""
    spec = json.loads(Path(positions_file).read_text())
    _, rows = rollout_outcomes(truth_dir)
    outcomes = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, position, option, rollout, won in rows:
        outcomes[position][option].append((rollout, won))
    truth = {}
    for r in spec['list']:
        p = r['position']
        usable = [o for o in range(r['options']) if outcomes[p][o]]
        if len(usable) < 2:
            continue
        pi = [r['pi'][o] for o in usable]
        _, a_star, rate = true_advantages(pi, [[w for _, w in outcomes[p][o]] for o in usable])
        halves = []
        for parity in (0, 1):
            part = [[w for k, w in outcomes[p][o] if k % 2 == parity] for o in usable]
            halves.append(None if any(not w for w in part) else dict(zip(usable, true_advantages(pi, part)[1])))
        truth[p] = {'options': usable, 'pi': dict(zip(usable, pi)), 'a_star': dict(zip(usable, a_star)),
                    'rate': dict(zip(usable, rate)), 'rollouts': {o: len(outcomes[p][o]) for o in usable},
                    'h1': halves[0], 'h2': halves[1]}
    return spec, truth, len(rows)


def stage_reports(stage_a):
    """{job: (status, outcome, winner)} from the stage-A replay reports (merged, else the chunk files); status is
    'mismatch' or 'exact'. Also the mismatch line count."""
    merged = Path(stage_a, 'report-tc.txt')
    files = [merged] if merged.exists() else sorted(Path(stage_a).glob('report-tc.c*.txt'))
    ends, mismatched, lines = {}, set(), 0
    for path in files:
        for line in path.read_text().splitlines():
            parts = line.split('\t')
            g = int(parts[0].rsplit('-', 1)[1])
            if parts[1] == 'mismatch':
                mismatched.add(g)
                lines += 1
            elif parts[1] == 'end':
                ends[g] = (parts[5] if len(parts) > 5 else 'exact', parts[3], parts[4] if len(parts) > 4 else '')
    for g in mismatched:
        if g in ends:
            ends[g] = ('mismatch',) + ends[g][1:]
    return ends, lines


def trace_dirs(stage_a):
    """The chunks' trace folders (G/chunkN.DONE names each one), else every folder under G/trace."""
    dirs = []
    for path in sorted(Path(stage_a).glob('chunk*.DONE')):
        try:
            name = Path(json.loads(path.read_text())['trace']).name
        except (ValueError, KeyError, TypeError):
            continue
        dirs.append(Path(stage_a, 'trace', name))
    if not dirs:
        dirs = sorted(p for p in Path(stage_a, 'trace').glob('*') if p.is_dir())
    return [d for d in dirs if d.is_dir()]


def load_file(path, tables, wanted):
    """The wanted pseudo-games of one trajectory file, reduced to what the readout needs (critic items and heads)."""
    from swccg_policy.v4_trajectory import TrajectoryReader
    games = []
    for game in TrajectoryReader(tables, []).load(path):
        if game['g'] in wanted:
            games.append({'g': game['g'], 's': game['s'], 'o': game['o'], 'w': game['w'], 'reward': game['reward'],
                          'critic_items': game['critic_items'], 'heads': [item['head'] for item in game['items']],
                          'seqs': game['seqs']})
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage-a', required=True, help='the stage-A folder (scripts/tc_stage_a.sh G)')
    parser.add_argument('--truth', required=True, help="E2's ground-truth folder (runs/cn-replay/gt)")
    parser.add_argument('--positions', help="E2's positions.json (default TRUTH/positions.json)")
    parser.add_argument('--checkpoint', required=True, help="D: the critic whose values the schemes use")
    parser.add_argument('--out', required=True, help='the readout JSON (a markdown table goes next to it, .md)')
    parser.add_argument('--boot', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=256, help='items per value call (token-budgeted microbatches)')
    parser.add_argument('--loaders', type=int, default=4, help='threads decoding trajectory files ahead of the GPU')
    args = parser.parse_args()
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    started = time.time()
    spec, truth, truth_rollouts = truth_table(args.truth, args.positions or Path(args.truth, 'positions.json'))
    meta = json.loads(Path(args.stage_a, 'meta-tc.json').read_text())
    ends, mismatch_lines = stage_reports(args.stage_a)
    described = {}
    for directory in trace_dirs(args.stage_a):
        log = directory / 'trace.jsonl'
        if log.exists():
            for line in log.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    described[row['id']] = row
    wanted = {f'rt|{g}' for g, (status, outcome, _) in ends.items() if status == 'exact' and outcome != 'exception'}
    _, critic, tables, _ = load(args.checkpoint, 'cuda')
    critic.eval()
    critic.encoder.packed_attention = False  # fp32 values, as E2's readout (advantage_readout power)
    config = PPOConfig(bf16=False)
    files = [str(f) for d in trace_dirs(args.stage_a) for f in sorted((d / 'traj').glob('*.traj'))]
    samples, rows, problems = [], [], collections.Counter()
    seen, steps_total, gpu_seconds = set(), 0, 0.0
    with concurrent.futures.ThreadPoolExecutor(max(1, args.loaders)) as pool:
        futures = [pool.submit(load_file, f, tables, wanted) for f in files[:max(1, args.loaders) * 2]]
        next_file = len(futures)
        while futures:
            games = futures.pop(0).result()
            if next_file < len(files):
                futures.append(pool.submit(load_file, files[next_file], tables, wanted))
                next_file += 1
            games = [game for game in games if game['g'] not in seen]
            if not games:
                continue
            items = [item for game in games for item in game['critic_items']]
            t0 = time.time()
            values = predict_values(critic, items, config, args.batch)
            gpu_seconds += time.time() - t0
            offset = 0
            for game in games:
                count = len(game['critic_items'])
                v = values[offset:offset + count]
                offset += count
                seen.add(game['g'])
                g = int(game['g'].split('|')[1])
                m = meta[g]
                status, outcome, winner = ends[g]
                d = described.get(game['g'], {})
                reward = float(game['reward'])
                if reward != seat_reward(outcome, winner, game['s']):
                    problems['reward differs from the report outcome'] += 1
                if d and (d.get('side') != game['s'] or d.get('steps') != count or d.get('forced') != m['option']):
                    problems['trace log differs (side, steps or forced option)'] += 1
                if game['seqs'][0] != [m['option']]:
                    problems['step 0 does not hold the forced option'] += 1
                heads = game['heads']
                if any(int(h[3]) != int(heads[0][3]) for h in heads):
                    problems['seat changes within a trace'] += 1
                anchors = turn_anchors(heads)
                turns = [int(heads[a][4]) for a in anchors]
                advantages = step0_advantages(v, reward, heads)
                samples.append((m['position'], m['option'], advantages))
                rows.append({'job': g, 'position': m['position'], 'option': m['option'], 'rollout': m['rollout'],
                             'reward': reward, 'outcome': outcome, 'steps': count, 'v0': float(v[0]),
                             'anchors': len(anchors), 'my_turn0': int(heads[0][7]),
                             'consecutive_turns': all(b - a == 1 for a, b in zip(turns, turns[1:])),
                             'round_lengths': [b - a for a, b in zip(anchors, anchors[1:])], **advantages})
                steps_total += count
            del games, items, values
            if len(rows) % 500 < 10:
                print(f'{len(rows)} rollouts, {steps_total} steps, {time.time() - started:.0f} s', flush=True)
    del critic
    torch.cuda.empty_cache()
    result = readout(truth, samples, args.boot)
    # Diagnostics: exactness (V(s_K) is one state per position), trace shape, timing.
    v0 = collections.defaultdict(list)
    for row in rows:
        v0[row['position']].append(row['v0'])
    spread = max((max(v) - min(v) for v in v0.values()), default=None)
    rounds = [length for row in rows for length in row['round_lengths']]
    steps = np.array([row['steps'] for row in rows]) if rows else np.zeros(1)
    planned = len(meta)
    result = {
        'stage_a': str(args.stage_a), 'truth': str(args.truth), 'checkpoint': str(args.checkpoint),
        'values': 'fp32 (PPOConfig(bf16=False), standard attention), as E2 advantage_readout power',
        'rewards': 'the game result only (+1 / -1 / 0 from the branching seat), no Dojo rewards',
        'rollouts_planned': planned, 'rollouts_reported': len(ends),
        'rollouts_mismatched': sum(status == 'mismatch' for status, *_ in ends.values()),
        'mismatch_lines': mismatch_lines,
        'rollouts_exception': sum(outcome == 'exception' for _, outcome, _ in ends.values()),
        'rollouts_unfinished': sum(outcome not in ('finished', 'exception') for _, outcome, _ in ends.values()),
        'rollouts_traced': len(rows), 'rollouts_usable': result.get('samples', 0),
        'truth_rollouts': truth_rollouts, 'truth_positions': len(truth),
        'problems': dict(problems),
        'trace': {'steps_total': int(steps_total), 'steps_mean': round(float(steps.mean()), 1),
                  'steps_median': float(np.median(steps)), 'steps_p10': float(np.percentile(steps, 10)),
                  'steps_p90': float(np.percentile(steps, 90)),
                  'anchors_mean': round(float(np.mean([row['anchors'] for row in rows])), 2) if rows else None,
                  'my_turn_at_k_share': round(float(np.mean([row['my_turn0'] for row in rows])), 4) if rows else None,
                  'consecutive_anchor_turns_share': round(float(np.mean([row['consecutive_turns'] for row in rows])), 4)
                  if rows else None,
                  'decisions_per_round_median': float(np.median(rounds)) if rounds else None,
                  'decisions_per_round_p10_p90': [float(np.percentile(rounds, 10)), float(np.percentile(rounds, 90))]
                  if rounds else None,
                  'v0_max_spread_within_position': None if spread is None else float(spread),
                  'wins': sum(row['reward'] > 0 for row in rows), 'losses': sum(row['reward'] < 0 for row in rows),
                  'zero': sum(row['reward'] == 0 for row in rows)},
        'seconds': {'total': round(time.time() - started, 1), 'gpu': round(gpu_seconds, 1)},
        **result}
    e2_power = Path(args.truth, 'power.json')
    if e2_power.exists():
        for value in json.loads(e2_power.read_text()).values():
            if isinstance(value, dict) and 'reliability' in value:
                result['e2_power_reliability'] = value['reliability']
                break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    out.with_name(out.stem + '.rows.jsonl').write_text(
        ''.join(json.dumps({k: v for k, v in row.items() if k != 'round_lengths'}) + '\n' for row in rows))
    out.with_suffix('.md').write_text(markdown(result))
    print(markdown(result))
    print(json.dumps({'choice': result.get('choice'), 'usable': result['rollouts_usable'],
                      'mismatched': result['rollouts_mismatched'], 'problems': result['problems']}))


def markdown(result):
    """The stage-A table (per-sample signal, pi-centred, bias, noise share, difference to GAE(0.95))."""
    def cell(row, key):
        value, ci = row.get(key), row.get(key + '_ci95')
        if value is None:
            return 'n/a'
        return f'{value:.3f}' + (f' [{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '')
    lines = [f"Positions {result.get('positions_scored')}, pairs {result.get('pairs')}, rollouts usable "
             f"{result.get('rollouts_usable')} of {result.get('rollouts_planned')} (mismatched "
             f"{result.get('rollouts_mismatched')}); truth reliability {result.get('truth_reliability')} "
             f"{result.get('truth_reliability_ci95')}.", '',
             '| Scheme | Per-sample signal (disatt.) | π-centred (disatt.) | Bias (disatt.) | Noise share | '
             'Δ signal vs GAE(0.95) |', '|---|---|---|---|---|---|']
    for name, row in result.get('schemes', {}).items():
        lines.append(f"| {name} | {cell(row, 'signal_dis')} | {cell(row, 'signal_centred_dis')} | "
                     f"{cell(row, 'bias_dis')} | {cell(row, 'noise_share')} | "
                     f"{cell(row, 'signal_dis_minus_reference')} |")
    lines += ['', f"Choice: {json.dumps(result.get('choice'))}", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
