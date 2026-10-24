"""Plan critic-followups, BL-49: the advantage-quality readout.

Ground truth (Stage B, runs/cf-replay): positions of recorded games, each option forced (ReplayControl) and played to
the end ROLLOUTS times by the phase-3 control policy against itself. Per option: the win rate of the branching seat
(report-*.txt "end" lines + the rollout CSV), and the critic input of the seat's first decision after the option
(replay.states pseudo-games rb|g; deterministic per option, so one value per option).

Readout per critic: at each position, the ranking of the options by V(state after the option) against the ranking by
the true win rate. V(state before) is the same for every option of a position, so this is the ranking of the one-step
advantages. Scores: mean Spearman rho over positions, the share of positions whose critic top option is the true best
(positions with a true best that leads the others by >= 2 standard errors), and bootstrap intervals over positions.

    advantage_readout.py jobs  --record DIR --report-p1 FILE... --out DIR/jobs-gt.tsv --positions 300 --rollouts 16
    advantage_readout.py score --gt DIR --critic NAME=CKPT ... --out JSON

Plan critic-noise, stage 1b (E2, BL-52): the powered readout. Positions come from fresh self-play games recorded with
position states (ReplayControl record mode with replay.states: the policy's input at every decision with 2-6 options).
`positions` keeps own-turn decisions of turns 3-12 with 2-6 options (single picks) where the deciding seat's critic
value is in [-0.5, 0.5], stores the policy's probabilities pi (T = 1, the encoder's answer order, which the forced
option indexes) and writes the rollout jobs (every option, ROLLOUTS times; scripts/cn_groundtruth.sh plays them).
`power` scores a critic against them:
- true advantage A*(o) = winrate(o) - sum_o' pi(o') winrate(o');
- critic advantage V(state after o) - V(state now) (the one-step advantage of BL-49);
- reliability of A*: split-half (even / odd rollouts) Pearson, Spearman-Brown corrected to the full count;
- Pearson(critic advantage, A*) over all (position, option) pairs, disattenuated by the reliability, with a bootstrap
  95% interval over positions; per-position Spearman (BL-49's statistic), top-1 share, sign agreement;
- E2 holds iff the disattenuated r <= 0.30 and its upper 95% bound <= 0.45; "unmeasured" if the reliability < 0.5.

    advantage_readout.py positions --source EVAL_DIR --record DIR --states DIR --checkpoint CKPT --out G \
        --positions 150 --rollouts 48
    advantage_readout.py power --gt G [--gt G2 ...] --positions G/positions.json --critic NAME=CKPT ... --out JSON

Plan critic-ceiling: `power --view actor` scores the critics on the actor items of the same states (the critic network
on the actor's view; default `critic`: the critic items, unchanged); `--rows FILE` also writes each critic's
per-position inputs (pi, one-step advantages, outcomes) for paired comparisons (critic_ceiling_readout.py option).
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import random
from pathlib import Path

import numpy as np


def cmd_jobs(args):
    """Positions with 2-6 options from pass-1 reports; one job line per (position, option, rollout)."""
    positions = []
    for report in args.report_p1:
        stem = Path(report).stem.replace('report-', '')
        jobs = Path(report).with_name(f'jobs-{stem}.tsv').read_text().splitlines()
        sched = Path(report).with_name(f'sched-{stem}.tsv').read_text().splitlines()
        for line in Path(report).read_text().splitlines():
            parts = line.split('\t')
            if parts[1] != 'branch':
                continue
            count = int(parts[3])
            g = int(parts[0].rsplit('-', 1)[1])
            if 2 <= count <= 6:
                positions.append((jobs[g], sched[g], count))
    rng = random.Random(args.seed)
    rng.shuffle(positions)
    positions = positions[:args.positions]
    lines, rows, meta = [], [], []
    for p, (job, sched_row, count) in enumerate(positions):
        seed, record, k, _, _ = job.split('\t')
        for option in range(count):
            for r in range(args.rollouts):
                lines.append(f'{seed}\t{record}\t{k}\t{option}\t0')
                rows.append(sched_row)
                meta.append({'position': p, 'option': option, 'rollout': r})
    out = Path(args.out)
    out.write_text('\n'.join(lines) + '\n')
    out.with_name(out.name.replace('jobs-', 'sched-')).write_text('\n'.join(rows) + '\n')
    out.with_name(out.name.replace('jobs-', 'meta-').replace('.tsv', '.json')).write_text(json.dumps(meta))
    print(len(positions), 'positions,', len(lines), 'rollouts')


def rollout_outcomes(gt):
    """meta and [(game index, position, option, rollout, won)] of the usable rollouts of a ground-truth folder: no
    replay mismatch, an end line, and an outcome state of the branching seat (which gives its side)."""
    meta = json.loads(Path(gt, 'meta-gt.json').read_text())
    branch_player, outcome = {}, {}
    for line in Path(gt, 'report-gt.txt').read_text().splitlines():
        parts = line.split('\t')
        g = int(parts[0].rsplit('-', 1)[1])
        if parts[1] == 'mismatch':
            outcome[g] = None
        elif parts[1] == 'end' and g not in outcome:
            outcome[g] = (parts[3], parts[4])
    sides = {}
    for path in sorted(Path(gt, 'states').rglob('states.jsonl')):  # one folder, or one per chunk (states/cN)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            sides[row['game']] = row['side']
    rows = []
    for g, m in enumerate(meta):
        if g not in sides or outcome.get(g) is None:
            continue
        status, winner = outcome[g]
        rows.append((g, m['position'], m['option'], m['rollout'], int(status == 'finished' and winner == sides[g])))
    return meta, rows


def ground_truth(gt):
    """{position: {option: [wins, games]}} and {game index: state id} from the Stage B run folder."""
    meta, rows = rollout_outcomes(gt)
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for _, position, option, _, won in rows:
        cell = table[position][option]
        cell[1] += 1
        cell[0] += won
    return meta, table


def ranks(x):
    """Ranks with ties averaged (a tie must not create a correlation)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        r[tied] = r[tied].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_score(args):
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    from swccg_policy.v4_trajectory import TrajectoryReader
    metas, table = [], collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for gt in args.gt:  # parts of one job list (a resumed run): positions and options keep their numbers
        meta, part = ground_truth(gt)
        metas.append(meta)
        for position, options in part.items():
            for option, (wins, games) in options.items():
                table[position][option][0] += wins
                table[position][option][1] += games
    out = {'positions': len(table)}
    states = None
    for spec in args.critic:
        name, path = spec.split('=', 1)
        actor, critic, tables, _ = load(path, 'cuda')
        critic.eval()
        critic.encoder.packed_attention = False
        if states is None:
            reader = TrajectoryReader(tables, [])
            states = {}
            for part, gt in enumerate(args.gt):
                for f in sorted(glob.glob(os.path.join(gt, 'states', '**', '*.traj'), recursive=True)):
                    for game in reader.load(f):
                        states[(part, int(game['g'].split('|')[1]))] = game['critic_items'][0]
        keys = sorted(states)
        values = dict(zip(keys, predict_values(critic, [states[k] for k in keys], PPOConfig(bf16=False), 64)))
        per_option = collections.defaultdict(dict)
        for part, g in keys:
            m = metas[part][g]
            per_option[m['position']].setdefault(m['option'], values[(part, g)])
        rhos, tops, tops_clear = [], [], []
        for position, options in table.items():
            keys = sorted(o for o in options if options[o][1] >= 4 and o in per_option[position])
            if len(keys) < 2:
                continue
            truth = np.array([options[o][0] / options[o][1] for o in keys])
            critic_values = np.array([per_option[position][o] for o in keys])
            rho = spearman(critic_values, truth)
            if rho is not None:
                rhos.append(rho)
            order = np.argsort(-truth)
            best, second = order[0], order[1]
            n = np.array([options[o][1] for o in keys])
            se = math.sqrt(truth[best] * (1 - truth[best]) / n[best] + truth[second] * (1 - truth[second]) / n[second])
            hit = int(np.argmax(critic_values) == best)
            tops.append(hit)
            if truth[best] - truth[second] >= 2 * max(se, 1e-9):
                tops_clear.append(hit)
        rng = np.random.default_rng(20261018)
        boot = [float(np.mean(rng.choice(rhos, len(rhos)))) for _ in range(2000)] if rhos else []
        out[name] = {'positions_scored': len(rhos), 'spearman_mean': round(float(np.mean(rhos)), 4) if rhos else None,
                     'spearman_ci95': [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)] if boot else None,
                     'top1_share': round(float(np.mean(tops)), 4) if tops else None,
                     'top1_share_clear_best': round(float(np.mean(tops_clear)), 4) if tops_clear else None,
                     'clear_best_positions': len(tops_clear), '_rhos': rhos}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != '_rhos'}), flush=True)
        del actor, critic
        torch.cuda.empty_cache()
    names = [s.split('=', 1)[0] for s in args.critic]
    if len(names) >= 2:  # paired difference of the first two critics over the same positions
        a, b = np.array(out[names[0]]['_rhos']), np.array(out[names[1]]['_rhos'])
        if len(a) == len(b) and len(a):
            rng = np.random.default_rng(20261019)
            idx = [rng.integers(0, len(a), len(a)) for _ in range(2000)]
            diffs = sorted(float(np.mean(a[i] - b[i])) for i in idx)
            out['paired'] = {'pair': names[:2], 'spearman_difference': round(float(np.mean(a - b)), 4),
                             'ci95': [round(diffs[50], 4), round(diffs[1949], 4)]}
            print('paired', json.dumps(out['paired']))
    for name in names:
        out[name].pop('_rhos', None)
    Path(args.out).write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- plan critic-noise stage 1b (E2, BL-52)

E2_MAX_R, E2_MAX_UPPER, MIN_RELIABILITY = 0.30, 0.45, 0.5


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman_brown(r, factor=2.0):
    """The reliability of a measure `factor` times as long, from the reliability r of one part."""
    return factor * r / (1 + (factor - 1) * r)


def true_advantages(pi, wins):
    """A*(o) = winrate(o) - sum pi(o') winrate(o') over the options with rollouts (pi renormalised over them)."""
    keep = [o for o, w in enumerate(wins) if len(w)]
    rate = np.array([np.mean(wins[o]) for o in keep])
    weight = np.array([pi[o] for o in keep], float)
    weight = weight / weight.sum() if weight.sum() > 0 else np.full(len(keep), 1 / len(keep))
    return keep, rate - float(weight @ rate), rate


def power_pairs(positions):
    """Per position: options kept, critic advantage, A* (all rollouts) and A* of the two halves (even / odd rollout
    numbers; None if a half lacks an option). positions: dicts with pi, adv (critic one-step advantage per option,
    None if unknown), outcomes (per option a list of (rollout, won))."""
    out = []
    for p in positions:
        usable = [o for o in range(len(p['pi'])) if p['adv'][o] is not None and p['outcomes'][o]]
        if len(usable) < 2:
            continue
        pi = [p['pi'][o] for o in usable]
        wins = [[w for _, w in p['outcomes'][o]] for o in usable]
        _, a_star, rate = true_advantages(pi, wins)
        halves = []
        for parity in (0, 1):
            part = [[w for r, w in p['outcomes'][o] if r % 2 == parity] for o in usable]
            halves.append(None if any(not w for w in part) else true_advantages(pi, part)[1])
        adv = np.array([p['adv'][o] for o in usable], float)
        centred = adv - float(np.asarray(pi) / np.sum(pi) @ adv)
        out.append({'adv': adv, 'centred': centred, 'a_star': a_star, 'rate': rate, 'h1': halves[0], 'h2': halves[1]})
    return out


def power_core(pairs, index=None):
    """Pooled statistics over the positions `index` (default all): reliability (split-half, Spearman-Brown), observed
    and disattenuated Pearson of the critic advantage (and the pi-centred variant) with A*."""
    chosen = [pairs[i] for i in (range(len(pairs)) if index is None else index)]
    adv = np.concatenate([p['adv'] for p in chosen])
    centred = np.concatenate([p['centred'] for p in chosen])
    a_star = np.concatenate([p['a_star'] for p in chosen])
    split = [p for p in chosen if p['h1'] is not None]
    half = pearson(np.concatenate([p['h1'] for p in split]), np.concatenate([p['h2'] for p in split])) if split else None
    reliability = spearman_brown(half) if half is not None else None
    r, r_centred = pearson(adv, a_star), pearson(centred, a_star)
    def dis(value):
        return value / math.sqrt(reliability) if value is not None and reliability is not None and reliability > 0 else None
    return {'split_half_r': half, 'reliability': reliability, 'r': r, 'r_disattenuated': dis(r),
            'r_centred': r_centred, 'r_centred_disattenuated': dis(r_centred), 'pairs': len(a_star)}


def verdict(r_dis, upper, reliability):
    """Plan critic-noise E2: holds iff r_dis <= 0.30 and its upper 95% bound <= 0.45; unmeasured below reliability 0.5."""
    if reliability is None or reliability < MIN_RELIABILITY or r_dis is None or upper is None:
        return 'unmeasured'
    return 'holds' if r_dis <= E2_MAX_R and upper <= E2_MAX_UPPER else 'fails'


def power_stats(positions, boot=2000, seed=20261018):
    """The E2 readout of one critic: pooled reliability and correlations, bootstrap over positions, per-position
    Spearman (critic V(after o) vs the true win rate), top-1 share and sign agreement, and the verdict."""
    pairs = power_pairs(positions)
    if not pairs:
        return {'positions_scored': 0, 'verdict': 'unmeasured'}
    main = power_core(pairs)
    rng = np.random.default_rng(seed)
    samples = collections.defaultdict(list)
    for _ in range(boot):
        result = power_core(pairs, rng.integers(0, len(pairs), len(pairs)))
        for key in ('r', 'r_disattenuated', 'reliability', 'r_centred_disattenuated'):
            if result[key] is not None:
                samples[key].append(result[key])
    def interval(key):
        values = samples[key]
        return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)] \
            if len(values) >= boot // 2 else None
    rhos, tops, signs = [], [], []
    for p in pairs:
        rho = spearman(p['adv'], p['rate'])
        if rho is not None:
            rhos.append(rho)
        if p['rate'].std() > 0:
            tops.append(int(p['rate'][np.argmax(p['adv'])] == p['rate'].max()))
        for a, b in zip(p['adv'], p['a_star']):
            if a != 0 and b != 0:
                signs.append(int((a > 0) == (b > 0)))
    ci = interval('r_disattenuated')
    def rounded(value):
        return None if value is None else round(value, 4)
    return {'positions_scored': len(pairs), 'pairs': main['pairs'],
            'positions_split_half': sum(p['h1'] is not None for p in pairs),
            'split_half_r': rounded(main['split_half_r']), 'reliability': rounded(main['reliability']),
            'reliability_ci95': interval('reliability'),
            'pearson': rounded(main['r']), 'pearson_ci95': interval('r'),
            'pearson_disattenuated': rounded(main['r_disattenuated']), 'pearson_disattenuated_ci95': ci,
            'pearson_centred': rounded(main['r_centred']),
            'pearson_centred_disattenuated': rounded(main['r_centred_disattenuated']),
            'pearson_centred_disattenuated_ci95': interval('r_centred_disattenuated'),
            'spearman_per_position_mean': rounded(float(np.mean(rhos))) if rhos else None,
            'spearman_positions': len(rhos), 'top1_share': rounded(float(np.mean(tops))) if tops else None,
            'top1_positions': len(tops), 'sign_agreement': rounded(float(np.mean(signs))) if signs else None,
            'sign_pairs': len(signs),
            'verdict': verdict(main['r_disattenuated'], ci[1] if ci else None, main['reliability']),
            'rule': f'E2 holds iff disattenuated r <= {E2_MAX_R} and its upper 95% bound <= {E2_MAX_UPPER}; '
                    f'unmeasured if the reliability < {MIN_RELIABILITY}'}


def load_states(directory, tables, wanted=None):
    """{pseudo-game id: (actor item, critic item)} of the one-step pseudo-games under directory (recursive)."""
    from swccg_policy.v4_trajectory import TrajectoryReader
    reader = TrajectoryReader(tables, [])
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, '**', '*.traj'), recursive=True)):
        for game in reader.load(f):
            if wanted is None or game['g'] in wanted:
                out[game['g']] = (game['items'][0], game['critic_items'][0])
    return out


def policy_probabilities(actor, items, device='cuda', batch=64):
    """pi at T = 1 (softmax over each item's options, in the encoder's answer order)."""
    import torch
    from swccg_policy.v4_data import stack
    actor.eval()
    out = []
    with torch.no_grad():
        actor.cache_card_table()
        for offset in range(0, len(items), batch):
            part = items[offset:offset + batch]
            data = stack(part, device)
            logits = actor(data).float().masked_fill(~data['omask'], float('-inf'))
            probs = torch.softmax(logits, -1).cpu().numpy()
            out.extend(probs[i, :len(item['opt'])].tolist() for i, item in enumerate(part))
    return out


def decision_texts(record):
    """{decision index: text} of a replay.record file (index, player, type, text, answer per line)."""
    out = {}
    for line in Path(record).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) >= 4:
            out[int(parts[0])] = parts[3]
    return out


def template_of(text, templates):
    """The first template the (stripped) decision text starts with, or None."""
    text = text.strip()
    return next((template for template in templates if text.startswith(template)), None)


def cmd_positions(args):
    """Select the positions from the recorded source games and write the rollout jobs (G/jobs-gt.tsv, sched, meta)."""
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    schedule = Path(args.source, 'schedule.tsv').read_text().splitlines()
    results = json.loads(Path(args.source, 'results.json').read_text())
    seed = int(results['seed']) if args.source_seed is None else args.source_seed
    records = {}
    for f in glob.glob(os.path.join(args.record, '*.tsv')):
        records[int(Path(f).stem.rsplit('-', 1)[1])] = os.path.abspath(f)
    types = set(args.types.split(','))
    rows = [json.loads(line) for line in Path(args.states, 'states.jsonl').read_text().splitlines() if line]
    candidates = [r for r in rows if r['own'] and args.turns[0] <= r['turn'] <= args.turns[1] and r['max'] == 1
                  and args.options[0] <= r['options'] <= args.options[1] and r['type'] in types and r['game'] in records]
    templates = None
    if args.templates:  # plan critic-ceiling stage C: keep decisions whose recorded text starts with a listed template
        templates = [line.strip() for line in Path(args.templates).read_text().splitlines() if line.strip()]
        texts = {}
        for g in {r['game'] for r in candidates}:
            texts.update(((g, index), text) for index, text in decision_texts(records[g]).items())
        candidates = [{**r, 'template': template_of(texts.get((r['game'], r['index']), ''), templates)}
                      for r in candidates]
        candidates = [r for r in candidates if r['template'] is not None]
    actor, critic, tables, _ = load(args.checkpoint, 'cuda')
    critic.eval()
    critic.encoder.packed_attention = False
    states = load_states(args.states, tables, {r['id'] for r in candidates})
    candidates = [r for r in candidates if r['id'] in states]
    values = predict_values(critic, [states[r['id']][1] for r in candidates], PPOConfig(bf16=False), 64)
    pis = policy_probabilities(actor, [states[r['id']][0] for r in candidates])
    del actor, critic
    torch.cuda.empty_cache()
    pool = collections.defaultdict(list)
    mismatched = 0
    for r, value, pi in zip(candidates, values, pis):
        if len(pi) != r['options']:
            mismatched += 1
            continue
        if abs(float(value)) <= args.value_band:
            pool[r['game']].append({**r, 'v_now': float(value), 'pi': [round(x, 6) for x in pi]})
    rng = random.Random(args.seed)
    games = sorted(pool)
    rng.shuffle(games)
    chosen = []  # round robin over the games: at most per-game positions from one game
    for _ in range(args.per_game):
        for g in games:
            if len(chosen) < args.positions and pool[g]:
                chosen.append(pool[g].pop(rng.randrange(len(pool[g]))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, sched, meta = [], [], []
    for p, r in enumerate(chosen):
        r.update(position=p, seed=seed + r['game'], record=records[r['game']], schedule=schedule[r['game']])
        for option in range(r['options']):
            for rollout in range(args.rollouts):
                lines.append(f"{r['seed']}\t{r['record']}\t{r['index']}\t{option}\t0")
                sched.append(r['schedule'])
                meta.append({'position': p, 'option': option, 'rollout': rollout})
    (out / 'jobs-gt.tsv').write_text('\n'.join(lines) + '\n')
    (out / 'sched-gt.tsv').write_text('\n'.join(sched) + '\n')
    (out / 'meta-gt.json').write_text(json.dumps(meta))
    summary = {'source': str(args.source), 'states': str(args.states), 'checkpoint': str(args.checkpoint),
               'candidates': len(candidates), 'option_count_mismatch': mismatched,
               'in_value_band': sum(len(v) for v in pool.values()) + len(chosen), 'games_with_positions': len(games),
               'positions': len(chosen), 'rollouts_per_option': args.rollouts, 'rollouts': len(lines),
               'selection': {'turns': args.turns, 'options': args.options, 'value_band': args.value_band,
                             'types': sorted(types), 'per_game': args.per_game, 'seed': args.seed,
                             'templates': templates}}
    if templates is not None:
        summary['templates_chosen'] = dict(collections.Counter(r['template'] for r in chosen))
    (out / 'positions.json').write_text(json.dumps({**summary, 'list': chosen}, indent=1))
    print(json.dumps(summary))


def cmd_power(args):
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_rl import predict_values
    spec = json.loads(Path(args.positions).read_text())
    listed = spec['list']
    outcomes = collections.defaultdict(lambda: collections.defaultdict(list))
    rollout_state = {}  # (position, option) -> (part, game index) of one outcome state (deterministic per option)
    for part, gt in enumerate(args.gt):
        _, rows = rollout_outcomes(gt)
        for g, position, option, rollout, won in rows:
            outcomes[position][option].append((rollout, won))
            rollout_state.setdefault((position, option), (part, g))
    out = {'positions': len(listed), 'rollouts_usable': sum(len(v) for o in outcomes.values() for v in o.values()),
           'rollouts_planned': spec['rollouts']}
    view = {'actor': 0, 'critic': 1}[args.view]  # load_states pairs: (actor item, critic item)
    dump = {}
    for item in args.critic:
        name, path = item.split('=', 1)
        actor, critic, tables, _ = load(path, 'cuda')
        del actor
        critic.eval()
        critic.encoder.packed_attention = False
        after_ids = {f'rb|{g}': key for key, (part, g) in rollout_state.items()}
        after = {}
        for part, gt in enumerate(args.gt):
            wanted = {f'rb|{g}' for key, (p, g) in rollout_state.items() if p == part}
            for sid, pair in load_states(os.path.join(gt, 'states'), tables, wanted).items():
                after[after_ids[sid]] = pair[view]
        now = load_states(spec['states'], tables, {r['id'] for r in listed})
        keys = sorted(after)
        v_after = dict(zip(keys, predict_values(critic, [after[k] for k in keys], PPOConfig(bf16=False), 64)))
        v_now = dict(zip([r['id'] for r in listed],
                         predict_values(critic, [now[r['id']][view] for r in listed], PPOConfig(bf16=False), 64)))
        positions = []
        for r in listed:
            p = r['position']
            adv = [float(v_after[(p, o)] - v_now[r['id']]) if (p, o) in v_after else None for o in range(r['options'])]
            positions.append({'pi': r['pi'], 'adv': adv, 'outcomes': [outcomes[p][o] for o in range(r['options'])]})
        out[name] = power_stats(positions, args.boot)
        out[name]['view'] = args.view
        dump[name] = [{'position': r['position'], 'id': r['id'], **p} for r, p in zip(listed, positions)]
        print(name, json.dumps(out[name]), flush=True)
        del critic
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(out, indent=1))
    if args.rows:
        Path(args.rows).write_text(json.dumps(dump))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    jobs = sub.add_parser('jobs')
    jobs.add_argument('--report-p1', nargs='+', required=True)
    jobs.add_argument('--out', required=True)
    jobs.add_argument('--positions', type=int, default=300)
    jobs.add_argument('--rollouts', type=int, default=16)
    jobs.add_argument('--seed', type=int, default=20261018)
    jobs.add_argument('--record')
    score = sub.add_parser('score')
    score.add_argument('--gt', action='append', required=True, help='a ground-truth folder; repeat for parts')
    score.add_argument('--critic', action='append', required=True)
    score.add_argument('--out', required=True)
    positions = sub.add_parser('positions')
    positions.add_argument('--source', required=True, help='the source evaluation folder (schedule.tsv, results.json)')
    positions.add_argument('--source-seed', type=int, help='the source run seed (default: results.json)')
    positions.add_argument('--record', required=True, help='replay.record folder of the source games')
    positions.add_argument('--states', required=True, help='replay.states folder of the source games')
    positions.add_argument('--checkpoint', required=True, help='the policy D: pi and the selection critic')
    positions.add_argument('--out', required=True, help='the ground-truth folder G')
    positions.add_argument('--positions', type=int, default=150)
    positions.add_argument('--per-game', type=int, default=2)
    positions.add_argument('--rollouts', type=int, default=48)
    positions.add_argument('--turns', type=int, nargs=2, default=[3, 12])
    positions.add_argument('--options', type=int, nargs=2, default=[2, 6])
    positions.add_argument('--value-band', type=float, default=0.5)
    positions.add_argument('--types', default='CARD_ACTION_CHOICE,CARD_SELECTION,MULTIPLE_CHOICE,INTEGER')
    positions.add_argument('--seed', type=int, default=20261018)
    positions.add_argument('--templates', help='plan critic-ceiling: a file of decision-text templates (one per line); '
                                               'only decisions whose text starts with one are kept')
    power = sub.add_parser('power')
    power.add_argument('--gt', action='append', required=True, help='a ground-truth folder; repeat for parts')
    power.add_argument('--positions', required=True)
    power.add_argument('--critic', action='append', required=True)
    power.add_argument('--boot', type=int, default=2000)
    power.add_argument('--view', choices=('critic', 'actor'), default='critic',
                       help='the critic items (default) or the actor items of the states')
    power.add_argument('--rows', help='also write the per-position inputs of every critic (JSON)')
    power.add_argument('--out', required=True)
    args = parser.parse_args()
    {'jobs': cmd_jobs, 'score': cmd_score, 'positions': cmd_positions, 'power': cmd_power}[args.command](args)


if __name__ == '__main__':
    main()
