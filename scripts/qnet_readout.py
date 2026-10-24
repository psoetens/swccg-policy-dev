"""Plan q-critic stage 1 readout: does the Q-net tell the options of E2's positions apart?

    qnet_readout.py --qnet runs/q-critic/qnet-s1.pt --checkpoint runs/critic-noise/D.pt --e2 runs/q-critic/e2 \
        --out reports/data/q-critic/stage1.json

For E2's 150 positions (their decision states: E2's position states, the same encodings E2's truth branched from):
A_Q(o) = Q(s, o) - sum_o' pi(o') Q(s, o') over the options with truth (pi as E2 listed it, renormalised), and Pearson
(A_Q, A*) over all (position, option) pairs, disattenuated by the truth's split-half reliability (Spearman-Brown to the
full rollout count). A_Q is deterministic per (position, option), so its per-sample and its bias readout coincide.
Compared, paired over the same positions (bootstrap over positions), with the per-sample signal of turn-credit stage
A's schemes on the same positions (`readout.rows.jsonl`): GAE 0.95 (today), GAE 0.8 (stage A's best), GAE 0 (the
one-step critic). Readout rule (plan §1): "A_Q beats GAE(0.95)" if the paired difference's 95% interval is above 0.
Also: sum pi Q against the critic's V(s) at the positions (a consistency check).
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import critic_probes as cp  # noqa: E402
from advantage_readout import load_states, spearman_brown  # noqa: E402
from turn_credit_readout import truth_table, wpearson  # noqa: E402
from swccg_policy.qnet import load_qnet, option_values  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402

COMPARED = ('gae-0.95', 'gae-0.8', 'gae-0')
RULE_REFERENCE = 'gae-0.95'


def pair_table(truth, spec, q_by_position):
    """Rows (position index, option, A*, h1, h2, A_Q, A_Q centred by pi) for positions with Q values and >= 2 options."""
    rows, positions = [], []
    for entry in spec['list']:
        p = entry['position']
        if p not in truth or p not in q_by_position:
            continue
        t = truth[p]
        options = sorted(t['options'])
        q = q_by_position[p]
        pi = np.array([t['pi'][o] for o in options], float)
        pi = pi / pi.sum() if pi.sum() > 0 else np.full(len(options), 1 / len(options))
        values = np.array([q[o] for o in options], float)
        a_q = values - pi @ values
        index = len(positions)
        positions.append(p)
        for k, o in enumerate(options):
            rows.append({'position': index, 'option': o, 'a_star': t['a_star'][o],
                         'h1': None if t['h1'] is None else t['h1'][o], 'h2': None if t['h2'] is None else t['h2'][o],
                         'a_q': float(a_q[k]), 'q': float(values[k])})
    return positions, rows


def stage_a_samples(rows_file, positions, truth):
    """Per compared scheme: arrays (position index, A*, value) of stage A's per-rollout advantages."""
    index_of = {p: i for i, p in enumerate(positions)}
    out = {name: ([], [], []) for name in COMPARED}
    for line in Path(rows_file).read_text().splitlines():
        row = json.loads(line)
        p, o = row['position'], row['option']
        if p not in index_of or o not in truth[p]['a_star']:
            continue
        for name in COMPARED:
            if row.get(name) is not None:
                out[name][0].append(index_of[p])
                out[name][1].append(truth[p]['a_star'][o])
                out[name][2].append(row[name])
    return {name: tuple(np.array(column) for column in columns) for name, columns in out.items()}


def core(rows, samples, weights):
    position = np.array([r['position'] for r in rows])
    w = weights[position].astype(float)
    split = np.array([r['h1'] is not None for r in rows])
    h1 = np.array([r['h1'] if r['h1'] is not None else 0.0 for r in rows])
    h2 = np.array([r['h2'] if r['h2'] is not None else 0.0 for r in rows])
    half = wpearson(h1[split], h2[split], w[split]) if split.any() else None
    reliability = spearman_brown(half) if half is not None else None
    root = math.sqrt(reliability) if reliability is not None and reliability > 0 else None
    a_star = np.array([r['a_star'] for r in rows])
    out = {'reliability': reliability, 'a_q': wpearson(np.array([r['a_q'] for r in rows]), a_star, w)}
    for name, (sample_position, sample_star, values) in samples.items():
        out[name] = wpearson(values, sample_star, weights[sample_position].astype(float)) if len(values) else None
    def dis(value):
        return None if value is None or root is None else value / root
    for name in ('a_q', *samples):
        out[name + '_dis'] = dis(out[name])
    for name in samples:
        out['a_q_minus_' + name] = None if out['a_q'] is None or out[name] is None else (
            (out['a_q'] - out[name]) / root if root else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qnet', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--e2', type=Path, required=True, help='E2 copy: positions.json, positions/, report-gt*.txt, '
                                                                'readout.rows.jsonl (turn-credit stage A)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261020)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load(args.checkpoint, 'cuda')
    _, critic, tables, _ = base
    qnet, meta = load_qnet(args.qnet, base, 'cuda')
    qnet.packed_attention = False
    spec, truth, _ = truth_table(args.e2, args.e2 / 'positions.json')
    ids = {entry['id']: entry['position'] for entry in spec['list']}
    states = load_states(str(args.e2 / 'positions'), tables, set(ids))
    order = [state_id for state_id in ids if state_id in states]
    items = [states[state_id][0] for state_id in order]
    q_values = option_values(qnet, items, cp.batches, stack)
    q_by_position = {ids[state_id]: values for state_id, values in zip(order, q_values)}
    critic.eval()
    critic.encoder.packed_attention = False
    with torch.no_grad():
        critic_items = [states[state_id][1] for state_id in order]
        v = np.zeros(len(order))
        for selected in cp.batches(critic_items):
            v[selected] = critic(stack([critic_items[i] for i in selected], 'cuda')).float().cpu().numpy()
    positions, rows = pair_table(truth, spec, q_by_position)
    samples = stage_a_samples(args.e2 / 'readout.rows.jsonl', positions, truth)
    count = len(positions)
    main_result = core(rows, samples, np.ones(count, int))
    rng = np.random.default_rng(args.seed)
    collected = collections.defaultdict(list)
    for _ in range(args.boot):
        result = core(rows, samples, np.bincount(rng.integers(0, count, count), minlength=count))
        for key, value in result.items():
            if value is not None:
                collected[key].append(value)
    def interval(key):
        values = collected[key]
        return None if len(values) < args.boot // 2 else [round(float(np.percentile(values, 2.5)), 4),
                                                          round(float(np.percentile(values, 97.5)), 4)]
    readout = {key: {'value': None if value is None else round(float(value), 4), 'ci95': interval(key)}
               for key, value in main_result.items()}
    difference = readout['a_q_minus_' + RULE_REFERENCE]['ci95']
    pi_q = []
    for state_id, values in zip(order, q_values):
        entry = next(e for e in spec['list'] if e['id'] == state_id)
        pi = np.asarray(entry['pi'][:len(values)], float)
        pi_q.append(float(pi / pi.sum() @ values))
    consistency = {'pearson_sum_pi_q_vs_v': round(float(np.corrcoef(pi_q, v)[0, 1]), 4),
                   'mean_sum_pi_q': round(float(np.mean(pi_q)), 4), 'mean_v': round(float(np.mean(v)), 4)}
    out = {'qnet': str(args.qnet), 'qnet_meta': meta, 'positions': count, 'pairs': len(rows), 'boot': args.boot,
           'readout': readout, 'consistency': consistency,
           'a_q_beats_gae095': bool(difference is not None and difference[0] > 0),
           'rule': 'A_Q beats GAE(0.95) if the paired difference of the disattenuated per-sample signals has a 95% '
                   'interval above 0 (plan q-critic §1)'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({key: out[key] for key in ('positions', 'pairs', 'a_q_beats_gae095', 'consistency')}))
    print(json.dumps({key: readout[key] for key in readout}, indent=None))


if __name__ == '__main__':
    main()
