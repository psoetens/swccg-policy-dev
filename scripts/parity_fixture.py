"""Plan remove-old-generations, end-goals 2 and 3: the outputs of a V3.1 and a V1 checkpoint, saved before the V2/V3
code is removed and checked after it (exact equality, CPU, one thread).

    parity_fixture.py save  --out DIR [inputs]
    parity_fixture.py check --out DIR            (reads the inputs recorded in DIR/inputs.json)
    parity_fixture.py save-cuda  --out DIR --code-root MAIN   (amendment 1: V3.1 on CUDA with the learner's TF32
    parity_fixture.py check-cuda --out DIR                     setting; the reference comes from main's tree)

V3.1: actor logits and critic values of --v31-checkpoint on --v31-items items spread over --v31-trajectories (SWTRJ003
files, read with the checkpoint's own tables), plus its state_dict keys / shapes and cfg.
V1: logits of --v1-checkpoint on the first --v1-items records of --v1-shard, packed with the checkpoint's template
vocabulary, plus SHA-256 of the packed arrays (parser / packer parity).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def v31(checkpoint, trajectories, count, device='cpu'):
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_trajectory import load_binary
    actor, critic, tables, _ = load(checkpoint, device)
    actor.eval()
    critic.eval()
    games = [game for path in trajectories for game in load_binary(path, tables)]
    rows = [(g, i) for g, game in enumerate(games) for i in range(len(game['items']))]
    chosen = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    logits, values = [], []
    with torch.no_grad():
        for k in chosen:
            g, i = rows[k]
            batch = stack([games[g]['items'][i]], device)
            logits.append(actor(batch).float().masked_fill(~batch['omask'], 0).cpu().numpy()[0])
            values.append(float(critic(stack([games[g]['critic_items'][i]], device)).float().cpu().numpy().ravel()[0]))
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    shapes = {name: list(t.shape) for name, t in ck['state'].items()}
    return {'logits': logits, 'values': np.asarray(values), 'shapes': shapes, 'cfg': ck['cfg'],
            'rows': [list(map(int, rows[k])) for k in chosen]}


def v1(checkpoint, shard, count):
    from swccg_policy.data import pack
    from swccg_policy.train import Packed, load_checkpoint
    model, vocab = load_checkpoint(checkpoint, 'cpu')
    model.eval()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    work = Path(tempfile.mkdtemp(prefix='parity-v1-'))
    try:
        split = work / 'split'
        split.mkdir()
        (split / Path(shard).name).symlink_to(Path(shard).resolve())
        (work / 'vocab.json').write_text(json.dumps({'q': vocab['q'], 'o': vocab['o']}))
        pack(split, work / 'packed', Path(ck['cards']), work / 'vocab.json', 1, 1)
        packed = Packed(work / 'packed')
        hashes = {name: hashlib.sha256(np.ascontiguousarray(getattr(packed, name)).tobytes()).hexdigest()
                  for name in ('rec', 'tok', 'tokf', 'opt', 'chosen')}
        idx = np.arange(min(count, packed.n))
        with torch.no_grad():
            b = packed.batch(idx, 'cpu')
            logits = model(b).float()
            logits = logits.masked_fill(~b['omask'], 0).numpy()
        return {'logits': logits, 'hashes': hashes, 'records': int(packed.n)}
    finally:
        shutil.rmtree(work)


def compute(inputs):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return (v31(inputs['v31_checkpoint'], inputs['v31_trajectories'], inputs['v31_items']),
            v1(inputs['v1_checkpoint'], inputs['v1_shard'], inputs['v1_items']))


def compute_cuda(inputs):
    import swccg_policy.rl  # noqa: F401  the learner's CUDA precision (TF32), set on import in both trees
    torch.use_deterministic_algorithms(True)  # plain CUDA repeats differ by ~1e-3 (atomics, cuBLAS split-K)
    torch.manual_seed(0)
    return v31(inputs['v31_checkpoint'], inputs['v31_trajectories'], inputs['v31_items'], 'cuda')


def check_cuda(out, a):
    ref = np.load(out / 'v31-cuda.npz')
    logit_diff = max(float(np.abs(x - ref[f'logits_{k}']).max()) if x.shape == ref[f'logits_{k}'].shape else float('inf')
                     for k, x in enumerate(a['logits']))
    report = {'items': len(a['logits']), 'tf32': torch.backends.cuda.matmul.allow_tf32,
              'max_abs_logit_diff': logit_diff, 'max_abs_value_diff': float(np.abs(a['values'] - ref['values']).max())}
    report['pass'] = logit_diff == 0 and bool(np.array_equal(a['values'], ref['values']))
    (out / 'check-cuda.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report['pass']


def save(out, a, b):
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'v31.npz', values=a['values'], **{f'logits_{k}': x for k, x in enumerate(a['logits'])})
    np.savez(out / 'v1.npz', logits=b['logits'])
    (out / 'v31-meta.json').write_text(json.dumps({'shapes': a['shapes'], 'cfg': a['cfg'], 'rows': a['rows']}, indent=1))
    (out / 'v1-meta.json').write_text(json.dumps({'hashes': b['hashes'], 'records': b['records']}, indent=1))


def check(out, a, b):
    ref31 = np.load(out / 'v31.npz')
    ref1 = np.load(out / 'v1.npz')
    meta31 = json.loads((out / 'v31-meta.json').read_text())
    meta1 = json.loads((out / 'v1-meta.json').read_text())
    logit_diff = max(float(np.abs(x - ref31[f'logits_{k}']).max()) if x.shape == ref31[f'logits_{k}'].shape else float('inf')
                     for k, x in enumerate(a['logits']))
    report = {
        'v31': {'items': len(a['logits']), 'max_abs_logit_diff': logit_diff,
                'max_abs_value_diff': float(np.abs(a['values'] - ref31['values']).max()),
                'identical': logit_diff == 0 and bool(np.array_equal(a['values'], ref31['values'])),
                'shapes_equal': a['shapes'] == meta31['shapes'], 'cfg_equal': a['cfg'] == meta31['cfg'],
                'rows_equal': a['rows'] == meta31['rows']},
        'v1': {'records': b['records'], 'max_abs_logit_diff': float(np.abs(b['logits'] - ref1['logits']).max())
               if b['logits'].shape == ref1['logits'].shape else float('inf'),
               'identical': bool(np.array_equal(b['logits'], ref1['logits'])),
               'packed_hashes_equal': b['hashes'] == meta1['hashes']},
    }
    report['pass'] = all(report['v31'][k] for k in ('identical', 'shapes_equal', 'cfg_equal', 'rows_equal')) \
        and report['v1']['identical'] and report['v1']['packed_hashes_equal']
    (out / 'check.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report['pass']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['save', 'check', 'save-cuda', 'check-cuda'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, default=ROOT, help='Tree whose src / scripts are imported')
    ap.add_argument('--v31-checkpoint')
    ap.add_argument('--v31-trajectories', nargs='*')
    ap.add_argument('--v31-items', type=int, default=256)
    ap.add_argument('--v1-checkpoint')
    ap.add_argument('--v1-shard')
    ap.add_argument('--v1-items', type=int, default=256)
    args = ap.parse_args()
    sys.path[:0] = [str(args.code_root / 'src'), str(args.code_root / 'scripts')]
    if args.mode == 'save-cuda':
        a = compute_cuda(json.loads((args.out / 'inputs.json').read_text()))
        np.savez(args.out / 'v31-cuda.npz', values=a['values'], **{f'logits_{k}': x for k, x in enumerate(a['logits'])})
        print(f"saved: {len(a['logits'])} V3.1 CUDA items from {args.code_root}")
    elif args.mode == 'check-cuda':
        sys.exit(0 if check_cuda(args.out, compute_cuda(json.loads((args.out / 'inputs.json').read_text()))) else 1)
    elif args.mode == 'save':
        inputs = {'v31_checkpoint': args.v31_checkpoint, 'v31_trajectories': args.v31_trajectories,
                  'v31_items': args.v31_items, 'v1_checkpoint': args.v1_checkpoint, 'v1_shard': args.v1_shard,
                  'v1_items': args.v1_items}
        a, b = compute(inputs)
        save(args.out, a, b)
        (args.out / 'inputs.json').write_text(json.dumps(inputs, indent=1))
        print(f"saved: {len(a['logits'])} V3.1 items, V1 {b['logits'].shape} of {b['records']} records")
    else:
        inputs = json.loads((args.out / 'inputs.json').read_text())
        a, b = compute(inputs)
        sys.exit(0 if check(args.out, a, b) else 1)


if __name__ == '__main__':
    main()
