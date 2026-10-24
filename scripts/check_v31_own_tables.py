"""Actual native serving with a grown learner and a historical opponent's own bundle."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from swccg_policy.native import write_bundle
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import load, save
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_native import export_ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=Path('runs/knowledge-v3.1/amendment9-own-table-serving'))
    parser.add_argument('--out', type=Path,
                        default=Path('reports/data/knowledge-v3.1/amendment9-own-table-serving.json'))
    parser.add_argument('--reuse-results', action='store_true',
                        help='Recheck saved native choices without rerunning Java')
    args = parser.parse_args()
    args.run.mkdir(exist_ok=args.reuse_results)
    for name in ('ckpt', 'snaps', 'traj'):
        (args.run / name).mkdir(exist_ok=args.reuse_results)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path('runs/knowledge-v3.1/amendment9-growth-candidate.pt')
    old, _, old_tables, _ = load(source, 'cuda')
    grown_tables = TableBundle.load('runs/knowledge-v3.1/tables-decipher-virtual-extension')
    grown, critic, _, _ = load(source, 'cuda', tables=grown_tables)
    if not args.reuse_results:
        save(args.run / 'grown.pt', grown, critic, grown_tables)
        export_ts(grown, args.run / 'ckpt/latest.ts')
        export_ts(old, args.run / 'old.ts')
        write_bundle(args.run / 'grown.pt', args.run / 'bundle', 'owned')
        write_bundle(source, args.run / 'old.ts.bundle', 'owned')
        (args.run / 'ckpt/latest.ts.json').write_text('{"iteration":0}\n')
    records = [json.loads(line)[:12] for line in
               Path('runs/knowledge-v3.1/amendment9-abi-check/records.jsonl').read_text().splitlines()[:48]]
    new_question = next(key for key in grown_tables.vocabulary.payload['q']
                        if key not in old_tables.vocabulary.payload['q'])
    for record in records[::2]:
        record[5] = new_question
    inputs = args.run / 'records.jsonl'
    encoded = ''.join(json.dumps(record) + '\n' for record in records)
    if args.reuse_results:
        assert inputs.read_text() == encoded
    else:
        inputs.write_text(encoded)
    expected = {}
    with torch.no_grad():
        for name, model, tables in (('old', old, old_tables), ('grown', grown, grown_tables)):
            model.cache_card_table()
            expected[name] = [model(stack([parse(record, tables)], 'cuda')).cpu().numpy()[0] for record in records]
    properties = {'knowledge': 'true', 'schema': 4, 'native.tf32': 'false',
                  'native.run': args.run.resolve(), 'native.bundle': (args.run / 'bundle').resolve(),
                  'native.bc': (args.run / 'ckpt/latest.ts').resolve(),
                  'native.lib': Path('native/build-v3.1/libswccgpolicy.so').resolve()}
    output = args.run / 'choices.json'
    command = ['bash', 'scripts/efficient_java_check.sh',
               *[f'-Dpolicy.{key}={value}' for key, value in properties.items()],
               'com.gempukku.swccgo.bench.OwnTablePolicyCheck', str(inputs),
               str((args.run / 'old.ts').resolve()), str(output)]
    if not args.reuse_results:
        with (args.run / 'engine.log').open('w') as stream:
            subprocess.run(command, env={**os.environ, 'SWCCG_V31_CLASSES': str(args.run / 'classes')},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    actual = json.loads(output.read_text())
    checks = []
    for index, choices in enumerate(actual):
        for name, choice in choices.items():
            logits = expected[name][index]
            ranked = np.argsort(-logits, kind='stable')
            gap = float(logits[ranked[0]] - logits[ranked[1]])
            checks.append({'record': index, 'model': name, 'choice': choice, 'python_choice': int(ranked[0]),
                           'matched': bool(choice == ranked[0] or gap < 1e-6 and choice in ranked[:2])})
    report = {'passed': all(row['matched'] for row in checks), 'checks': checks,
              'old_table_sha': old_tables.sha, 'grown_table_sha': grown_tables.sha,
              'new_question': new_question, 'old_question_id': old_tables.vocabulary.lookup('q', new_question),
              'grown_question_id': grown_tables.vocabulary.lookup('q', new_question),
              'different_owned_bundles_loaded': True, 'command': command}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('checks', 'command')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
