"""Plan titles-and-learner-ablation A1/A3: card-name marks are invisible to everyone but a normalizer-v2 encoder.

`identity`: the same seeded AdvancedAI-vs-AdvancedAI games twice (identity hash pinned), each writing every decision
as its non-encoder readers see it (text, parameters, answer: NameMarkCheck) plus the encoder's v1 record lines. Both
dumps must be byte-identical and carry no marker. While the marks had an on/off switch, this compared off with on
(reports/titles-and-learner-ablation.md A1); the marks are now always on.

`export`: record lines encoded with marks visible (normalizer v2), for tables-v7's keys and the unknown-lookup
comparison; `--unmarked` exports today's (v1) encoding of the same games.

    .venv/bin/python scripts/name_marks_check.py identity --games 10 --out reports/data/titles-and-learner-ablation/identity
    .venv/bin/python scripts/name_marks_check.py export --games 100 --out runs/name-marks/export-100
"""
import argparse
import csv
import filecmp
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / 'data/schedules/rl_pool_decipher.txt'


def run(folder, games, seed, export_marked, threads, pin_hash=False, jvm=()):
    folder.mkdir(parents=True, exist_ok=False)
    # SWCCG_CHECK_CLASSES: a compiled overlay other than check-classes (e.g. an older engine's, with SWCCG_ENGINE_TREE).
    environment = {**os.environ, 'SWCCG_V31_CLASSES': os.environ.get('SWCCG_CHECK_CLASSES', str(ROOT / 'runs/knowledge-v3.1/check-classes'))}
    native = {'knowledge': 'true', 'schema': 4, 'context': 5, 'corpusThreadSeeds': 'true',
              'titleTable': ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv',
              'nameMarksExport': str(export_marked).lower(), 'exportSchema4': 'false',
              'checkNameMarks': folder / 'decisions.tsv'}
    # Identity runs pin Java's identity hash (constant): the engine orders some sets by it, and its per-start
    # variation alone changes which player a response window asks first (measured: two unmarked runs differ too).
    pinned = ['-XX:+UnlockExperimentalVMOptions', '-XX:hashCode=2'] if pin_hash else []
    pinned.append('-Xmx3g')  # beside a local training run (29 GB machine)
    command = ['bash', 'scripts/v31_engine.sh', *pinned, *[f'-Dpolicy.{key}={value}' for key, value in native.items()], *jvm,
               'com.gempukku.swccgo.bench.TrainGen', '--decks', str(ROOT), '--pool', str(POOL),
               '--games', str(games), '--threads', str(threads), '--warmup', '0', '--agent', 'advanced',
               '--opponent', 'advanced', '--agentSide', 'alternate', '--cap', '6000', '--maxGameMs', '300000',
               '--seed', str(seed), '--pairSeed', str(seed), '--tag', 'nm', '--export', str(folder / 'export'),
               '--stackLog', str(folder / 'errors.log'), '--out', str(folder / 'games.csv')]
    (folder / 'command.json').write_text(json.dumps(command, indent=1) + '\n')
    with (folder / 'run.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def exported(folder):
    return sorted(path.relative_to(folder) for path in (folder / 'export').rglob('*') if path.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['identity', 'export'])
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--threads', type=int, default=4, help='export only; identity runs use 1')
    parser.add_argument('--unmarked', action='store_true', help='export only: marks off (the v1 encoding)')
    parser.add_argument('--pin-hash', action='store_true', help='export only: pinned identity hash, for identical games')
    parser.add_argument('--jvm', action='append', default=[], help='export only: extra JVM option (checks)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'export':
        run(args.out, args.games, args.seed, not args.unmarked, args.threads, args.pin_hash, args.jvm)
        print(json.dumps({'folder': str(args.out), 'files': len(exported(args.out))}))
        return
    run(args.out / 'first', args.games, args.seed, False, 1, True)
    run(args.out / 'second', args.games, args.seed, False, 1, True)
    off, on = args.out / 'first', args.out / 'second'
    decisions = filecmp.cmp(off / 'decisions.tsv', on / 'decisions.tsv', shallow=False)
    files = exported(off)
    lines = files == exported(on) and all(filecmp.cmp(off / f, on / f, shallow=False) for f in files)
    # Game rows without wall_ms (timing differs between any two runs).
    rows = lambda folder: [row[:13] + row[14:] for row in csv.reader((folder / 'games.csv').open())]
    games = rows(off) == rows(on)
    marks_seen = sum(line.count('') for line in (on / 'decisions.tsv').read_text().splitlines())
    result = {'games': args.games, 'decisions': len((off / 'decisions.tsv').read_text().splitlines()),
              'decisions_identical': decisions, 'record_lines_identical': lines, 'export_files': len(files),
              'game_rows_identical_except_wall_time': games, 'marks_visible_to_readers': marks_seen}
    (args.out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if decisions and lines and games and marks_seen == 0 else 1)


if __name__ == '__main__':
    main()
