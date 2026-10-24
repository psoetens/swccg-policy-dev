"""Fixed held-out, paired-seat native evaluation with each model's owned ABI-4 tables."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import churn_scan
import dojo_coverage
from game_length_metrics import summarize
from swccg_policy.native import export_ts, write_bundle
from swccg_policy.rl import wilson
from swccg_policy.v4_checkpoint import load


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


UNFINISHED = ('cap', 'timeout', 'loop')


def adjudicated(rows, agent):
    """Plan decision-cap-fixes F3, for reporting only: an unfinished game (cap, timeout, loop) is scored by the
    life-force difference at its end (higher wins, equal counts 1/2), as a tournament time limit would. The official
    win rates stay as they are."""
    won = lost = tied = unknown = 0
    for row in rows:
        if row['outcome'] not in UNFINISHED:
            continue
        try:
            stats = json.loads(row.get('stats') or '')
            seat = 'ds' if row['ds_ai'] == agent else 'ls'
            difference = stats[seat]['life_force'] - stats['ls' if seat == 'ds' else 'ds']['life_force']
        except (ValueError, KeyError, TypeError):
            unknown += 1
            continue
        won, lost, tied = won + (difference > 0), lost + (difference < 0), tied + (difference == 0)
    wins = sum(row['outcome'] == 'finished' and row.get(row['winner'] + '_ai') == agent for row in rows)
    score = wins + won + tied / 2
    return {'unfinished': won + lost + tied + unknown, 'unfinished_won': won, 'unfinished_lost': lost,
            'unfinished_tied': tied, 'unfinished_unknown': unknown,
            'win_rate': score / len(rows) if rows else None,
            'ci95': wilson(round(score), len(rows)) if rows else None}


def statistics(rows, agent):
    wins = sum(row['outcome'] == 'finished' and row.get(row['winner'] + '_ai') == agent for row in rows)
    finished = sum(row['outcome'] == 'finished' for row in rows)
    return {'games': len(rows), 'wins': wins, 'finished': finished,
            'win_rate_all_games': wins / len(rows) if rows else None,
            'ci95_all_games': wilson(wins, len(rows)) if rows else None,
            'win_rate_finished': wins / finished if finished else None,
            'adjudicated': adjudicated(rows, agent),
            'game_length': summarize(rows, agent)}


def prepare_model(checkpoint, path):
    import torch

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actor, critic, tables, _ = load(checkpoint, 'cuda')
    export_ts(actor, path)
    write_bundle(checkpoint, Path(str(path) + '.bundle'), 'owned')
    identity = tables.sha
    del actor, critic, tables
    torch.cuda.empty_cache()
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=919001)
    # Plan nvf-b-local A4 (Peter, 2026-10-15): tests play the highest score. 0 = greedy (argmax); 1 = the sampled policy
    # of every evaluation before 2026-10-15. Training rollouts always sample at 1.
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--opponents', default='random,advanced,initial',
                        help='Comma-separated subset of random, advanced, rando, initial (the reference checkpoint); '
                             'plan rando-sweep: rando = GEMP\'s RandoCalAi')
    parser.add_argument('--schedule', type=Path,
                        help='Use this DS<TAB>LS schedule verbatim (one row per game) instead of the paired held-out one')
    parser.add_argument('--agent-side', choices=['alternate', 'ds', 'ls'], default='alternate')
    # Plan v-cards (evaluation only): the title table for name normalization, where unknown keys go, and skipping the
    # Dojo coverage gate (virtual objectives have no Dojo rewards yet, BL-58). Defaults: unchanged.
    parser.add_argument('--title-table', type=Path, default=ROOT / 'runs/knowledge-v3.1/decipher/titles.tsv')
    parser.add_argument('--unknown-keys-dir', type=Path, default=ROOT / 'runs/unknown-keys')
    parser.add_argument('--no-dojo-coverage', action='store_true')
    # Plan v-cards-fixes: Decipher-only pairings play "Decipher Cards Only" (TrainGen default); this keeps every game "open".
    parser.add_argument('--format-open', action='store_true')
    # Plan permanent-play-rules: the activation, cancel and free-move rules are engine rules for every game.
    parser.add_argument('--flip-log', action='store_true', help='Write the engine message log of games with an earned objective flip')
    parser.add_argument('--property', action='append', default=[], metavar='KEY=VALUE',
                        help='Plan critic-counterfactuals: an extra -DKEY=VALUE for the engine JVM (repeatable)')
    parser.add_argument('--game-log', action='store_true', help='Plan dojo-sweep: write the engine message log of every game')
    parser.add_argument('--export', action='store_true',
                        help='Plan policy-diagnostics D3: also export every decision (both seats, the choice taken) '
                             'as record lines to <folder>/<opponent>-export')
    parser.add_argument('--stats-check', action='store_true',
                        help='Diagnostic: cross-check GameStats against the engine chat messages (stats are always on)')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or not args.tag.replace('-', '').isalnum():
        raise ValueError('An even game count and alphanumeric/hyphen tag are required')
    folder = ROOT / 'runs/knowledge-v3.1/evaluations' / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('ckpt', 'snaps', 'traj'):
        (folder / name).mkdir()
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'SWCCG_GRAPH_T': '1536', 'SWCCG_V31_CLASSES': str(folder / 'classes'),
                   'CUDA_MPS_PIPE_DIRECTORY': str(folder / 'no-mps')}
    subprocess.run(['bash', 'scripts/efficient_java_check.sh', 'CorpusRandomCheck'],
                   env=environment, check=True, stdout=subprocess.DEVNULL)
    table_sha = prepare_model(args.checkpoint, folder / 'agent.ts')
    reference_sha = prepare_model(args.reference, folder / 'reference.ts')
    if args.schedule:
        schedule = [line for line in args.schedule.read_text().splitlines() if line and not line.startswith('#')]
        if len(schedule) != args.games:
            raise ValueError(f'--schedule has {len(schedule)} rows, --games is {args.games}')
    else:
        source = ROOT / 'data/schedules/eval_decipher_heldout.tsv'
        pairings = [line for line in source.read_text().splitlines() if line and not line.startswith('#')]
        schedule = [pairings[index % len(pairings)] for index in range(args.games // 2) for _ in range(2)]
    (folder / 'schedule.tsv').write_text('\n'.join(schedule) + '\n')
    if not args.no_dojo_coverage:
        dojo_coverage.require([folder / 'schedule.tsv'])  # plan objective-dojo A2 (fatal)
    agent = 'policy:ck:' + str(folder / 'agent.ts')
    reference = 'policy:ck:' + str(folder / 'reference.ts')
    report = {'checkpoint': str(args.checkpoint), 'reference': str(args.reference), 'seed': args.seed,
              'threads': args.threads, 'table_sha': table_sha, 'reference_table_sha': reference_sha,
              'sampling': ('Native greedy policy (highest score; exact ties at random)' if args.temperature <= 0
                           else f'Native sampled policy at temperature {args.temperature:g}')
                          + '; fixed thread-local seeds and paired seats', 'temperature': args.temperature,
              'censoring': 'Unfinished games count in all-game denominators and are reported separately',
              'schedule_sha256': hashlib.sha256((folder / 'schedule.tsv').read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'complete': False, 'results': {}}
    wanted = args.opponents.split(',')
    if not wanted or not set(wanted) <= {'random', 'advanced', 'rando', 'initial'}:
        raise ValueError('--opponents must be a subset of random, advanced, rando, initial')
    for label, opponent in (('random', 'random'), ('advanced', 'advanced'), ('rando', 'rando'), ('initial', reference)):
        if label not in wanted:
            continue
        # Plan context-overhaul: each model's tensor builder keeps its own context version's rows; models of two
        # versions in one game need both.
        versions = {json.loads((folder / f'{name}.ts.bundle/vocab.json').read_text()).get('context_version', 4)
                    for name in ('agent', 'reference')}
        properties = {'knowledge': 'true', 'schema': 4, 'context': versions.pop() if len(versions) == 1 else 'both',
                      'corpusThreadSeeds': 'true',
                      'titleTable': args.title_table.resolve(),
                      'unknownKeysDir': args.unknown_keys_dir.resolve(),  # plan actor-v6-inputs: unknown texts
                      'native.run': folder, 'native.bundle': folder / 'agent.ts.bundle',
                      'native.bc': folder / 'reference.ts',
                      'native.lib': ROOT / 'native/build-v3.1/libswccgpolicy.so',
                      'native.tf32': 'false', 'native.seed': args.seed, 'native.waitMs': .5,
                      'native.temperature': args.temperature}
        command = ['bash', 'scripts/v31_engine.sh',
                   *[f'-Dpolicy.{key}={value}' for key, value in properties.items()],
                   *(['-Dtraingen.statsCheck=true'] if args.stats_check else []),
               *([f'-Dtraingen.flipLog={folder / "flip-logs"}'] if args.flip_log else []),
               *([f'-Dtraingen.gameLog={folder / "game-logs"}'] if args.game_log else []),
                   *[f'-D{item}' for item in args.property],
                   *(['-Dtraingen.dojoCoverage=false'] if args.no_dojo_coverage else []),
                   *(['-Dtraingen.formatByDecks=false'] if args.format_open else []),
                   'com.gempukku.swccgo.bench.TrainGen', '--decks', str(ROOT), '--schedule', str(folder / 'schedule.tsv'),
                   '--games', str(args.games), '--threads', str(args.threads), '--warmup', '0',
                   '--seed', str(args.seed), '--agent', agent, '--opponent', opponent,
                   '--agentSide', args.agent_side, '--cap', '6000', '--maxGameMs', '120000',
                   '--tag', args.tag + '-' + label, '--stackLog', str(folder / f'{label}-errors.log'),
                   '--out', str(folder / f'{label}.csv'),
                   *(['--export', str(folder / f'{label}-export')] if args.export else [])]
        write(folder / f'{label}-command.json', command)
        with (folder / f'{label}.log').open('w') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (folder / f'{label}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        # Plan rando-sweep: Rando_Cal sometimes answers invalidly 20+ times in a row (intermittent, its own bug); with that
        # opponent such a game counts as unfinished instead of failing the run. Every other opponent: unchanged.
        tolerated = (*UNFINISHED, 'invalid_loop') if label == 'rando' else UNFINISHED
        if len(rows) != args.games or any(row['outcome'] not in ('finished', *tolerated) for row in rows):
            raise RuntimeError('Incomplete evaluation or game exception; evidence retained')
        if 'native batch failed' in (folder / f'{label}.log').read_text():
            raise RuntimeError('Native inference failure; evidence retained')
        seats = defaultdict(list)
        pairs = defaultdict(list)
        for row in rows:
            seats['ds' if row['ds_ai'] == agent else 'ls'].append(row)
            pairs[(row['ds_deck'], row['ls_deck'])].append(row)
        result = {**statistics(rows, agent),
                  'by_seat': {seat: statistics(group, agent) for seat, group in seats.items()},
                  'by_deck_pair': [{'ds_deck': pair[0], 'ls_deck': pair[1], **statistics(group, agent)}
                                   for pair, group in pairs.items()]}
        report['results'][label] = result
        write(folder / 'results.json', report)
        print(json.dumps({'opponent': label, **statistics(rows, agent)}), flush=True)
    # Plan churn-and-drain-cancel: the back-and-forth metric of every run with message logs (scripts/churn_scan.py).
    if args.game_log:
        csvs = [folder / f'{label}.csv' for label in report['results']]
        write(folder / 'churn.json', churn_scan.scan(folder / 'game-logs', churn_scan.policy_sides(csvs)))
    report['complete'] = True
    write(folder / 'results.json', report)


if __name__ == '__main__':
    main()
