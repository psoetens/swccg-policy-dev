#!/usr/bin/env python3
"""Check data/decks/objective-coverage/ (plan objective-deck-coverage, section 1) and print the coverage table.

Every staged deck: 60 cards and one side; every card has a class in today's Gemp and none is a Legacy 601 card; its
manifest row has url, kind, objective (matching the deck) and how (as-is / repaired, with the recovered cards for a
repaired deck from a GEMP file); different cards from every corpus deck and every other staged deck.
Usage: .venv/bin/python scripts/check_objective_coverage.py [--target 10]
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_sources as DS  # noqa: E402
C = DS.C

D = DS.ROOT / 'data/decks/objective-coverage'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=int, default=10)
    args = ap.parse_args()
    ctx = DS.Context()
    manifest = list(csv.DictReader(open(D / 'manifest.csv')))
    files = sorted((D / 'staged').glob('*.xml'))
    fails = collections.defaultdict(list)
    fps = {}
    for m in manifest:
        f = D / 'staged' / f'{m["deck_id"]}.xml'
        if not f.exists():
            fails['manifest row has no file'].append(m['deck_id'])
            continue
        deck = ctx.load_ids(*DS.IP.parse_gemp_xml(f.read_bytes()))
        if len(deck['in']) != 60 or deck['side'] not in ('DARK', 'LIGHT'):
            fails['not 60 cards / one side'].append(m['deck_id'])
        if not all(i in ctx.gemp and not i.startswith('601_') for i in deck['in'] + deck['out']):
            fails['card without a Gemp class, or Legacy'].append(m['deck_id'])
        if ctx.objective_of_ids(deck['in'] + deck['out']) != m['objective']:
            fails['objective differs from manifest'].append(m['deck_id'])
        if not (m['url'] and m['kind'] in ('tournament', 'public') and m['how'] in ('as-is', 'repaired')):
            fails['manifest row incomplete'].append(m['deck_id'])
        fp = C.fingerprint(deck)
        if fp in ctx.corpus:
            fails['same cards as a corpus deck'].append(m['deck_id'])
        if fp in fps:
            fails['same cards as another staged deck'].append(f'{m["deck_id"]} = {fps[fp]}')
        fps[fp] = m['deck_id']
    if len(files) != len(manifest):
        fails['staged files != manifest rows'].append(f'{len(files)} files, {len(manifest)} rows')

    # coverage: distinct corpus decks per objective + staged decks per objective and kind
    corpus = collections.Counter()
    for f in sorted(set((DS.ROOT / 'data/decks/corpus').glob('*.xml')) | set((DS.ROOT / 'data/decks/v4/corpus').glob('*.xml'))):
        pass
    seen = set()
    for f in sorted(list((DS.ROOT / 'data/decks/corpus').glob('*.xml')) + list((DS.ROOT / 'data/decks/v4/corpus').glob('*.xml'))):
        deck = ctx.load_ids(*DS.IP.parse_gemp_xml(f.read_bytes()))
        fp = C.fingerprint(deck)
        if fp not in seen:
            seen.add(fp)
            corpus[ctx.objective_of_ids(deck['in'] + deck['out'])] += 1
    staged = collections.Counter((m['objective'], m['kind']) for m in manifest)
    repaired = collections.Counter(m['objective'] for m in manifest if m['how'] == 'repaired')
    print(f'| Objective | Side | Corpus | Staged tournament | Staged public | (of which repaired) | Total | >= {args.target} |')
    print('|---|---|--:|--:|--:|--:|--:|---|')
    short = []
    for side, t in ctx.dec_obj:
        total = corpus[t] + staged[(t, 'tournament')] + staged[(t, 'public')]
        if total < args.target:
            short.append(t)
        print(f'| {t} | {side[0]} | {corpus[t]} | {staged[(t, "tournament")]} | {staged[(t, "public")]} | {repaired[t]} | '
              f'{total} | {"yes" if total >= args.target else "**no**"} |')
    print()
    print(f'staged decks: {len(manifest)} (as-is {sum(m["how"] == "as-is" for m in manifest)}, '
          f'repaired {sum(m["how"] == "repaired" for m in manifest)}); tournament {sum(m["kind"] == "tournament" for m in manifest)}, '
          f'public {sum(m["kind"] == "public" for m in manifest)}')
    print('staged by source:')
    for (cat, kind), n in sorted(collections.Counter((m['category'], m['kind']) for m in manifest).items(), key=lambda kv: -kv[1]):
        print(f'  {n:4}  {cat} ({kind})')
    print('staged by objective class:', dict(collections.Counter(
        'Decipher objective' if not m['objective'].startswith('(') else m['objective'] for m in manifest)))
    print(f'objectives below {args.target}: {len(short)}' + (': ' + ', '.join(short) if short else ''))
    checks = ['manifest row has no file', 'not 60 cards / one side', 'card without a Gemp class, or Legacy',
              'objective differs from manifest', 'manifest row incomplete', 'same cards as a corpus deck',
              'same cards as another staged deck', 'staged files != manifest rows']
    for c in checks:
        print(f'CHECK {"FAIL" if fails[c] else "ok  "} {c}' + (f': {fails[c][:5]}' if fails[c] else ''))
    if any(fails.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
