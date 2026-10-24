#!/usr/bin/env python3
"""Census of every deck we know of (plan objective-deck-coverage, step 0): category x status and Decipher objective x
status. Each distinct deck (same cards) is counted once, in the first category that holds it; text-only lists
(unparsed / todo) cannot be de-duplicated and are counted per list.

Writes OUT/census-by-category.csv and OUT/census-by-objective.csv (default data/decks/objective-coverage/).
Usage: .venv/bin/python scripts/deck_census.py --decktech-archive DIR [--out DIR]
DIR is a clone of github.com/stevetotheizz0/decktech_archives; the wiki mirror comes from fetch_wiki_decks.py.
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_sources as DS  # noqa: E402

STATUSES = ['in corpus', 'parsed, playable', 'parsed, not playable', 'unparsed', 'todo']


def census(records, scans):
    seen = {}
    by_cat = collections.defaultdict(collections.Counter)
    by_obj = collections.defaultdict(collections.Counter)
    copies = collections.Counter()
    order = []
    for r in records:
        if r['category'] not in order:
            order.append(r['category'])
        if r['fp'] is not None:
            if r['fp'] in seen:
                if seen[r['fp']] != r['category']:
                    copies[r['category']] += 1
                continue
            seen[r['fp']] = r['category']
        by_cat[r['category']][r['status']] += 1
        o = by_obj[r['objective']]
        o[r['status']] += 1
        o[f'{r["status"]}|{r["kind"]}'] += 1
        if r['status'] == 'in corpus' and r['deck']['decipher_only']:
            o['in corpus, no V'] += 1
    cat = 'wiki: decklist scans (images/PDF)'
    order.append(cat)
    by_cat[cat]['todo'] += len(scans)
    by_obj['(unknown)']['todo'] += len(scans)
    return order, by_cat, by_obj, copies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decktech-archive', required=True)
    ap.add_argument('--out', type=Path, default=DS.ROOT / 'data/decks/objective-coverage')
    args = ap.parse_args()
    ctx = DS.Context()
    order, by_cat, by_obj, copies = census(DS.records(ctx, args.decktech_archive), DS.scans())
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'census-by-category.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category'] + STATUSES + ['total', 'copies of a deck listed in an earlier category'])
        for c in order:
            w.writerow([c] + [by_cat[c][s] for s in STATUSES] + [sum(by_cat[c][s] for s in STATUSES), copies[c]])
        w.writerow(['TOTAL'] + [sum(by_cat[c][s] for c in order) for s in STATUSES]
                   + [sum(by_cat[c][s] for c in order for s in STATUSES), sum(copies.values())])
    side = {t: s for s, t in ctx.dec_obj}
    objs = [t for _, t in ctx.dec_obj] + ['(Virtual objective)', '(no objective)', '(unknown)']
    with open(args.out / 'census-by-objective.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['objective', 'side', 'in corpus', 'in corpus, no V', 'playable, tournament', 'playable, public',
                    'parsed, not playable', 'unparsed', 'todo', 'total'])
        for o in objs:
            c = by_obj[o]
            w.writerow([o, side.get(o, '')[:1], c['in corpus'], c['in corpus, no V'], c['parsed, playable|tournament'],
                        c['parsed, playable|public'], c['parsed, not playable'], c['unparsed'], c['todo'],
                        sum(c[s] for s in STATUSES)])
    for name in ('census-by-category.csv', 'census-by-objective.csv'):
        print(open(args.out / name).read())


if __name__ == '__main__':
    main()
