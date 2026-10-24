#!/usr/bin/env python3
"""Stage decks for the Decipher objectives (plan objective-deck-coverage, steps 1-3). Writes into OUT
(default data/decks/objective-coverage/); the live corpus and pools are not touched.

Step 1 (as-is): every distinct deck that is complete and playable in today's Gemp and not in the corpus is copied into
  staged/ (Gemp XML re-written from its card ids).
Step 2 (repair): for each Decipher objective still below --target (corpus + staged), candidates with that objective
  whose GEMP file lost cards, or that never parsed, are re-read from their complete source list (wiki deck page,
  decktech post, PC text list) and resolved with the approved aliases and our resolvers. Tournament decks first. A deck
  is kept only if every card has a class in today's Gemp (no substitution). Stops at the target.
  Unresolved names of a deck that misses at most --max-unresolved names get a proposed alias (closest title of that
  side) in aliases-proposed.tsv; such a deck is listed as pending and does not count.
Step 3: corrections-for-wiki-files.csv (+ corrections/*.txt): for each repaired deck from a wiki GEMP file, the cards the
  file lacked and the corrected Gemp XML.

Outputs: staged/*.xml, manifest.csv, pending.csv, aliases-proposed.tsv, corrections-for-wiki-files.csv, corrections/.
Usage: .venv/bin/python scripts/stage_objective_decks.py --decktech-archive DIR [--target 10]
"""
import argparse
import collections
import csv
import difflib
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_sources as DS  # noqa: E402
C = DS.C

SOURCE_PREF = ['wiki: tournament pages', 'PC tournaments: retro events', 'PC tournaments: modern', 'BillBisco2 Drive folder',
               'decktech.net archive', 'wiki: files no page links']
# Common SWCCG nicknames seen in the candidate lists, as *proposed* aliases (Peter reviews them; a deck that needs one is
# pending until then). Every target was checked to be the only Decipher Light Side card with that title.
CURATED = {('LIGHT', n): cid for cid, names in {
    '108_1': ['EPP Han'], '108_3': ['EPP Luke', 'Epp Luke', 'Luke w/ Saber', 'Luke with Saber'],
    '108_2': ['EPP Leia'], '108_4': ['EPP Obi', 'Obi w/ Saber'], '103_2': ['RL in R1'], '103_1': ['GL in G1', 'GLIG1'],
    '111_2': ['Artoo in Red 5', 'r2 in red 5'], '109_2': ['Lando In Falcon', 'Lando in Falcon'], '1_143': ['Falcon'],
    '1_139': ['Yavin 4 War Room', 'Y4 War Room', 'Yavin 4War Room', 'Yavin War Room'], '3_57': ['Hoth War Room', 'HothWar Room'],
    '9_51': ['HFTMF'], '5_68': ['Shocking Info'], '1_124': ['DS Docking Bay', 'DSDocking Bay', 'DS DB', 'D Docking Bay'],
    '1_46': ['D Plans'], '7_12': ['Cmdr. Wedge'], '1_3': ['Biggs'], '1_157': ["Obi's Saber"], '110_1': ['Boussh'],
    '7_126': ['spaceport D Bay'], '7_118': ['DS Detention Corridor'], '2_62': ['DSTrench'], '2_68': ['Y4Messasi Ruins'],
    '1_136': ['Yavin DB'], '6_77': ['Signal'], '2_49': ['Grmitassh'], '8_71': ["Endor CC's Hut"], '102_3': ['Jedi Pack Leia'],
}.items() for n in names}

FIELDS = ['deck_id', 'how', 'kind', 'objective', 'side', 'decipher_only', 'category', 'title', 'url', 'source_id',
          'recovered_cards', 'resolver_matches', 'note']


def slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')[:90]


def write_xml(ctx, ins, outs, path):
    root = ET.Element('deck')
    for cid in ins:
        ET.SubElement(root, 'card', blueprintId=cid, title=ctx.cards.title(cid))
    for cid in outs:
        ET.SubElement(root, 'cardOutsideDeck', blueprintId=cid, title=ctx.cards.title(cid))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='UTF-8', xml_declaration=True)


def propose(ctx, name, side, decipher_source):
    """Closest title of that side with a class in today's Gemp, or None."""
    raw = re.sub(r'\s*\((V|AI)\)\s*$', '', name)
    titles = {}
    for cid, c in ctx.cards.by_id.items():
        if c.get('side') == side and cid in ctx.gemp and not cid.startswith('601_') and \
                (not decipher_source or int(cid.split('_')[0]) < 200):
            titles.setdefault(DS.DT.norm(c['title']), cid)
    best = difflib.get_close_matches(DS.DT.norm(raw), list(titles), n=1, cutoff=0.6)
    return titles[best[0]] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decktech-archive', required=True)
    ap.add_argument('--out', type=Path, default=DS.ROOT / 'data/decks/objective-coverage')
    ap.add_argument('--target', type=int, default=10)
    ap.add_argument('--max-unresolved', type=int, default=2)
    args = ap.parse_args()
    ctx = DS.Context()
    recs = DS.records(ctx, args.decktech_archive)
    for d in ('staged', 'corrections'):
        shutil.rmtree(args.out / d, ignore_errors=True)
        (args.out / d).mkdir(parents=True)

    corpus_obj = collections.Counter()
    for fp, stem in ctx.corpus.items():
        pass
    seen = set(ctx.corpus)
    # corpus count per objective: one per distinct corpus deck
    for r in recs:
        if r['status'] == 'in corpus' and r['fp'] in seen and r['fp'] not in corpus_obj:
            corpus_obj[r['fp']] = r['objective']
    corpus_count = collections.Counter(corpus_obj.values())

    manifest, staged_count, used_ids = [], collections.Counter(), set()

    def stage(r, ins, outs, how, recovered='', matches=''):
        deck = ctx.load_ids(ins, outs)
        deck_id = slug(f'{r["category"].split(":")[0].split()[0]}-{r["title"]}') or slug(r['id'])
        base, n = deck_id, 1
        while deck_id in used_ids:
            n += 1
            deck_id = f'{base}-{n}'
        used_ids.add(deck_id)
        write_xml(ctx, deck['in'], deck['out'], args.out / 'staged' / f'{deck_id}.xml')
        objective = ctx.objective_of_ids(deck['in'] + deck['out'])
        decipher_only = deck['decipher_only'] and (r['decipher_source'] or r['lines'] is None or how == 'repaired')
        manifest.append(dict(deck_id=deck_id, how=how, kind=r['kind'], objective=objective, side=deck['side'],
                             decipher_only=decipher_only, category=r['category'], title=r['title'], url=r['url'],
                             source_id=r['id'], recovered_cards=recovered, resolver_matches=matches, note=r['note']))
        staged_count[objective] += 1
        return deck_id

    # ---- step 1: as-is
    for r in recs:
        if r['fp'] is None or r['fp'] in seen:
            continue
        seen.add(r['fp'])
        if r['status'] == 'parsed, playable':
            stage(r, r['deck']['in'], r['deck']['out'], 'as-is')
    step1 = collections.Counter(staged_count)

    # ---- step 2: repair for objectives under target
    pending, proposals, corrections = [], collections.OrderedDict(), []
    targets = [t for _, t in ctx.dec_obj if corpus_count[t] + staged_count[t] < args.target]
    attempts = collections.Counter()
    pending_count = collections.Counter()
    for objective in targets:
        cands = [r for r in recs if r['objective'] == objective and r['lines'] and r['side'] in ('DARK', 'LIGHT')
                 and r['status'] in ('parsed, not playable', 'unparsed', 'todo')]
        cands.sort(key=lambda r: (r['kind'] != 'tournament',
                                  SOURCE_PREF.index(r['category']) if r['category'] in SOURCE_PREF else len(SOURCE_PREF)))
        # pass 1 keeps only decks that resolve completely; pass 2 adds decks pending alias review
        for r in cands + [dict(c, _pass=2) for c in cands]:
            if corpus_count[objective] + staged_count[objective] + pending_count[objective] >= args.target:
                break
            second = r.get('_pass') == 2
            attempts[objective] += not second
            ins, outs, problems, matches = ctx.resolve_lines(r['lines'], r['side'], r['decipher_source'])
            if second and not problems:
                continue
            if problems and not second:
                continue
            if problems:
                # pending: every unresolved name has a proposal (curated nickname, or the closest title for at most
                # --max-unresolved names) and the deck with those proposals is complete and playable
                names, guesses = [], 0
                for p in problems:
                    m = re.match(r'(\d+)x (.*) \[(.*)\]$', p)
                    if not m or m.group(3) == 'original virtual card':
                        names = None
                        break
                    count, name = int(m.group(1)), m.group(2)
                    cid = CURATED.get((r['side'], name))
                    if cid is None:
                        guesses += 1
                        cid = propose(ctx, name, r['side'], r['decipher_source'])
                    if cid is None or guesses > args.max_unresolved:
                        names = None
                        break
                    names.append((name, cid, count))
                if names:
                    trial_in, trial_out = list(ins), list(outs)
                    for _, cid, count in names:
                        shield = ctx.cards.by_id.get(cid, {}).get('cardCategory') == 'DEFENSIVE_SHIELD'
                        (trial_out if shield else trial_in).extend([cid] * count)
                    trial = ctx.load_ids(trial_in, trial_out)
                    if ctx.playable(trial) and C.fingerprint(trial) not in seen and \
                            ctx.objective_of_ids(trial_in + trial_out) == objective:
                        seen.add(C.fingerprint(trial))
                        for name, cid, _ in names:
                            key = (DS.DT.SIDES[r['side']], name)
                            v = proposals.setdefault(key, {'card_id': cid, 'decks': [],
                                                           'how': 'curated nickname' if (r['side'], name) in CURATED else 'closest title'})
                            v['decks'].append(r['id'])
                        pending.append(dict(objective=objective, kind=r['kind'], title=r['title'], url=r['url'],
                                            source_id=r['id'], needs='; '.join(f'{n} -> {ctx.cards.title(c)} ({c})' for n, c, _ in names)))
                        pending_count[objective] += 1
                continue
            deck = ctx.load_ids(ins, outs)
            fp = C.fingerprint(deck)
            if not ctx.playable(deck) or fp in seen or ctx.objective_of_ids(ins + outs) != objective:
                continue
            seen.add(fp)
            recovered = ''
            if r['deck'] is not None:
                diff = collections.Counter(deck['keys']) - collections.Counter(r['deck']['keys'])
                recovered = '; '.join(f'{n}x {t}' for (s, t), n in sorted(diff.items()))
            deck_id = stage(r, ins, outs, 'repaired', recovered, ' | '.join(matches))
            if r['id'].startswith('wiki:') and r['deck'] is not None:
                shutil.copy(args.out / 'staged' / f'{deck_id}.xml', args.out / 'corrections' / f'{deck_id}.txt')
                corrections.append(dict(wiki_file=r['id'][5:], page=r['url'], objective=objective, missing_cards=recovered,
                                        corrected_file=f'corrections/{deck_id}.txt'))

    with open(args.out / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(manifest)
    with open(args.out / 'pending.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['objective', 'kind', 'title', 'url', 'source_id', 'needs'])
        w.writeheader()
        w.writerows(pending)
    with open(args.out / 'aliases-proposed.tsv', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['kind', 'side', 'name', 'card_id', 'title', 'status', 'post', 'note'])
        for (side, name), v in proposals.items():
            w.writerow(['nickname', side, name, v['card_id'], ctx.cards.title(v['card_id']), 'proposed', '',
                        f'{v["how"]}; needed by {len(v["decks"])} deck(s): ' + ', '.join(v['decks'][:3])])
    with open(args.out / 'corrections-for-wiki-files.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['wiki_file', 'page', 'objective', 'missing_cards', 'corrected_file'])
        w.writeheader()
        w.writerows(corrections)

    print(f'staged: {len(manifest)} (as-is {sum(1 for m in manifest if m["how"] == "as-is")}, '
          f'repaired {sum(1 for m in manifest if m["how"] == "repaired")}); pending alias review: {len(pending)}; '
          f'proposed aliases: {len(proposals)}; corrections: {len(corrections)}')
    print('objectives below target after step 1:', ', '.join(f'{t} ({corpus_count[t] + step1[t]})' for t in targets))
    for t in targets:
        print(f'  {t}: corpus {corpus_count[t]}, as-is {step1[t]}, repaired {staged_count[t] - step1[t]}, '
              f'pending alias review {pending_count[t]}, candidates tried {attempts[t]}')


if __name__ == '__main__':
    main()
