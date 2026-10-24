#!/usr/bin/env python3
"""Compare BillBisco2's retro decklists (data/decks/community-bisco/raw/, a mirror of his Google Drive folder,
and wiki-1996-worlds/, his 1996 Worlds top-4 transcriptions on wiki.swccg.com) with every
deck we hold, and list our Decipher-only decks he does not have (plan: .claude/plans/community-bisco-decks.md).

Card identity is (side, normalised title): reprints (premium 109_6 vs the original, alternate images) count as one card.
Match classes, on the cards inside the deck (shields are compared separately):
  exact    - same multiset as one of our decks
  variant  - at least --variant (default 55) of 60 cards shared with one of our decks of the same side
  new      - neither
Our collection: data/decks/{corpus,v4/corpus,online-decipher}/*.xml, every member of the PC tournament ZIPs
(data/all-tournament-decks-2021-2026/) and of the starwarsccg.org forum attachments (data/raw/pc-modern/), so decks we
marked duplicate or discarded count too. Text lists in those archives are resolved with the corpus resolver.

Writes into data/decks/community-bisco/: parsed/*.xml (his unique decks), compare.csv, share-candidates.csv, formats.csv.
Usage: python3 scripts/compare_community_decks.py [--variant 55]
"""
import argparse
import collections
import csv
import glob
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/decks'))
import ingest_pc_modern as IP  # noqa: E402  (parse_gemp_xml, parse_text_deck, iter_archive, norm_id)

OUT = ROOT / 'data/decks/community-bisco'
CARD_DB = next(Path(p) for p in sorted(glob.glob(str(ROOT / 'data/cards/card_blueprint_database-*.json'))) if 'provenance' not in p)
DECK_EXT = ('.xml', '.txt', '.gemp', '.json')
# His format codes: Premiere through <set>, Decipher cards only (CNV = Classic No Virtual = all of sets 1-14). Premium
# prints (sets 101-112) never raise a deck's latest set: his folders use them in every format.
CEILINGS = [(2, 'PANH'), (3, 'PHOTH'), (9, 'PDS2'), (10, 'PREF2'), (11, 'PTAT'), (14, 'CNV')]


RETRO = re.compile(r'retro|decipher-cards-only|PREF2|Eclipse', re.I)


def norm(text):
    return re.sub(r'[^a-z0-9]', '', text.lower().replace('’', "'").replace('&', 'and'))


class Cards:
    def __init__(self):
        self.by_id = {c['cardId']: c for c in json.loads(CARD_DB.read_text())}
        # earliest main set (1-14) a title appears in, per (side, title); premium-only titles count as set 1
        self.first_set = {}
        for cid, c in self.by_id.items():
            if cid.endswith('_BACK') or not cid.split('_')[0].isdigit():
                continue
            s = int(cid.split('_')[0])
            s = 1 if 100 < s < 200 else s
            if s < 100:
                k = (c.get('side'), norm(c['title']))
                self.first_set[k] = min(self.first_set.get(k, 99), s)

    def key(self, cid):
        c = self.by_id.get(cid)
        return (c.get('side'), norm(c['title'])) if c else ('?', cid)

    def title(self, cid):
        c = self.by_id.get(cid)
        return c['title'] if c else ''


def load(ins, outs, cards):
    ids_in = [IP.norm_id(x) for x in ins if x]
    ids_out = [IP.norm_id(x) for x in outs if x]
    sides = {cards.by_id[i].get('side') for i in ids_in if i in cards.by_id and cards.by_id[i].get('side')}
    return {'in': ids_in, 'out': ids_out,
            'side': sides.pop() if len(sides) == 1 else ('MIXED' if sides else ''),
            'keys': collections.Counter(cards.key(i) for i in ids_in),
            'shield_keys': collections.Counter(cards.key(i) for i in ids_out),
            'decipher_only': all(int(i.split('_')[0]) < 200 for i in ids_in + ids_out)}


def fingerprint(deck):
    return tuple(sorted(deck['keys'].items()))


def parse_member(name, data, cards):
    """(deck, problems) for one archive member, or None if it is not a decklist."""
    try:
        ins, outs = IP.parse_gemp_xml(data)
        return load(ins, outs, cards), []
    except Exception:
        pass
    if not name.lower().endswith('.txt'):
        return None
    text = data.decode('utf-8', errors='replace')
    if '\x00' in text[:200]:
        return None
    ins, outs, problems, _ = IP.parse_text_deck(text)
    return load(ins, outs, cards), problems


def our_collection(cards):
    """List of (deck, meta) for every deck we can rebuild."""
    manifest = {r['deck_id']: r for r in csv.DictReader(open(ROOT / 'data/decks/v4/manifest.csv'))}
    for r in csv.DictReader(open(ROOT / 'data/decks/manifest.csv')):
        manifest.setdefault(r['deck_id'], r)
    decktech = {}
    src = ROOT / 'data/decks/online-decipher/sources.csv'
    for r in csv.DictReader(open(src)):
        decktech[Path(r.get('file') or r.get('xml') or '').stem or r.get('deck_id', '')] = r
    decks = []
    for d in ('data/decks/corpus', 'data/decks/v4/corpus', 'data/decks/online-decipher'):
        for f in sorted((ROOT / d).glob('*.xml')):
            ins, outs = IP.parse_gemp_xml(f.read_bytes())
            m = manifest.get(f.stem, {})
            if not m and f.stem in decktech:
                t = decktech[f.stem]
                m = {'source': 'decktech.net archive', 'event': 'decktech post', 'player': t.get('author', ''),
                     'deck_name': t.get('title', ''), 'date': t.get('date', ''), 'source_path': t.get('source', '').split(' ')[0]}
            decks.append((load(ins, outs, cards), {
                'our_id': f.stem, 'where': f'{d}/{f.name}', 'source': m.get('source', ''), 'event': m.get('event', ''),
                'date': m.get('date', ''), 'player': m.get('player', ''), 'deck_name': m.get('deck_name', f.stem),
                'url': m.get('source_path', ''), 'status': m.get('status', 'passed' if 'corpus' in d else '')}))
    for z in sorted((ROOT / 'data/all-tournament-decks-2021-2026').iterdir()):
        items = IP.iter_archive(z.read_bytes(), z.name) if z.name.lower().endswith(IP.ARCH) else [(z.name, z.read_bytes())]
        for name, data in items:
            if not name.lower().endswith(DECK_EXT):
                continue
            got = parse_member(name, data, cards)
            if got and not got[1]:
                decks.append((got[0], {'our_id': f'pcz:{z.stem}/{Path(name).stem}', 'where': f'{z.relative_to(ROOT)}#{name}',
                                       'source': 'PC tournament ZIPs 2021-2026', 'event': z.stem,
                                       'date': (re.match(r'\d{4}', z.stem) or [''])[0], 'player': '',
                                       'deck_name': Path(name).stem, 'url': '', 'status': 'raw'}))
    raw = ROOT / 'data/raw/pc-modern'
    for line in open(raw / 'manifest.jsonl'):
        m = json.loads(line)
        if m.get('status') != 'ok':
            continue
        try:
            content = (raw / m['file']).read_bytes()
            items = list(IP.iter_archive(content, m['file'])) if m['file'].lower().endswith(IP.ARCH) else [(Path(m['file']).name, content)]
        except Exception:
            continue
        for name, data in items:
            if not name.lower().endswith(DECK_EXT):
                continue
            got = parse_member(name, data, cards)
            if got and not got[1]:
                decks.append((got[0], {'our_id': f'pcm:{m["event"][:48]}/{Path(name).stem}', 'where': f'data/raw/pc-modern/{m["file"]}#{name}',
                                       'source': 'starwarsccg.org forum', 'event': m['event'],
                                       'date': (re.match(r'\d{4}', m['event']) or [''])[0], 'player': '',
                                       'deck_name': Path(name).stem, 'url': m.get('attachment_url', ''), 'status': 'raw'}))
    return decks


def his_meta(path):
    parts = path.split('/')
    fmt = parts[0]
    year = (re.findall(r'(?<!\d)((?:19|20)\d\d)(?!\d)', path) or [''])[-1]
    return {'format': fmt, 'year': year, 'archetype': parts[2] if len(parts) > 3 and parts[1] in ('Dark', 'Light', 'DS', 'LS') else ''}


def overlap(a, b):
    return sum((a & b).values())


def write_xml(deck, cards, path):
    root = ET.Element('deck')
    for cid in deck['in']:
        ET.SubElement(root, 'card', blueprintId=cid, title=cards.title(cid))
    for cid in deck['out']:
        ET.SubElement(root, 'cardOutsideDeck', blueprintId=cid, title=cards.title(cid))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='UTF-8', xml_declaration=True)


def slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', type=int, default=55)
    args = ap.parse_args()
    cards = Cards()

    # --- his files
    sources = [(s['path'], OUT / 'raw' / s['path'], s['url'], his_meta(s['path']))
               for s in csv.DictReader(open(OUT / 'sources.csv'))]
    # his 1996 Worlds top-4 transcriptions on wiki.swccg.com (GEMP files linked from each deck page)
    for page in sorted((OUT / 'wiki-1996-worlds/raw').glob('*_[DL]S.wiki')):
        media = re.search(r'Media:([^|\]]+)', page.read_text()).group(1).strip().replace(' ', '_')
        url = 'https://wiki.swccg.com/wiki/1996_Decipher_World_Championship_' + page.stem
        sources.append((f'wiki-1996-worlds/{media}', OUT / 'wiki-1996-worlds/raw' / media, url,
                        {'format': 'PANH', 'year': '1996', 'archetype': 'Worlds top 4'}))
    his, rows = {}, []
    for path, file, url, meta in sources:
        data = file.read_bytes()
        row = {'path': path, 'source_url': url, **meta}
        got = None
        if path.lower().endswith('.txt') and b'<deck' in data:
            got = parse_member(path, data, cards)
        if got is None:
            kind = ('saved web page assets (no decklist)' if path.endswith('.zip') else
                    'scanned handwritten standings (not a decklist)' if path.endswith('.pdf') else 'standings text (not a decklist)')
            row.update(status=f'unparsed ({kind})')
            rows.append(row)
            continue
        deck, problems = got
        ok_ids = all(i in cards.by_id for i in deck['in'] + deck['out'])
        row.update(side=deck['side'], cards=len(deck['in']), shields=len(deck['out']), decipher_only=deck['decipher_only'],
                   status='parsed' if ok_ids and len(deck['in']) == 60 and deck['side'] in ('DARK', 'LIGHT') else
                   f'parsed (check: {len(deck["in"])} cards, side {deck["side"] or "?"}' +
                   (', unknown ids' if not ok_ids else '') + ')')
        fp = fingerprint(deck)
        if fp in his:
            row['his_duplicate_of'] = his[fp]['path']
        else:
            his[fp] = {'path': path, 'deck': deck, 'row': row}
        rows.append(row)

    # --- our decks, grouped by fingerprint
    ours = collections.OrderedDict()
    for deck, meta in our_collection(cards):
        ours.setdefault(fingerprint(deck), []).append((deck, meta))
    print(f'our decks: {sum(map(len, ours.values()))} files, {len(ours)} distinct card lists', file=sys.stderr)
    by_side = collections.defaultdict(list)
    for fp, group in ours.items():
        by_side[group[0][0]['side']].append((fp, group))

    def label(group):
        m = group[0][1]
        return m['our_id']

    # --- compare his unique decks
    (OUT / 'parsed').mkdir(exist_ok=True)
    matched_ours = set()
    for fp, h in his.items():
        deck, row = h['deck'], h['row']
        write_xml(deck, cards, OUT / 'parsed' / (slug(Path(h['path']).with_suffix('').as_posix()) + '.xml'))
        if len(deck['in']) != 60:
            best = max(((overlap(deck['keys'], g[0][0]['keys']), g) for _, g in by_side[deck['side']]),
                       default=(0, None), key=lambda t: t[0])
            row.update(match='incomplete', overlap=best[0], our_id=label(best[1]) if best[1] else '')
            continue
        # any of our decks within the variant threshold counts as "he has it" for the share list
        for ofp, g in by_side[deck['side']]:
            if overlap(deck['keys'], g[0][0]['keys']) >= args.variant:
                matched_ours.add(ofp)
        if fp in ours:
            group = ours[fp]
            row.update(match='exact', overlap=60, our_id=label(group), our_copies=len(group),
                       shields_same=group[0][0]['shield_keys'] == deck['shield_keys'])
            matched_ours.add(fp)
            continue
        best = max(((overlap(deck['keys'], g[0][0]['keys']), ofp, g) for ofp, g in by_side[deck['side']]),
                   default=(0, None, None), key=lambda t: t[0])
        if best[0] >= args.variant:
            row.update(match='variant', overlap=best[0], our_id=label(best[2]), our_copies=len(best[2]))
        else:
            row.update(match='new', overlap=best[0], our_id=label(best[2]) if best[2] else '')
    for r in rows:
        if r.get('his_duplicate_of'):
            first = next(x for x in rows if x['path'] == r['his_duplicate_of'])
            r.update(match=first.get('match', ''), overlap=first.get('overlap', ''), our_id=first.get('our_id', ''))

    fields = ['path', 'format', 'year', 'archetype', 'side', 'cards', 'shields', 'decipher_only', 'status',
              'his_duplicate_of', 'match', 'overlap', 'our_id', 'our_copies', 'shields_same', 'source_url']
    with open(OUT / 'compare.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r['path']))

    # --- his formats: latest main set and premium sets used, per top folder
    def latest_set(deck):
        sets = [cards.first_set.get(k) for k in deck['keys']]
        return max((s for s in sets if s), default=0)

    fmt = collections.defaultdict(lambda: {'decks': 0, 'max_set': 0, 'sets': collections.Counter()})
    for h in his.values():
        if not h['deck']['decipher_only'] or h['row']['status'] != 'parsed':
            continue
        f_ = fmt[h['row']['format']]
        f_['decks'] += 1
        s = latest_set(h['deck'])
        f_['max_set'] = max(f_['max_set'], s)
        f_['sets'][s] += 1
    with open(OUT / 'formats.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['format', 'unique_decipher_decks', 'latest_main_set_seen', 'decks_by_latest_set'])
        for name, v in sorted(fmt.items(), key=lambda kv: kv[1]['max_set']):
            w.writerow([name, v['decks'], v['max_set'], ' '.join(f'{k}:{n}' for k, n in sorted(v['sets'].items()))])
    ceilings = CEILINGS

    def fits(deck):
        s = latest_set(deck)
        return next((name for ceiling, name in ceilings if s <= ceiling), f'set {s}'), s

    # --- share candidates: our Decipher-only 60-card decks with no exact/variant counterpart in his folder
    share = []
    for fp, group in ours.items():
        deck, meta = group[0]
        if fp in matched_ours or not deck['decipher_only'] or len(deck['in']) != 60 or deck['side'] not in ('DARK', 'LIGHT'):
            continue
        near = [(overlap(deck['keys'], h['deck']['keys']), h) for h in his.values() if h['deck']['side'] == deck['side']]
        best = max((n for n, _ in near), default=0)
        # he holds only an incomplete transcription (his 'v0' files) of a deck we have complete: worth sending
        completes = [h['path'] for n, h in near if len(h['deck']['in']) < 60 and n >= len(h['deck']['in']) - 2]
        if best >= args.variant and not completes:
            continue
        # prefer the copy with the richest metadata
        meta = max((m for _, m in group), key=lambda m: (bool(m['player']), bool(m['url']), m['status'] == 'passed'))
        name, s = fits(deck)
        oid = meta['our_id']
        kind = ('swccgdb.com user deck' if oid.startswith('swdb-') else 'decktech idea' if oid.startswith('decktech_') else
                'web article' if oid.startswith('web-') else 'tournament')
        if kind == 'swccgdb.com user deck':
            meta = {**meta, 'event': ''}  # swccgdb's manifest 'event' holds the deck description
        share.append({'kind': kind, **{k: meta[k] for k in ('our_id', 'source', 'event', 'date', 'player', 'deck_name', 'url', 'where')},
                      'side': deck['side'], 'his_format': name, 'latest_set': s, 'copies': len(group),
                      'closest_his_overlap': best, 'completes_his': ' | '.join(completes)})
    kinds = ['tournament', 'web article', 'decktech idea', 'swccgdb.com user deck']
    share.sort(key=lambda r: (kinds.index(r['kind']), r['his_format'], r['event'], r['side'], r['our_id']))
    with open(OUT / 'share-candidates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'our_id', 'side', 'his_format', 'latest_set', 'event', 'date', 'player', 'deck_name',
                                          'source', 'url', 'where', 'copies', 'closest_his_overlap', 'completes_his'])
        w.writeheader()
        w.writerows(share)

    # our retro events: distinct lists, how many hold Virtual cards, how many he has (exact/variant), how many we can share
    shared_ids = {r['our_id'] for r in share}
    events = collections.defaultdict(lambda: collections.Counter())
    for fp, group in ours.items():
        deck = group[0][0]
        for ev in {m['event'] for _, m in group if RETRO.search(m['event']) and not m['our_id'].startswith('swdb-')}:
            e = events[ev]
            e['lists'] += 1
            e['with_virtual'] += not deck['decipher_only']
            e['he_has'] += fp in matched_ours
            e['share'] += any(m['our_id'] in shared_ids for _, m in group)
    with open(OUT / 'our-retro-events.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['event', 'distinct_lists', 'with_virtual', 'he_has', 'share_candidates', 'other'])
        for ev, e in sorted(events.items()):
            w.writerow([ev, e['lists'], e['with_virtual'], e['he_has'], e['share'],
                        e['lists'] - e['with_virtual'] - e['he_has'] - e['share']])
    summary = {
        'his_files': len(rows), 'his_parsed': sum(r['status'] == 'parsed' for r in rows),
        'his_unique_decks': len(his),
        'his_unique_by_match': dict(collections.Counter(h['row'].get('match') for h in his.values())),
        'our_distinct_lists': len(ours), 'share_candidates': len(share),
        'share_by_kind_and_format': {k: dict(collections.Counter(r['his_format'] for r in share if r['kind'] == k)) for k in kinds},
    }
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
