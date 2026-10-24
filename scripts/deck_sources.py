"""Every deck we know of, as one list of records (plan objective-deck-coverage). Used by deck_census.py and
stage_objective_decks.py.

A record is a dict:
  id, category, kind ('tournament' / 'public'), title, url, side ('DARK'/'LIGHT'/''), objective, status, deck, fp, lines,
  decipher_source, note
- deck: compare_community_decks.load() of the card ids we have (None for text-only records); fp its fingerprint.
- lines: [(count, name, section)] of the complete source list, when we have one (wiki deck page, decktech post,
  PC text list); used to repair a deck whose GEMP file lost cards.
- decipher_source: the source predates Virtual cards (resolve names to Decipher cards only).
- status: 'in corpus' / 'parsed, playable' / 'parsed, not playable' / 'unparsed' / 'todo'.
Card identity follows compare_community_decks: (side, normalised title).
"""
import collections
import csv
import glob
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_community_decks as C  # noqa: E402
import import_decktech_decks as DT  # noqa: E402

IP = C.IP
WIKI = ROOT / 'data/decks/objective-coverage/raw/wiki'
BISCO = ROOT / 'data/decks/community-bisco'
WIKI_URL = 'https://wiki.swccg.com/wiki/'
DECKTECH_URL = 'https://www.stephenskilton.com/decktech_archives/'
EVENT_WORDS = re.compile(r'regional|champion|world|national|state|grand slam|gencon|gen con|origins|open\b|tourn|'
                         r'tourney|winner|1st place|top ?8|top ?4|finals?\b|dpc|major|invitational|qualifier', re.I)
# Section headers DT.SECTION does not know: "Ships", "Creatures", "Systems", "Rebels", "Unknown Type",
# "Locations(3 more)", "Characters(18 + starting Prisoner 2187)", "Vehicles/Starships 8".
EXTRA_SECTION = re.compile(r"^(ships|starships|creatures?|systems?|rebels|imperials|aliens|droids|unknown type|"
                           r"locations?|sites?|characters?|chars|vehicles?/starships?|starships?/vehicles?|"
                           r"admiral.?s orders?|epic events?|jedi tests?)\s*[:\-]?\s*\d*\s*(\(.*\))?\s*\d*\s*:?$", re.I)
RETRO = re.compile(r'retro|decipher-cards-only|PREF2|PSETop|Classic NoV', re.I)
SECTION_PLURAL = {'Character': 'CHARACTERS', 'Effect': 'EFFECTS', 'Interrupt': 'INTERRUPTS', 'Location': 'LOCATIONS',
                  'Starship': 'STARSHIPS', 'Vehicle': 'VEHICLES', 'Weapon': 'WEAPONS', 'Device': 'DEVICES'}

# ---- objectives -------------------------------------------------------------------------------------------------
PATTERNS = {  # distinctive start of the objective line in a hand-typed list (normalised: lowercase a-z0-9 only)
    'Agents Of Black Sun': ['agentsofblacksun'], 'Bring Him Before Me': ['bringhimbeforeme'],
    'Carbon Chamber Testing': ['carbonchambertesting'], 'Court Of The Vile Gangster': ['courtofthevilegangster'],
    'Endor Operations': ['endoroperations'], 'Hunt Down And Destroy The Jedi': ['huntdown'],
    'ISB Operations': ['isboperations'], 'Imperial Occupation': ['imperialoccupation'], 'Invasion': ['invasion'],
    'Let Them Make The First Move': ['letthemmakethefirstmove'], 'My Kind Of Scum': ['mykindofscum'],
    'My Lord, Is That Legal?': ['mylordisthatlegal'], 'No Money, No Parts, No Deal!': ['nomoneynoparts'],
    'Ralltiir Operations': ['ralltiiroperations'], 'Set Your Course For Alderaan': ['setyourcourse'],
    'This Deal Is Getting Worse All The Time': ['thisdealisgettingworse'], 'Agents In The Court': ['agentsinthecourt'],
    'Dantooine Base Operations': ['dantooinebaseoperations'], 'Hidden Base': ['hiddenbase'],
    'Local Uprising': ['localuprising'], 'Massassi Base Operations': ['massassibaseoperations'],
    'Mind What You Have Learned': ['mindwhatyouhavelearned'], 'Plead My Case To The Senate': ['pleadmycase'],
    'Quiet Mining Colony': ['quietminingcolony'], 'Rebel Strike Team': ['rebelstriketeam'],
    'Rescue The Princess': ['rescuetheprincess'], "The Hyperdrive Generator's Gone": ['thehyperdrivegenerator', 'hyperdrivegenerator'],
    'There Is Good In Him': ['thereisgoodinhim'], 'Watch Your Step': ['watchyourstep'], 'We Have A Plan': ['wehaveaplan'],
    "We'll Handle This": ['wellhandlethis'], 'You Can Either Profit By This...': ['youcaneitherprofit', 'profitfromthis', 'profitbythis'],
}


class Context:
    """Card tables, resolvers and the corpus, loaded once."""

    def __init__(self):
        self.cards = C.Cards()
        self.gemp = IP.class_index()
        self.obj = {cid: c for cid, c in self.cards.by_id.items()
                    if c.get('cardCategory') == 'OBJECTIVE' and not cid.endswith('_BACK') and cid.split('_')[0].isdigit()}
        self.dec_obj = sorted({(c['side'], c['title']) for cid, c in self.obj.items() if int(cid.split('_')[0]) < 200})
        assert set(PATTERNS) == {t for _, t in self.dec_obj}
        self.by_side, self.shields, self.dt_by_id = DT.load_cards()
        self.aliases = DT.load_aliases(ROOT / 'data/decks/online-decipher/aliases.tsv', {'approved'}, self.dt_by_id)
        self.res = IP.resolver()
        self.corpus = {}
        for f in sorted(glob.glob(str(ROOT / 'data/decks/corpus/*.xml')) + glob.glob(str(ROOT / 'data/decks/v4/corpus/*.xml'))):
            self.corpus.setdefault(C.fingerprint(self.load_ids(*IP.parse_gemp_xml(open(f, 'rb').read()))), Path(f).stem)

    def load_ids(self, ins, outs):
        return C.load(ins, outs, self.cards)

    def objective_of_ids(self, ids):
        found = [i for i in ids if i in self.obj]
        if not found:
            return '(no objective)'
        return self.obj[found[0]]['title'] if int(found[0].split('_')[0]) < 200 else '(Virtual objective)'

    def playable(self, deck):
        ids = deck['in'] + deck['out']
        return (len(deck['in']) == 60 and deck['side'] in ('DARK', 'LIGHT')
                and all(i in self.gemp and not i.startswith('601_') for i in ids))

    def status(self, deck):
        fp = C.fingerprint(deck)
        if fp in self.corpus:
            return 'in corpus'
        return 'parsed, playable' if self.playable(deck) else 'parsed, not playable'

    # ---- name resolution for repairs
    def resolve_name(self, name, side, section=None, decipher_source=False):
        cid, how = self._resolve_name(name, side, section, decipher_source)
        if cid is None and how != 'original virtual card':
            # combo halves in the other order ("Endor Celebration & Free Ride"), or a trailing comment: "Battle Plan (Start if ISB or Ops)", "Romas Lock Navander (No reacting)", "X - note"
            swapped = ' & '.join(reversed([x.strip() for x in name.split('&')])) if name.count('&') == 1 else ''
            for stripped in (swapped, re.sub(r'\s*\((?![VA]I?\))[^()]*\s[^()]*\)?\s*$', '', name), re.sub(r'\s+-\s+.*$', '', name)):
                if stripped and stripped != name:
                    cid2, how2 = self._resolve_name(stripped, side, section, decipher_source)
                    if cid2:
                        return cid2, how2 if how2 != 'exact' else 'partial'
        return cid, how

    def _resolve_name(self, name, side, section=None, decipher_source=False):
        """(cardId or None, how). Approved alias, then the corpus resolver (exact title, (V)/(AI) markers, near-identical
        spelling), then the decktech resolver's partial/fuzzy match on Decipher titles. Only cards with a class in today's
        Gemp count."""
        name = name.strip()
        short = DT.SIDES[side]
        alias = self.aliases[short].get(('', DT.norm(name)))
        if alias and alias['cardId'] in self.gemp:
            return alias['cardId'], 'alias'
        if re.search(r'\((Virtual Set|VS)\s*\d+\)', name):  # an original Virtual Set card: no Gemp blueprint
            return None, 'original virtual card'
        cid, note = self.res.resolve(name, side, section=SECTION_PLURAL.get(section), decipher_only=decipher_source)
        if cid and note != 'legacy_601' and IP.norm_id(cid) in self.gemp:
            return IP.norm_id(cid), 'exact'
        if decipher_source or not re.search(r'\(V\)', name):
            for table in (self.by_side[short], self.shields[short]):
                found, how = DT._lookup(name, table)
                if found and len(found) == 1 and found[0]['cardId'] in self.gemp:
                    return found[0]['cardId'], how.split(':')[0]
        return None, note or 'unresolved'

    def resolve_lines(self, lines, side, decipher_source):
        ins, outs, problems, matches = [], [], [], []
        for count, name, section in lines:
            cid, how = self.resolve_name(name, side, section, decipher_source)
            if cid is None:
                problems.append(f'{count}x {name} [{how}]')
                continue
            if how in ('partial', 'fuzzy', 'alias'):
                matches.append(f'{name} -> {self.cards.title(cid)} ({cid}, {how})')
            card = self.cards.by_id.get(cid, {})
            (outs if card.get('cardCategory') == 'DEFENSIVE_SHIELD' else ins).extend([cid] * count)
        return ins, outs, problems, matches


def objective_of_text(text):
    for raw in text.splitlines():
        line = re.sub(r'^\W*(\d+\s*x?\s*)?', '', raw.strip(), flags=re.I)
        line = re.sub(r'^(objective|obj)\s*[:\-]?\s*', '', line, flags=re.I)
        key = re.sub(r'[^a-z0-9]', '', line.lower().replace('&', 'and'))
        for title, pats in PATTERNS.items():
            if any(key.startswith(p) for p in pats):
                if title == 'Invasion' and not re.search(r'invasion\s*(/|$|x\d|\()', line, re.I):
                    continue
                return title
    return '(no objective)'


def text_lines(text):
    """[(count, name, section)] from a hand-typed list (decktech post / PC text list), with the decktech parser rules."""
    out, section = [], None
    for raw in html.unescape(text).replace('\r', '').split('\n'):
        line = raw.strip().strip("'").strip()
        if not line:
            continue
        if DT.HEADER.match(line) or DT.SECTION.match(line) or EXTRA_SECTION.match(line):
            section = re.sub(r'[^A-Za-z]', '', line).capitalize().rstrip('s') or None
            if section and 'shield' in line.lower():
                section = 'Defensive Shield'
            continue
        line = re.sub(r'^(objective|obj|starting|epic event)\s*[:\t]\s*', '', line, flags=re.I).strip()
        line = DT.NOTE.sub('', line)
        line = re.sub(r'(?<=[a-z])x\s*(\d+)$', r' x\1', line)
        line = re.sub(r'\s*\(x\s*(\d+)\)\s*$', r' x\1', line, flags=re.I)
        m = DT.COUNT.match(line)
        count = int(next((g for g in m.group(1, 2, 3, 5, 6, 7) if g), 1))
        name = m.group(4).strip(' -*')
        if name:
            out.append((count, name, section))
    return out


# ---- wiki deck pages ---------------------------------------------------------------------------------------------
def page_info(text):
    info = dict(re.findall(r"^\* '''([^:']+):''' *(.*)$", text, re.M))
    side = 'LIGHT' if '[[Light]]' in info.get('Side', '') else 'DARK' if '[[Dark]]' in info.get('Side', '') else ''
    return info, side


def page_lines(text):
    """[(count, name, section)] of a wiki deck page's decklist; deck-title lines in 'Unknown' are skipped."""
    body = text.split('== Decklist ==', 1)[-1]
    out, section = [], None
    for line in body.splitlines():
        m = re.match(r"^'''([A-Za-z' ]+)'''$", line.strip())
        if m:
            section = m.group(1)
            continue
        m = re.match(r'^\*\s*(\d+)x\s*(.+?)\s*$', line)
        if not m:
            continue
        count, ref = int(m.group(1)), m.group(2)
        cl = re.match(r'\{\{CardLink\|([^|}]+)', ref)
        if cl:
            name = cl.group(1)
        else:
            lk = re.match(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', ref)
            name = lk.group(1) if lk else ref
        name = re.sub(r'\s*/\s*.*$', '', name) if section == 'Objective' else name  # "Front / Back" -> front
        name = re.sub(r'\s*\((Dark|Light)\)$', '', name)  # wiki disambiguation "Alter (Dark)"
        if section == 'Unknown' and ' - ' in name:
            continue
        out.append((count, name.strip(), section))
    return out


# ---- the records -------------------------------------------------------------------------------------------------
def collection_category(meta):
    oid = meta['our_id']
    if oid.startswith('swdb-'):
        return 'swccgdb.com user decks', 'public'
    if oid.startswith('decktech_'):
        return 'decktech.net archive', 'public'
    if oid.startswith(('web-', 'local-')) or meta['source'].startswith('gemp-swccg-agent'):
        return 'web articles / local / agent', 'public'
    if RETRO.search(meta['event']) or RETRO.search(meta['where']):
        return 'PC tournaments: retro events', 'tournament'
    return 'PC tournaments: modern', 'tournament'


PRIORITY = ['PC tournaments: modern', 'PC tournaments: retro events', 'swccgdb.com user decks', 'web articles / local / agent',
            'decktech.net archive', 'BillBisco2 Drive folder', 'wiki: tournament pages']


def priority(category):
    return PRIORITY.index(category) if category in PRIORITY else len(PRIORITY) + (category == 'wiki: files no page links')


def records(ctx, decktech_archive=None):
    """All records, in category priority order (a deck seen earlier wins the fingerprint)."""
    return sorted(_records(ctx, decktech_archive), key=lambda r: priority(r['category']))


def _records(ctx, decktech_archive=None):
    cards = ctx.cards
    out = []
    # 1. our collection
    for deck, meta in C.our_collection(cards):
        cat, kind = collection_category(meta)
        out.append(dict(id=meta['our_id'], category=cat, kind=kind, title=meta['deck_name'], url=meta['url'] or meta['where'],
                        side=deck['side'], objective=ctx.objective_of_ids(deck['in'] + deck['out']), status=ctx.status(deck),
                        deck=deck, fp=C.fingerprint(deck), lines=None, decipher_source=False, note=''))
    # 2. PC/forum text lists whose names do not all resolve
    seen = set()

    def archive_members():
        for z in sorted((ROOT / 'data/all-tournament-decks-2021-2026').iterdir()):
            items = IP.iter_archive(z.read_bytes(), z.name) if z.name.lower().endswith(IP.ARCH) else [(z.name, z.read_bytes())]
            for name, data in items:
                yield z.stem, f'{z.relative_to(ROOT)}#{name}', '', name, data
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
                yield m['event'], f'data/raw/pc-modern/{m["file"]}#{name}', m.get('attachment_url', ''), name, data

    for ev, where, url, name, data in archive_members():
        if not name.lower().endswith('.txt') or b'<deck' in data[:2000]:
            continue
        got = C.parse_member(name, data, cards)
        if not got or not got[1]:
            continue
        h = hashlib.sha1(data).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        text = data.decode('utf-8', 'replace')
        cat = 'PC tournaments: retro events' if RETRO.search(ev) else 'PC tournaments: modern'
        out.append(dict(id=f'pctext:{h[:12]}', category=cat, kind='tournament', title=Path(name).stem, url=url or where,
                        side=got[0]['side'], objective=objective_of_text(text), status='unparsed', deck=None, fp=None,
                        lines=text_lines(text), decipher_source=False, note='; '.join(got[1][:5])))
    # 3. decktech: lists already tried (parsed ones are in our collection above), then the posts never tried
    index = list(csv.DictReader(open(ROOT / 'data/decks/online-decipher/search-index.tsv'), delimiter='\t'))
    parsed_posts = {re.search(r'-(\d+)_', f).group(1) for f in glob.glob(str(ROOT / 'data/decks/online-decipher/decktech_*.xml'))}
    for r in index:
        if r['post_id'] in parsed_posts:
            continue
        text = (ROOT / 'data/decks/online-decipher' / r['file']).read_text(errors='replace')
        side = {'DS': 'DARK', 'LS': 'LIGHT'}.get(r['side'], '')
        out.append(dict(id=f'decktech:{r["post_id"]}', category='decktech.net archive',
                        kind='tournament' if EVENT_WORDS.search(r['title']) else 'public', title=r['title'], url=r['url'],
                        side=side, objective=objective_of_text(text), status='unparsed', deck=None, fp=None,
                        lines=text_lines('\n'.join(l for l in text.splitlines() if not l.startswith('#'))),
                        decipher_source=True, note=r['status']))
    tried = {r['post_id'] for r in index}
    if decktech_archive:
        for f in sorted(glob.glob(str(Path(decktech_archive) / '_posts/*.md'))):
            pid = re.search(r'-(\d+)\.md$', f).group(1)
            t = open(f, errors='ignore').read()
            m = re.search(r'Cards:(.*?)\n\s*Strategy:', t, re.S)
            if not m or len(m.group(1).strip(" '\n")) < 50 or pid in tried:
                continue
            hdr = dict(re.findall(r'^(title|author|date): !? ?"?(.*?)"?$', t, re.M))
            side = 'DARK' if '- Dark' in t[:400] else 'LIGHT' if '- Light' in t[:400] else ''
            out.append(dict(id=f'decktech:{pid}', category='decktech.net archive',
                            kind='tournament' if EVENT_WORDS.search(hdr.get('title', '')) else 'public',
                            title=hdr.get('title', ''), url=f'{DECKTECH_URL}{pid}/', side=side,
                            objective=objective_of_text(m.group(1)), status='todo', deck=None, fp=None,
                            lines=text_lines(m.group(1)), decipher_source=True, note=''))
    # 4. BillBisco2: Drive folder and his 1996 Worlds wiki decks
    rows = list(csv.DictReader(open(BISCO / 'compare.csv')))
    tournament_paths = collections.defaultdict(bool)
    for r in rows:
        first = r['his_duplicate_of'] or r['path']
        tournament_paths[first] |= 'Tournament' in r['path'] or r['path'].startswith('wiki-1996')
    for r in rows:
        if not r['status'].startswith('parsed') or r['his_duplicate_of']:
            continue
        src = (BISCO / 'raw' / r['path']) if not r['path'].startswith('wiki-1996') else \
            (BISCO / 'wiki-1996-worlds/raw' / r['path'].split('/', 1)[1])
        deck = ctx.load_ids(*IP.parse_gemp_xml(src.read_bytes()))
        incomplete = r['match'] == 'incomplete'
        out.append(dict(id=f'bisco:{r["path"]}', category='BillBisco2 Drive folder',
                        kind='tournament' if tournament_paths[r['path']] else 'public', title=Path(r['path']).stem,
                        url=r['source_url'], side=deck['side'], objective=ctx.objective_of_ids(deck['in'] + deck['out']),
                        status='unparsed' if incomplete else ctx.status(deck), deck=None if incomplete else deck,
                        fp=None if incomplete else C.fingerprint(deck), lines=None, decipher_source=True,
                        note='incomplete transcription (his v0 file)' if incomplete else ''))
    # 5. wiki.swccg.com GEMP files, with their deck page when one links the file
    pages = {}
    for f in (WIKI / 'pages').glob('*.wiki'):
        pages[f.stem] = f.read_text(errors='replace')
    for w in csv.DictReader(open(WIKI / 'wiki-files.csv')):
        data = (WIKI / 'files' / w['file']).read_bytes()
        linked = [p for p in w['pages'].split(' | ') if p]
        page_title = next((p for p in linked if 'Event:' in pages.get(re.sub(r'[/\\:*?"<>|]', '_', p), '')), linked[0] if linked else '')
        text = pages.get(re.sub(r'[/\\:*?"<>|]', '_', page_title), '') if page_title else ''
        info, pside = page_info(text) if text else ({}, '')
        if text:
            published = re.search(r'\(([^)]+)\)\s*$', info.get('Published', ''))
            cat = 'wiki: tournament pages' if 'Event' in info else f'wiki: {published.group(1) if published else "other"} (catalog)'
            kind = 'tournament' if 'Event' in info else 'public'
        else:
            cat, kind = 'wiki: files no page links', 'tournament' if EVENT_WORDS.search(w['name']) or re.search(r'Day|T8|T16|PC\d\d|OCS|EGP|Nats|MPC', w['name']) else 'public'
        if b'<deck' not in data[:2000]:  # text list: drop the title line and the singular section headers
            tl = data.decode('utf-8', 'replace').replace('\r', '').split('\n')
            if tl and ' - ' in tl[0]:
                tl = tl[1:]
            tl = [l for l in tl if l.strip() and not re.fullmatch(r"[A-Z' ]+\s*(\(\d+\))?:?\s*", l.strip())]
            data = '\n'.join(tl).encode()
        got = C.parse_member(w['file'], data, cards)
        lines = page_lines(text) if text else None
        decipher_source = bool(text) and not re.search(r'\(V\)|Virtual|Original VS', text)
        base = dict(id=f'wiki:{w["name"]}', category=cat, kind=kind, title=page_title or w['name'],
                    url=(WIKI_URL + page_title.replace(' ', '_')) if page_title else w['url'], lines=lines,
                    decipher_source=decipher_source)
        if got is None:
            continue
        deck, problems = got
        if problems:
            out.append(dict(base, side=deck['side'] or pside, objective=objective_of_text(data.decode('utf-8', 'replace')),
                            status='unparsed', deck=None, fp=None, note='; '.join(problems[:5])))
            continue
        st = ctx.status(deck)
        page_count = sum(c for c, n, s in lines if s != 'Defensive Shield') if lines else None
        note = ''
        if page_count and page_count > len(deck['in']):
            note = f'page lists {page_count} cards, file has {len(deck["in"])}'
            if st == 'parsed, playable':
                st = 'parsed, not playable'  # the file lost cards: it is not the deck as played
        out.append(dict(base, side=deck['side'], objective=ctx.objective_of_ids(deck['in'] + deck['out']), status=st,
                        deck=deck, fp=C.fingerprint(deck), note=note))
    return out


def scans():
    return list(csv.DictReader(open(WIKI / 'scans.csv')))
