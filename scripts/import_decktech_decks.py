"""Plan import-decktech-decks: convert decktech.net archive card lists to gemp deck XML, strictly.

    import_decktech_decks.py DECKS_DIR [--out data/decks/online-decipher] [--index INDEX.tsv]
                             [--aliases data/decks/online-decipher/aliases.tsv] [--alias-status approved]
                             [--targets 11_80,12_180,12_89,13_65,13_73]

A list is saved only if every card line resolves to exactly one Decipher-era card (set < 200) of the deck's side, the
deck has exactly 60 cards (Defensive Shields go outside the deck), and it holds at least one target card.

Resolution of a card name:
- The alias table first (only rows whose status is in --alias-status): nicknames ("EPP Vader") and version choices.
- Then titles after normalisation (case, punctuation, spacing); "A / B" also tries the combo title "A & B" and each
  half; a title without its parenthetical; a unique title containing the name (names of 10+ characters); a single
  close title (ratio >= 0.85, names of 6+ characters). Partial and fuzzy matches are listed in sources.csv.
- A title is one card only if all its Decipher prints have the same game text (alternate images). Titles reprinted with
  different text (e.g. Alter 1_71 vs 12_54, Boba Fett 5_91 vs 7_167) are ambiguous: an approved 'version' alias must
  pick one, otherwise the deck is rejected. Lines in a "Defensive Shields" section resolve to the Defensive Shield
  print; elsewhere Defensive Shields are never picked.
Nothing is guessed: an unresolved or ambiguous line rejects the deck. Writes <out>/*.xml, sources.csv, rejected.csv.
"""
import argparse
import csv
import difflib
import glob
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[1]
CARD_DB = next(Path(p) for p in sorted(glob.glob(str(ROOT / 'data/cards/card_blueprint_database-*.json'))) if 'provenance' not in p)
# Counts: "2x Name", "x2 Name", "2 Name", "Name x2", "Name (2)", "Name (2x)", "Name x2 (comment)", "Name 2x".
COUNT = re.compile(r'^(?:(\d+)\s*x\s+|x\s*(\d+)\s+|(\d+)\s+(?=[A-Za-z]))?(.*?)'
                   r'(?:\s*x\s*(\d+)(?:\s*\(.*\))?|\s*\((\d+)\s*x?\)|\s+(\d+)\s*x)?\s*$', re.I)
HEADER = re.compile(r'^[A-Za-z /&\-]+\(\d+\)\s*:?\s*$')
# Section headers written without a count, or with a count glued on ("Starting6", "Locations - 4", "=Starting Stuff").
SECTION = re.compile(r'^[=*\-\s]*(starting( stuff)?|start|objective|obj|locations?|sites?|characters?|chars|people|droids?|'
                     r'aliens?|starships?|vehicles?|starships?/vehicles?|weapons?|weapon.s|weapons?/devices?|devices?|'
                     r'effects?|fx|interrupts?|interupts?|interrupts?/effects?|epic events?|podracers?|admiral.s orders?|'
                     r'defensive sh(ie|ei)lds?|shields?|other( stuff)?|misc\w*)'
                     r'\s*[-:=]?\s*\d*\s*:?\s*$', re.I)
# Annotations after a card name: "- STARTING", "(start)", "(starting)".
NOTE = re.compile(r'\s*(-\s*starting|\((start|starting)\))\s*$', re.I)
SIDES = {'DARK': 'DS', 'LIGHT': 'LS'}


def norm(text):
    return re.sub(r'[^a-z0-9]', '', text.lower().replace('’', "'").replace('&', 'and'))


def card_key(card_id):
    return tuple(int(part) for part in card_id.split('_'))


def load_cards():
    """Decipher-era cards (set < 200), front sides: (by_side, shields, by_id).

    by_side[side][title] is the list of distinct versions of that title: prints with identical game text (alternate
    images) are one version (the earliest print); prints with different text stay separate. Defensive Shields are kept
    apart in shields[side][title]."""
    versions = defaultdict(lambda: defaultdict(dict))
    shields = defaultdict(dict)
    by_id = {}
    for card in json.loads(CARD_DB.read_text()):
        card_id = card['cardId']
        if card_id.endswith('_BACK') or not card_id.split('_')[0].isdigit() or int(card_id.split('_')[0]) >= 200:
            continue
        side = SIDES.get(card.get('side'))
        if not side:
            continue
        by_id[card_id] = card
        key = norm(card['title'])
        if card.get('cardCategory') == 'DEFENSIVE_SHIELD':
            shields[side][key] = [card]
            continue
        text = '|'.join(norm(card.get(field) or '') for field in ('gameText', 'locationDarkSideGameText', 'locationLightSideGameText'))
        text += '|' + str(card.get('cardCategory'))
        current = versions[side][key].get(text)
        if current is None or card_key(card_id) < card_key(current['cardId']):
            versions[side][key][text] = card
    by_side = {side: {key: sorted(found.values(), key=lambda c: card_key(c['cardId'])) for key, found in titles.items()}
               for side, titles in versions.items()}
    return by_side, shields, by_id


def load_aliases(path, statuses, by_id):
    """aliases[side][(post, normalised name)] = card, from rows whose status is in statuses ('*' = any status). post is
    '' for a row that applies to every list, else the decktech post id the row is limited to."""
    aliases = defaultdict(dict)
    if not path or not Path(path).exists():
        return aliases
    for row in csv.DictReader(open(path), delimiter='\t'):
        if not row['card_id'] or ('*' not in statuses and row['status'] not in statuses):
            continue
        card = by_id.get(row['card_id'])
        if card is None or SIDES.get(card.get('side')) != row['side']:
            raise SystemExit(f'alias {row["name"]!r}: {row["card_id"]} is not a Decipher-era {row["side"]} card')
        aliases[row['side']][(row.get('post') or '', norm(row['name']))] = card
    return aliases


def _lookup(name, cards):
    """Name -> (versions list, how) or (None, 'unresolved')."""
    candidates = [name]
    if '/' in name:
        parts = [part.strip() for part in name.split('/')]
        candidates = [' & '.join(parts), *parts]
    for candidate in candidates:
        found = cards.get(norm(candidate))
        if found:
            return found, 'exact'
    key = norm(candidates[0])
    # A title without its parenthetical (e.g. "(Docking Bay)"), or a unique title containing the name ("Sail Barge:
    # Passenger Deck" in "Jabba's Sail Barge: Passenger Deck"); names of at least 10 characters only.
    stripped = [k for k, found in cards.items() if norm(re.sub(r'\(.*?\)', '', found[0]['title'])) == key]
    if len(stripped) == 1:
        return cards[stripped[0]], f'partial:{cards[stripped[0]][0]["title"]}'
    if len(key) >= 10:
        contained = [k for k in cards if key in k]
        if len(contained) == 1:
            return cards[contained[0]], f'partial:{cards[contained[0]][0]["title"]}'
    # Typos: a single close title (ratio >= 0.85, names of at least 6 characters), never between two candidates.
    best = difflib.get_close_matches(key, list(cards), n=2, cutoff=0.85) if len(key) >= 6 else []
    if len(best) == 1:
        return cards[best[0]], f'fuzzy:{cards[best[0]][0]["title"]}'
    return None, 'unresolved'


def resolve(name, cards, aliases, post=''):
    """(card, how); card None with how 'unresolved' or 'ambiguous:<ids>'."""
    alias = aliases.get((post, norm(name))) or aliases.get(('', norm(name)))
    if alias:
        return alias, 'alias'
    found, how = _lookup(name, cards)
    if found is None:
        return None, how
    if len(found) > 1:
        return None, 'ambiguous:' + '/'.join(card['cardId'] for card in found)
    return found[0], how


def parse(path):
    meta, lines, shield_section = {}, [], False
    for raw in path.read_text(errors='replace').splitlines():
        line = html.unescape(raw).strip()
        if line.startswith('#'):
            match = re.match(r'#\s*(\w+):\s*(.*)', line)
            if match:
                meta[match.group(1).lower()] = match.group(2)
            continue
        if not line:
            continue
        if HEADER.match(line) or SECTION.match(line):
            shield_section = 'sh' in line.lower() and ('shield' in line.lower() or 'sheild' in line.lower())
            continue
        line = re.sub(r'^(objective|obj|starting|epic event)\s*[:\t]\s*', '', line, flags=re.I).strip()
        line = NOTE.sub('', line)
        line = re.sub(r'(?<=[a-z])x\s*(\d+)$', r' x\1', line)  # 'Wattox4' -> 'Watto x4'
        lines.append((line, shield_section))
    return meta, lines


def convert(path, side, by_side, shields, aliases, targets):
    """(meta, inside, outside, problems, matches, held) for one card list."""
    meta, lines = parse(path)
    post = (re.search(r'decktech_archives/(\d+)', meta.get('source', '')) or re.search(r'-(\d+)_', path.name)).group(1)
    inside, outside, problems, matches = [], [], [], []
    for line, in_shields in lines:
        match = COUNT.match(line)
        count = int(next((group for group in match.group(1, 2, 3, 5, 6, 7) if group), 1))
        name = match.group(4).strip(' -*')
        card, how = (None, '')
        if in_shields:
            card, how = resolve(name, shields[side], aliases[side], post)
        if card is None:
            card, how = resolve(name, by_side[side], aliases[side], post)
        if card is None and how == 'unresolved' and not in_shields:
            # A title that exists only as a Defensive Shield, listed outside a shields section.
            shield, shield_how = resolve(name, shields[side], {}, post)
            if shield is not None and shield_how == 'exact':
                card, how = shield, shield_how
        if card is None:
            problems.append(f'{name} [{how}]' if how.startswith('ambiguous') else name)
            continue
        if how.startswith(('fuzzy', 'partial', 'alias')):
            matches.append(f'{name} -> {card["title"]} ({card["cardId"]}, {how.split(":")[0]})')
        (outside if card.get('cardCategory') == 'DEFENSIVE_SHIELD' else inside).extend([card] * count)
    held = sorted({card['cardId'] for card in inside + outside} & targets)
    return meta, inside, outside, problems, matches, held


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('decks', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/decks/online-decipher')
    parser.add_argument('--targets', default='11_80,12_180,12_89,13_65,13_73')
    parser.add_argument('--index', type=Path, help='INDEX.tsv of the search (side and status per list)')
    parser.add_argument('--aliases', type=Path, default=ROOT / 'data/decks/online-decipher/aliases.tsv')
    parser.add_argument('--alias-status', default='approved', help="comma-separated alias statuses to use ('*' = all)")
    args = parser.parse_args()
    targets = set(args.targets.split(','))
    by_side, shields, by_id = load_cards()
    aliases = load_aliases(args.aliases, set(args.alias_status.split(',')), by_id)
    target_sides = {target: SIDES.get(by_id[target].get('side')) for target in targets if target in by_id}
    sides = {}
    if args.index:
        for row in csv.DictReader(args.index.open(), delimiter='\t'):
            sides[Path(row['file']).name] = (row['side'], row['status'])
    args.out.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for path in sorted(args.decks.glob('*.txt')):
        side, status = sides.get(path.name, ('', ''))
        if status.startswith(('EXCLUDED', 'incomplete')):
            rejected.append({'file': path.name, 'reason': f'search status: {status}'})
            continue
        if not side:  # no index: the side of the target cards named in the header ("# Targets: ['12_180']")
            header = path.read_text(errors='replace')[:1000]
            held_sides = {target_sides.get(t) for t in re.findall(r"'(\d+_\d+)'", header.split('# Targets:')[-1].split('\n')[0])}
            side = held_sides.pop() if len(held_sides) == 1 else ''
        if side not in ('DS', 'LS'):
            rejected.append({'file': path.name, 'reason': f'unknown side {side!r}'})
            continue
        meta, inside, outside, problems, matches, held = convert(path, side, by_side, shields, aliases, targets)
        reason = ('unresolved: ' + '; '.join(problems) if problems else
                  f'{len(inside)} cards in deck (not 60)' if len(inside) != 60 else
                  'no target card' if not held else None)
        if reason:
            rejected.append({'file': path.name, 'reason': reason})
            continue
        name = re.sub(r'\.txt$', '', path.name)
        xml = ["<?xml version='1.0' encoding='UTF-8'?>", '<deck>']
        xml += [f'  <card blueprintId="{card["cardId"]}" title={quoteattr(card["title"])} />' for card in inside]
        xml += [f'  <cardOutsideDeck blueprintId="{card["cardId"]}" title={quoteattr(card["title"])} />' for card in outside]
        xml.append('</deck>')
        (args.out / f'{name}_{side}.xml').write_text('\n'.join(xml) + '\n')
        saved.append({'file': f'{name}_{side}.xml', 'side': side, 'targets': ' '.join(held), 'source': meta.get('source', ''),
                      'title': meta.get('title', ''), 'author': meta.get('author', ''), 'date': meta.get('date', ''),
                      'outside_deck': len(outside), 'matches': ' | '.join(matches)})
    for name, rows in (('sources.csv', saved), ('rejected.csv', rejected)):
        target = args.out / name
        if rows:
            with target.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        elif target.exists():
            target.unlink()
    per_target = defaultdict(int)
    for row in saved:
        for target in row['targets'].split():
            per_target[target] += 1
    print(json.dumps({'saved': len(saved), 'rejected': len(rejected), 'per_target': dict(per_target)}, indent=1))


if __name__ == '__main__':
    main()
