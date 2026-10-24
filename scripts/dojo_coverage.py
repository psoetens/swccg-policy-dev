"""Plan objective-dojo A2: fatal coverage gate for the Dojo reward model.

Every objective, Epic Event and Jedi Test in a schedule's decks must have an entry in data/dojo/rewards.json (and its
predicates in ObjectiveDojo.java, which validates itself when it loads). The Java engine repeats this check when a
run loads its decks; this script lets the launchers fail before anything starts.

    python scripts/dojo_coverage.py SCHEDULE [SCHEDULE ...]     # exit 0 = covered, 1 = missing rewards (listed)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD_DB = ROOT / 'data/cards/card_blueprint_database-5f2f57d1336527ecf9505db0052f2c54ca54a7ed.json'
REWARDS = ROOT / 'data/dojo/rewards.json'
CATEGORIES = ('OBJECTIVE', 'EPIC_EVENT', 'JEDI_TEST')


def schedule_decks(path):
    """Deck files of a pool ("DS|LS<TAB>path") or schedule ("ds.xml<TAB>ls.xml") file."""
    decks = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = [part.strip() for part in line.split('\t')]
        if parts[0].upper() in ('DS', 'LS'):
            decks.append(parts[1])
        else:
            decks.extend(part for part in parts if part.endswith('.xml'))
    return sorted(set(decks))


def dojo_cards(deck, categories):
    ids = set(re.findall(r'blueprintId="([^"]+)"', (ROOT / deck).read_text()))
    return {card for card in ids if categories.get(card) in CATEGORIES}


def missing(schedules):
    """{card id: (title, category, [decks])} for every Dojo-category card without a reward entry."""
    cards = json.loads(CARD_DB.read_text())
    categories = {card['cardId']: card['cardCategory'] for card in cards}
    titles = {card['cardId']: card['title'] for card in cards}
    covered = {key.split('~')[0] for key in json.loads(REWARDS.read_text())['items']}
    result = {}
    for schedule in schedules:
        for deck in schedule_decks(schedule):
            for card in dojo_cards(deck, categories) - covered:
                result.setdefault(card, (titles[card], categories[card], set()))[2].add(deck)
    return {card: (title, category, sorted(decks)) for card, (title, category, decks) in sorted(result.items())}


def require(schedules):
    """Raise SystemExit with the user instruction if any Dojo reward is missing."""
    gaps = missing(schedules)
    if not gaps:
        return
    lines = [f'Dojo coverage: no reward for {len(gaps)} card(s). Training with these decks is refused.']
    for card, (title, category, decks) in gaps.items():
        lines.append(f'  supply the Dojo reward for {title} ({card}, {category}) in data/dojo/rewards.json + '
                     f'ObjectiveDojo.java before training with these decks: {", ".join(Path(d).name for d in decks)}')
    raise SystemExit('\n'.join(lines))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    require(sys.argv[1:])
    for schedule in sys.argv[1:]:
        print(f'covered: {schedule} ({len(schedule_decks(schedule))} decks)')


if __name__ == '__main__':
    main()
