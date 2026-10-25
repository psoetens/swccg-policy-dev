"""Plan beta-server: the official server's library (sample) decks, as SQL for the beta DB image
(db-init/45-library-decks.sql). Peter: identical to the official server, so the SQL replaces Gemp's own sample decks
(sample_decks.sql, utinni_sample_decks.sql: 69 decks, 2 not on the official list, 4 with other contents) with the
official library: all of Librarian's decks are deleted, then every official deck is inserted (owner Librarian).

    python deploy/beta/fetch_library.py [--base https://gemp.starwarsccg.org] > deploy/beta/db-init/45-library-decks.sql

Reads the public, no-login endpoints GET /gemp-swccg-server/deck/libraryList and /deck/library?deckName=..., one
request per second. Contents use Gemp's DeckSerialization format: "cards|cards outside the deck", in the server's
listed order. Gemp reads the library at start: a running server needs a restart after the SQL is applied.
"""
import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'brainiac-beta-library-sync'}), timeout=30) as r:
        return r.read()


def sql_string(text):
    return "'" + text.replace('\\', '\\\\').replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='https://gemp.starwarsccg.org')
    a = ap.parse_args()
    server = a.base.rstrip('/') + '/gemp-swccg-server'
    names = [e.text for e in ET.fromstring(get(server + '/deck/libraryList')) if e.text]
    print(f'-- Plan beta-server: the {len(names)} library decks of {a.base}, fetched {time.strftime("%Y-%m-%d")} '
          f'(deploy/beta/fetch_library.py); they replace Gemp\'s own sample decks.')
    print("DELETE FROM deck WHERE player_id = (SELECT id FROM player WHERE name='Librarian');")
    for name in names:
        time.sleep(1)
        deck = ET.fromstring(get(server + '/deck/library?' + urllib.parse.urlencode({'deckName': name})))
        cards = [c.get('blueprintId') for c in deck.findall('card')]
        outside = [c.get('blueprintId') for c in deck.findall('cardOutsideDeck')]
        if not cards:
            raise SystemExit(f'{name}: empty on the server')
        contents = ','.join(cards) + '|' + ','.join(outside)
        print("INSERT INTO deck (player_id, name, contents) VALUES((SELECT id FROM player WHERE name='Librarian'),"
              f"{sql_string(name)},{sql_string(contents)});")
        print(f'{name}: {len(cards)} cards, {len(outside)} outside', file=sys.stderr)


if __name__ == '__main__':
    main()
