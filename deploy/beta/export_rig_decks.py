"""Plan beta-server Amendment 2: the decks Peter played against Brainiac in the local test rig (his personal decks
there: "DS2/Endor: Battle Of Endor 2.0 (D)" etc.) as library decks of the beta (owner Librarian), so every tester sees
them next to the official library: deploy/beta/db-init/46-brainiac-test-decks.sql.

    python deploy/beta/export_rig_decks.py [--container gemp_swccg_db_1] [--owner peter] \\
        [--env ~/src/gemp-swccg-agent/gemp-swccg-public/src/.env] > deploy/beta/db-init/46-brainiac-test-decks.sql

Reads the rig's MariaDB with `docker exec` (credentials from the rig's .env). Runs after 45-library-decks.sql; a deck
name the official library already has is refused (no silent duplicates).
"""
import argparse
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--container', default='gemp_swccg_db_1')
    ap.add_argument('--owner', default='peter')
    ap.add_argument('--env', type=Path, default=Path.home() / 'src/gemp-swccg-agent/gemp-swccg-public/src/.env')
    ap.add_argument('--library-sql', type=Path, default=Path(__file__).parent / 'db-init/45-library-decks.sql')
    a = ap.parse_args()
    env = dict(line.split('=', 1) for line in a.env.read_text().splitlines() if '=' in line and not line.startswith('#'))
    query = ("SELECT d.name, d.contents FROM deck d JOIN player p ON p.id = d.player_id "
             f"WHERE p.name = '{a.owner}' ORDER BY d.name")
    out = subprocess.run(['docker', 'exec', a.container, 'mariadb', '-u' + env['MYSQL_USER'], '-p' + env['MYSQL_PASSWORD'],
                          env['MYSQL_DATABASE'], '-N', '--raw', '-e', query], check=True, capture_output=True, text=True).stdout
    decks = [line.split('\t', 1) for line in out.splitlines() if line]
    library = a.library_sql.read_text(encoding='utf-8')
    print(f'-- Plan beta-server Amendment 2: the {len(decks)} decks of "{a.owner}" in the local Brainiac test rig '
          f'({a.container}), exported {time.strftime("%Y-%m-%d")} (deploy/beta/export_rig_decks.py), as library decks.')
    for name, contents in decks:
        quoted = "'" + name.replace('\\', '\\\\').replace("'", "''") + "'"
        if f'),{quoted},' in library:
            raise SystemExit(f'{name}: already in the official library')
        body = "'" + contents.replace('\\', '\\\\').replace("'", "''") + "'"
        print("INSERT INTO deck (player_id, name, contents) VALUES((SELECT id FROM player WHERE name='Librarian'),"
              f"{quoted},{body});")


if __name__ == '__main__':
    main()
