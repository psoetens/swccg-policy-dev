"""Plan actor-v6-inputs: the persistent list of unknown vocabulary keys seen during play.

Every encoding JVM writes its unknown question / option / context keys (cumulative counts) to
runs/unknown-keys/<host>-<pid>.json (VocabularyV4). `merge` folds them into data/vocab/unknown-keys.tsv
(family, key, count, first_seen, last_seen, sources); a file is merged by the change since its last merge
(data/vocab/unknown-keys.state.json), so merging again, also while a JVM still runs, never double counts.
`--tables` drops keys that a (grown) table bundle now contains. A vocabulary update reads the list:

    .venv/bin/python scripts/unknown_keys.py merge [--source runs/unknown-keys] [--tables runs/knowledge-v3.1/tables-v6]
    .venv/bin/python scripts/unknown_keys.py export --out keys.json [--min-count 2]   # {"q": {key: count}, ...}
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data/vocab/unknown-keys.tsv"
STATE = ROOT / "data/vocab/unknown-keys.state.json"
FIELDS = ("family", "key", "count", "first_seen", "last_seen", "sources")


def read_list(path=None):
    path = path or LIST
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        return {(row["family"], row["key"]): {**row, "count": int(row["count"])} for row in rows}


def write_list(rows, path=None):
    path = path or LIST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writeheader()
        for (family, key), row in sorted(rows.items(), key=lambda item: (item[0][0], -item[1]["count"], item[0][1])):
            writer.writerow({name: row[name] for name in FIELDS})
    temporary.replace(path)


def merge(sources, tables=None):
    rows = read_list()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    today = datetime.date.today().isoformat()
    added = 0
    for path in sorted(path for source in sources for path in Path(source).glob("*.json")):
        report = json.loads(path.read_text())
        seen = state.setdefault(path.name, {})
        for family, keys in report.get("families", {}).items():
            for key, count in keys.items():
                delta = count - seen.get(f"{family}\t{key}", 0)
                if delta <= 0:
                    continue
                seen[f"{family}\t{key}"] = count
                row = rows.setdefault((family, key), {"family": family, "key": key, "count": 0, "first_seen": today,
                                                      "last_seen": today, "sources": ""})
                row["count"] += delta
                row["last_seen"] = today
                names = set(filter(None, row["sources"].split(",")))
                row["sources"] = ",".join(sorted(names | {path.stem.rsplit("-", 1)[0]}))
                added += delta
    if tables:
        vocabulary = json.loads((Path(tables) / "vocab.json").read_text())
        rows = {key: row for key, row in rows.items() if key[1] not in vocabulary.get(key[0], {})}
    write_list(rows)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"lookups_added": added, "keys": len(rows),
                      "by_family": {family: sum(1 for key in rows if key[0] == family)
                                    for family in sorted({key[0] for key in rows})}}))


def export(out, min_count):
    result = {}
    for (family, key), row in read_list().items():
        if row["count"] >= min_count:
            result.setdefault(family, {})[key] = row["count"]
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({family: len(keys) for family, keys in result.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    merging = commands.add_parser("merge")
    merging.add_argument("--source", nargs="+", default=[str(ROOT / "runs/unknown-keys")])
    merging.add_argument("--tables", help="drop keys this table bundle contains")
    exporting = commands.add_parser("export")
    exporting.add_argument("--out", required=True)
    exporting.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "merge":
        merge(args.source, args.tables)
    else:
        export(args.out, args.min_count)


if __name__ == "__main__":
    main()
