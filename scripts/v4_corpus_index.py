"""Index normalized templates only after an engine batch has closed its exports."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import orjson

from swccg_policy.v4_vocabulary import FAMILIES


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def keys(record):
    if record[0] != 4:
        raise ValueError("Schema 4 required for normalized template indexing")
    yield "q", record[5]
    for option in record[9]:
        yield "o", option[0]
    for rows in (record[8], record[11] if len(record) >= 13 else []):
        for token in rows:
            if token[0] == 7:
                yield "context", token[4]


def closed_batches(run):
    for path in sorted((run / "logs").glob("*.csv")):
        log = path.with_suffix(".log")
        if not log.exists() or " games on " not in log.read_text():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        if any(row["outcome"] not in ("finished", "cap") or int(row["unmapped"]) for row in rows):
            raise ValueError(f"Invalid completed corpus batch: {path}")
        yield path.stem, sum(int(row["exported"]) for row in rows)


def index_batch(run, tag, expected_records):
    output = run / "template-index" / f"{tag}.json"
    paths = sorted((run / "exports" / tag).glob("*.jsonl.gz"))
    if not paths:
        raise ValueError(f"Missing exports for completed batch {tag}")
    sources = {str(path.relative_to(run)): digest(path) for path in paths}
    code_hash = digest(Path(__file__))
    if output.exists():
        result = json.loads(output.read_text())
        if result["sources"] != sources or result["indexer_sha256"] != code_hash:
            raise ValueError(f"Indexed source changed: {output}")
        if result["records"] != expected_records:
            raise ValueError(f"Indexed record count changed: {output}")
        return result
    counts = {family: Counter() for family in FAMILIES}
    decisions = 0
    references = 0
    for path in paths:
        with gzip.open(path, "rb") as stream:
            for line in stream:
                record = orjson.loads(line)
                for family, key in keys(record):
                    counts[family][key] += 1
                references += sum(token[0] == 9 for token in record[8])
                decisions += 1
    if decisions != expected_records:
        raise ValueError(f"CSV/export record count mismatch in {tag}: {decisions} != {expected_records}")
    result = {
        "records": decisions, "question_references": references, "families": counts,
        "sources": sources, "indexer_sha256": code_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"indexed": str(output), "records": decisions,
        "unique_keys": {family: len(values) for family, values in counts.items()}}), flush=True)
    return result


def collect(run):
    counts = {family: Counter() for family in FAMILIES}
    records = 0
    references = 0
    sources = {}
    for tag, expected in closed_batches(run):
        result = index_batch(run, tag, expected)
        for family in FAMILIES:
            counts[family].update(result["families"][family])
        records += result["records"]
        references += result["question_references"]
        sources.update(result["sources"])
    return {"families": counts, "records": records, "question_references": references, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    seen = set()
    while True:
        for tag, expected in closed_batches(args.run):
            if tag not in seen:
                index_batch(args.run, tag, expected)
                seen.add(tag)
        if not args.watch or (args.run / "STOP").exists():
            break
        manifest = json.loads((args.run / "manifest.json").read_text())
        closed_games = 0
        for tag in seen:
            with (args.run / "logs" / f"{tag}.csv").open() as stream:
                closed_games += sum(1 for _ in csv.DictReader(stream))
        if closed_games == manifest["planned_games"]:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
