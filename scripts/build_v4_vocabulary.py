"""Build frozen schema-4 vocabularies and measure separately supplied held-out games."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from swccg_policy.v4_vocabulary import FAMILIES, Vocabulary, fingerprint
from v4_corpus_index import collect


def actual_coverage(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text())
    schedule = [tuple(line.split("\t")) for line in (corpus / "schedule.tsv").read_text().splitlines()]
    coverage = Counter()
    games = set()
    for path in sorted((corpus / "logs").glob("*.csv")):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row["outcome"] not in ("finished", "cap") or int(row["unmapped"]):
                    raise ValueError(f"Invalid corpus game: {path}: {row['game']}")
                game = int(row["game"].rsplit("-", 1)[1])
                if game in games:
                    raise ValueError(f"Duplicate corpus game: {game}")
                if not 0 <= game < len(schedule) or (row["ds_deck"], row["ls_deck"]) != schedule[game]:
                    raise ValueError(f"Corpus game does not match its planned pairing: {path}: {game}")
                games.add(game)
                for side in ("ds", "ls"):
                    learner = side == ("ds" if game % 2 == 0 else "ls")
                    coverage[(row[f"{side}_deck"], "learner" if learner else "opponent")] += 1
    missing = [
        {**entry, "actual": coverage[(entry["deck"], entry["role"])]}
        for entry in manifest["planned_coverage"]
        if coverage[(entry["deck"], entry["role"])] < 2
    ]
    return {
        "games": len(games), "complete": not missing and games == set(range(manifest["planned_games"])),
        "missing": missing, "actual": [
            {"deck": deck, "role": role, "games": count} for (deck, role), count in sorted(coverage.items())
        ],
    }


def heldout_disjoint(corpus, heldout):
    training = json.loads((corpus / "manifest.json").read_text())
    testing = json.loads((heldout / "manifest.json").read_text())
    training_seeds = set(range(training["seed"], training["seed"] + training["planned_games"]))
    testing_seeds = set(range(testing["seed"], testing["seed"] + testing["planned_games"]))
    training_pairs = set((corpus / "schedule.tsv").read_text().splitlines())
    testing_pairs = set((heldout / "schedule.tsv").read_text().splitlines())
    if training_seeds & testing_seeds or training_pairs & testing_pairs:
        raise ValueError("Vocabulary corpus and held-out seeds/pairings overlap")
    actual = actual_coverage(heldout)
    if not actual["complete"]:
        raise ValueError("Held-out games are incomplete")
    return {"seeds_disjoint": True, "pairings_disjoint": True, "games": actual["games"]}


def main():
    raise RuntimeError("Superseded by Amendment 3: use scripts/build_v31_tables.py; do not freeze the old corpus")


def archived_amendment2_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True, help="Separate held-out run with manifest and logs")
    parser.add_argument("--out", type=Path, default=Path("runs/knowledge-v3.1/vocab.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data/knowledge-v3.1/vocabulary.json"))
    parser.add_argument("--audit", type=Path, required=True, help="Passing amended audit of these exact exports")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifests = [json.loads((run / "manifest.json").read_text()) for run in (args.corpus, args.heldout)]
    if any(manifest.get("normalization_revision") != 2 for manifest in manifests):
        raise ValueError("Only the Amendment 2 regeneration may freeze a vocabulary")
    if manifests[0]["title_table_sha256"] != manifests[1]["title_table_sha256"]:
        raise ValueError("Corpus and held-out title normalization differ")
    audit = json.loads(args.audit.read_text())
    if not audit.get("passed") or not audit.get("end_goal_4_passed"):
        raise ValueError("Full amended title/integer audit has not passed")
    coverage = actual_coverage(args.corpus)
    if not coverage["complete"]:
        raise ValueError(f"Corpus deck/role coverage incomplete: {coverage['games']} games")
    disjoint = heldout_disjoint(args.corpus, args.heldout)
    paths = sorted((args.corpus / "exports").rglob("*.jsonl.gz"))
    heldout_paths = sorted((args.heldout / "exports").rglob("*.jsonl.gz"))
    if not paths or not heldout_paths:
        raise ValueError("Corpus or held-out exports are empty")
    if {path.resolve() for path in paths} & {path.resolve() for path in heldout_paths}:
        raise ValueError("Vocabulary and held-out inputs overlap")
    training = collect(args.corpus)
    testing = collect(args.heldout)
    expected_sources = {}
    for run, indexed in ((args.corpus, training), (args.heldout, testing)):
        expected_sources.update({str((run / path).resolve()): digest for path, digest in indexed["sources"].items()})
    audited_sources = {str(Path(path).resolve()): digest for path, digest in audit["inputs_sha256"].items()}
    if audited_sources != expected_sources or audit["title_table_sha256"] != manifests[0]["title_table_sha256"]:
        raise ValueError("Passing audit does not cover these exact exports/title table")
    payload = Vocabulary.build(**training["families"])
    vocabulary = Vocabulary(payload)
    for family, counts in testing["families"].items():
        for key, count in counts.items():
            vocabulary.lookup(family, key, count=count)
    telemetry = vocabulary.coverage()
    for family, statistics in telemetry.items():
        statistics["rate"] = statistics["unknown"] / max(1, statistics["total"])
        statistics["threshold"] = 0.02 if family == "context" else 0.01
        statistics["passed"] = statistics["rate"] <= statistics["threshold"]
    report = {
        "corpus_records": training["records"], "heldout_records": testing["records"],
        "deck_role_coverage": coverage,
        "disjoint": disjoint,
        "heldout": telemetry, "passed": all(item["passed"] for item in telemetry.values()),
        "sizes": {family: len(payload[family]) for family in FAMILIES},
        "vocabulary_sha256": fingerprint(payload).hex(), "oov_buckets": 512, "hash": "fnv1a32-utf8",
        "oov_note": "OOV buckets can collide and are not identities.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "export_sha256": {"corpus": training["sources"], "heldout": testing["sources"]},
        "title_audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "normalization_revision": 2,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "sizes": report["sizes"], "heldout": telemetry}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
