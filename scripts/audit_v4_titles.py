"""Report normalized templates and residual card titles without modifying inputs."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from v4_corpus_index import closed_batches, collect, index_batch


def title_like_bullet(key):
    # The rules notation "unique (•)" is a uniqueness symbol, not a card title.
    return "•" in key.replace("(•)", "")


INTEGER = re.compile(r"(?<![\w-])\d+(?![\w-])")


def residual_reasons(key, family, title_pattern, suffixes):
    reasons = []
    if title_pattern.search(key):
        reasons.append("card_title")
    without_exemptions = key.replace("(•)", "")
    without_exemptions = re.sub(r"(?:•+|<>|◇)<c>", "", without_exemptions)
    if re.search(r"•|<>|◇|&lt;&gt;", without_exemptions):
        reasons.append("unresolved_title_marker")
    if any("<c>" + suffix in key for suffix in suffixes):
        reasons.append("variant_suffix_after_reference")
    if family in ("q", "o") and INTEGER.search(key):
        reasons.append("standalone_integer")
    if re.search(r"\$\d+", key):
        reasons.append("anonymous_class_number")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--records", type=Path, nargs="+")
    inputs.add_argument("--index-run", type=Path, nargs="+", help="Use verified indices of closed batches")
    inputs.add_argument("--index-files", type=Path, nargs="+", help="Audit explicitly selected completed indices")
    parser.add_argument("--cards", type=Path, default=Path("data/cards/card_table_knowledge_v2.npz"))
    parser.add_argument("--titles", type=Path, default=Path("runs/knowledge-v3.1/titles-amendment2.tsv"))
    parser.add_argument("--title-report", type=Path,
        default=Path("reports/data/knowledge-v3.1/title-table-amendment2.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/data/knowledge-v3.1/title-audit.json"))
    args = parser.parse_args()
    cards = np.load(args.cards)
    all_titles = set(map(str, cards["titles"][4:]))
    all_titles.update(line.split("\t", 1)[1] for line in args.titles.read_text().splitlines())
    titles = {re.sub(r"^(?:•+|<>|◇)+", "", title) for title in all_titles if len(title.split()) >= 2}
    title_metadata = json.loads(args.title_report.read_text())
    suffixes = title_metadata["display_suffixes"]
    alternatives = "|".join(map(re.escape, sorted(titles, key=len, reverse=True)))
    title_pattern = re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)")
    names = {family: Counter() for family in ("q", "o", "context")}
    decisions = 0
    question_references = 0
    sources = {}
    for path in args.index_files or []:
        run = path.parent.parent
        expected = dict(closed_batches(run))[path.stem]
        indexed = index_batch(run, path.stem, expected)
        for family in names:
            names[family].update(indexed["families"][family])
        decisions += indexed["records"]
        question_references += indexed["question_references"]
        sources.update({str(run / source): value for source, value in indexed["sources"].items()})
    for run in args.index_run or []:
        indexed = collect(run)
        for family in names:
            names[family].update(indexed["families"][family])
        decisions += indexed["records"]
        question_references += indexed["question_references"]
        sources.update({str(run / path): value for path, value in indexed["sources"].items()})
    for path in args.records or []:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if record[0] != 4:
                    raise ValueError(f"Schema 4 required: {path}")
                names["q"][record[5]] += 1
                names["o"].update(option[0] for option in record[9])
                views = record[8] + (record[11] if len(record) >= 13 else [])
                names["context"].update(token[4] for token in views if token[0] == 7)
                question_references += sum(token[0] == 9 for token in record[8])
                decisions += 1
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    residual = {}
    for family, keys in names.items():
        residual[family] = [
            {"key": key, "count": count, "reasons": reasons}
            for key, count in keys.most_common()
            if (reasons := residual_reasons(key, family, title_pattern, suffixes))
        ]
    report = {
        "decisions": decisions,
        "question_references": question_references,
        "top_300": {family: keys.most_common(300) for family, keys in names.items()},
        "residual_titles_or_anonymous_names": residual,
        "literal_uniqueness_symbols": {
            family: [(key, count) for key, count in keys.most_common() if "(•)" in key]
            for family, keys in names.items()
        },
        "false_match_note": "The exact notation (•) denotes uniqueness; it is preserved, not a title reference.",
        "normalization_revision": 2,
        "context_integer_literals": [
            {"key": key, "count": count} for key, count in names["context"].most_common() if INTEGER.search(key)
        ],
        "integer_gate_scope": "B1c question/option text; context gate scope awaits user clarification of B2 conflict",
        "end_goal_4_passed": False,
        "pending_clarifications": ["integer gate scope for context", "literal diamond rules notation"],
        "title_table_sha256": hashlib.sha256(args.titles.read_bytes()).hexdigest(),
        "title_report_sha256": hashlib.sha256(args.title_report.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": not any(residual.values()),
        "inputs_sha256": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "residual_keys": {k: len(v) for k, v in residual.items()}}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
