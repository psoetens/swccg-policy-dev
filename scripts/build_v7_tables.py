"""Plan titles-and-learner-ablation A: tables-v7 = tables-v6 + the keys of marked card names, normalizer v2.

Same cards, context and existing rows (IDs and text priors unchanged: a child bundle). Appended as one batch:
- every question / option template of a marked export (scripts/name_marks_check.py export) that tables-v6 lacks;
- the server's unknown keys (data/vocab/unknown-keys.tsv, today's encoding) seen at least --min-count times, as they
  are (literal names in card code keep that form) and with each one-word card title replaced by <c> (the form a
  marked name takes). Rows a game never meets cost only their prior vector.
The bundle states normalizer v2, so only a seat encoding inside NameMarks.raw uses it; v1 bundles are unchanged.

    PYTHONPATH=scripts:src .venv/bin/python scripts/build_v7_tables.py --export runs/name-marks/export-100 \\
        --out runs/knowledge-v3.1/tables-v7
"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re

from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import NORMALIZER_V2, PRIOR_MODEL, SoftVocabulary
from swccg_policy.table_bundle import TableBundle

ROOT = Path(__file__).resolve().parents[1]


def export_templates(folder):
    counts = {"q": Counter(), "o": Counter()}
    for path in sorted((folder / "export").glob("shard-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                counts["q"][record[5]] += 1
                counts["o"].update(option[0] for option in record[9])
    return counts


def one_word_titles(path):
    titles = set()
    for line in path.read_text().splitlines():
        title = re.sub(r"^(?:•+|<>|◇)+", "", line.split("\t")[1].replace("&lt;&gt;", "<>"))
        if title and not title.startswith("<") and len(title.split()) == 1:
            titles.add(title)
    return titles


def marked_form(key, titles):
    """Each one-word title, standing as a word, becomes <c> (the template a marked name produces)."""
    def replace(match):
        return "<c>" if match.group(0) in titles else match.group(0)
    return re.sub(r"(?<![\w-])[\w][\w'-]*(?![\w-])", replace, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-v6")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--unknown", type=Path, default=ROOT / "data/vocab/unknown-keys.tsv")
    parser.add_argument("--titles", type=Path, default=ROOT / "runs/knowledge-v3.1/decipher/titles.tsv")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-v7")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "reports/data/titles-and-learner-ablation/tables-v7.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = TableBundle.load(args.source)
    old = source.vocabulary
    exported = export_templates(args.export)
    titles = one_word_titles(args.titles)
    with args.unknown.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
                if row["family"] in ("q", "o") and int(row["count"]) >= args.min_count]
    families, origin = {"q": set(), "o": set(), "context": set()}, Counter()
    for family in ("q", "o"):
        for key in exported[family]:
            if key not in old.payload[family]:
                families[family].add(key)
                origin["export"] += 1
    for row in rows:
        for key in {row["key"], marked_form(row["key"], titles)}:
            if key not in old.payload[row["family"]] and key not in families[row["family"]]:
                families[row["family"]].add(key)
                origin["unknown-list" if key == row["key"] else "unknown-list-marked"] += 1
    vocabulary = SoftVocabulary.build({family: sorted(keys) for family, keys in families.items()},
                                      lambda texts: embed_texts(list(texts), PRIOR_MODEL),
                                      f"name-marks-{args.export.name}", old, normalizer=NORMALIZER_V2)
    # A declared child of the source (plan night-integer-fp32): a checkpoint on the source grows into it.
    bundle = TableBundle(source.cards, vocabulary, source.sha, (*source.ancestors, source.sha))
    bundle.save(args.out)
    summary = {"table_sha": bundle.sha, "source_sha": source.sha, "normalizer": NORMALIZER_V2,
               "added_keys": {family: len(keys) for family, keys in families.items() if family != "context"},
               "added_by_origin": dict(origin), "unknown_list_rows_used": len(rows), "min_count": args.min_count,
               "export_template_lookups": {family: sum(counts.values()) for family, counts in exported.items()}}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
