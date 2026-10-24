"""Plan v-cards §1.3: append the decision-prompt keys the policy did not know in the gathering games.

Reads the per-JVM unknown-key reports (VocabularyV4: {normalizer_version, families: {q, o, context: {key: count}}}) of
the gathering runs (their "normalizer_version" is a constant of VocabularyV4, always v1: the keys are in the form the
recording seat looked them up, so the runs must have had only seats of this bundle encoding), and appends every key seen at least --min-count
times as it is (the runtime form of the key) with BGE text priors, as scripts/build_v7_tables.py did. Existing rows and
IDs never change (a child bundle); the checkpoint grows into it with zero residual rows (grow_checkpoint_tables.py).
    PYTHONPATH=scripts:src .venv/bin/python scripts/vc_grow_prompts.py --tables BUNDLE --unknown DIR [DIR ...] \\
        --out NEW_BUNDLE --report REPORT [--min-count 2] [--families q o context]
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import PRIOR_MODEL, SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_vocabulary import FAMILIES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--unknown", type=Path, nargs="+", required=True, help="directories of unknown-key reports")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    parent = TableBundle.load(args.tables)
    normalizer = parent.vocabulary.payload.get("normalizer_version")
    counts = {family: Counter() for family in FAMILIES}
    files = 0
    labels = Counter()
    for directory in args.unknown:
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            labels[data.get("normalizer_version")] += 1
            files += 1
            for family in FAMILIES:
                counts[family].update(data["families"].get(family, {}))
    keep = {family: sorted(key for key, count in counts[family].items()
                           if count >= args.min_count and key not in parent.vocabulary.payload[family])
            if family in args.families else [] for family in FAMILIES}
    vocabulary = SoftVocabulary.build(keep, lambda texts: embed_texts(list(texts), PRIOR_MODEL),
                                      "v-cards-gathering", parent.vocabulary, normalizer=normalizer)
    bundle = TableBundle(parent.cards, vocabulary, parent.sha, (*parent.ancestors, parent.sha))
    parent.assert_prefix_of(bundle)
    bundle.save(args.out)
    report = {"parent_sha": parent.sha, "sha": bundle.sha, "normalizer": normalizer, "reports_read": files,
              "report_labels": dict(labels), "min_count": args.min_count, "families": args.families,
              "unknown_keys": {f: len(counts[f]) for f in FAMILIES},
              "unknown_occurrences": {f: sum(counts[f].values()) for f in FAMILIES},
              "added": {f: len(keep[f]) for f in FAMILIES},
              "added_occurrences": {f: sum(counts[f][k] for k in keep[f]) for f in FAMILIES},
              "below_min_count": {f: sum(1 for k, c in counts[f].items() if c < args.min_count) for f in FAMILIES},
              "top_added": {f: [[k, counts[f][k]] for k in sorted(keep[f], key=lambda k: -counts[f][k])[:25]]
                            for f in FAMILIES},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("reports_read", "report_labels", "unknown_keys",
                                             "unknown_occurrences", "added", "added_occurrences")}))


if __name__ == "__main__":
    main()
