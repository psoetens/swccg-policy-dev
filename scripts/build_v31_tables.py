"""Build the approved decipher soft vocabulary from audited game and static source batches."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_v4_vocabulary import actual_coverage, heldout_disjoint
from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import NORMALIZER, PRIOR_MODEL, SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_tables import CardTable
from swccg_policy.v4_vocabulary import FAMILIES
from v4_corpus_index import collect


def coverage(vocabulary, observed):
    result = {}
    for family in FAMILIES:
        counts = observed[family]
        unknown = Counter({key: count for key, count in counts.items() if key not in vocabulary.payload[family]})
        total = sum(counts.values())
        rate = sum(unknown.values()) / max(1, total)
        threshold = 0.02 if family == "context" else 0.01
        result[family] = {"lookups": total, "unknown": sum(unknown.values()), "rate": rate,
                          "threshold": threshold, "met": rate <= threshold,
                          "top_unknown": unknown.most_common(50)}
    return result


def scan_coverage(scan, observed):
    counts = observed["o"]
    covered = sum(count for key, count in counts.items() if key in scan["o"])
    return {"lookups": sum(counts.values()), "covered": covered,
            "rate": covered / max(1, sum(counts.values())),
            "distinct_observed": len(counts), "distinct_covered": len(set(counts) & set(scan["o"]))}


def load_normalized(path):
    data = json.loads(path.read_text())
    if data["normalizer_version"] != NORMALIZER:
        raise ValueError(f"Normalizer version mismatch: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/knowledge-v3.1/tables-decipher"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path("runs/knowledge-v3.1")
    reports = Path("reports/data/knowledge-v3.1")
    corpus = root / "decipher-seed"
    heldout = root / "decipher-heldout"
    roles = actual_coverage(corpus)
    if not roles["complete"]:
        raise ValueError("Incomplete decipher seed deck/role coverage")
    disjoint = heldout_disjoint(corpus, heldout)
    deck_sets = [{deck for line in (run / "schedule.tsv").read_text().splitlines()
                  for deck in line.split("\t")} for run in (corpus, heldout)]
    if deck_sets[0] & deck_sets[1]:
        raise ValueError("Decipher-only seed and held-out decks overlap")
    disjoint.update(decks_disjoint=True, seed_decks=len(deck_sets[0]), heldout_decks=len(deck_sets[1]))
    indexed = {name: collect(run) for name, run in (("corpus", corpus), ("heldout", heldout))}
    sources = {name: reports / filename for name, filename in (
        ("corpus", "decipher-corpus-seed.json"), ("static", "decipher-static-seed.json"),
        ("heldout", "decipher-heldout-normalized.json"), ("all_pool", "decipher-all-pool-heldout-normalized.json"))}
    data = {name: load_normalized(path) for name, path in sources.items()}
    for name, run in (("corpus", corpus), ("heldout", heldout)):
        expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((run / "template-index").glob("*.json"))}
        if data[name]["input_sha256"] != expected:
            raise ValueError(f"Normalized {name} does not cover the exact current indices")
    cards = CardTable.load(root / "decipher/cards.npz")
    if any(int(blueprint.split("_", 1)[0]) >= 200 for blueprint in cards.ids[4:]):
        raise ValueError("Virtual card in initial decipher table")
    vocabulary = None
    for name in ("corpus", "static"):
        vocabulary = SoftVocabulary.build(data[name]["families"],
                                          lambda texts: embed_texts(texts, PRIOR_MODEL),
                                          f"decipher-{name}", vocabulary)
    bundle = TableBundle(cards, vocabulary)
    bundle.save(args.out)
    report = {"table_sha": bundle.sha, "card_rows": len(cards.ids),
              "vocabulary_rows": {family: len(vocabulary.payload[family]) for family in FAMILIES},
              "deck_role_coverage": roles, "disjoint": disjoint,
              "decipher_heldout": coverage(vocabulary, data["heldout"]["observed_families"]),
              "all_pool_before_extension": coverage(vocabulary, data["all_pool"]["observed_families"]),
              "static_option_coverage": {name: scan_coverage(data["static"]["families"],
                                                           data[name]["observed_families"])
                                         for name in ("corpus", "heldout")},
              "dropped_seed_keys": {family: len(data["corpus"]["dropped"][family]) for family in FAMILIES},
              "records": {name: value["records"] for name, value in indexed.items()},
              "export_sha256": {name: value["sources"] for name, value in indexed.items()},
              "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources.values()},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "oov_note": "512 FNV-1a buckets per family can collide and are not identities.",
              "scope": "Tables and coverage only; full-policy growth and backend parity remain pending."}
    (reports / "decipher-vocabulary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("table_sha", "card_rows", "vocabulary_rows",
                                                  "static_option_coverage")}))


if __name__ == "__main__":
    main()
