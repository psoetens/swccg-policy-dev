"""Manually append audited vocabulary/card rows between stages; existing IDs never change."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from normalize_v31_seed import normalize
from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import NORMALIZER, PRIOR_MODEL, SoftVocabulary
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_vocabulary import FAMILIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--unknown-keys", type=Path, nargs="+", required=True)
    parser.add_argument("--cards", type=Path, required=True,
                        help="Printed rows prepared with cards.py's feature/text schema")
    parser.add_argument("--titles", type=Path, required=True)
    parser.add_argument("--title-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional parent checkpoint to extend with the new tables")
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.checkpoint_out):
        raise ValueError("Supply both --checkpoint and --checkpoint-out")
    if args.checkpoint_out and args.checkpoint_out.exists():
        raise FileExistsError(args.checkpoint_out)
    parent = TableBundle.load(args.tables)
    counts = {family: Counter() for family in FAMILIES}
    for path in args.unknown_keys:
        data = json.loads(path.read_text())
        if data.get("normalizer_version") != NORMALIZER:
            raise ValueError(f"Unknown-key normalizer mismatch: {path}")
        for family in FAMILIES:
            counts[family].update(data["families"].get(family, {}))
    normalized = normalize(counts, args.titles, args.title_report, args.out.with_name(args.out.name + "-normalization"))
    vocabulary = SoftVocabulary.build(normalized["families"], lambda texts: embed_texts(texts, PRIOR_MODEL),
                                      "manual-growth", parent.vocabulary)
    cards = parent.cards.grow(dict(np.load(args.cards)))
    grown = TableBundle(cards, vocabulary, parent.sha, (*parent.ancestors, parent.sha))
    parent.assert_prefix_of(grown)
    grown.save(args.out)
    if args.checkpoint:
        from swccg_policy.v4_checkpoint import load, save
        actor, critic, _, checkpoint = load(args.checkpoint, tables=grown)
        save(args.checkpoint_out, actor, critic, grown,
             {**checkpoint.get("rl_meta", {}), "manual_growth_parent_sha": parent.sha})
    report = {"parent_sha": parent.sha, "sha": grown.sha, "card_rows_added": len(cards.ids) - len(parent.cards.ids),
              "vocabulary_rows_added": {family: len(vocabulary.payload[family]) - len(parent.vocabulary.payload[family])
                                        for family in FAMILIES},
              "dropped": normalized["dropped"],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prior_precision": "float16 normalized embeddings; projection in model precision; "
                                 "no runtime text encoder"}
    (args.out / "growth-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "dropped"}))


if __name__ == "__main__":
    main()
