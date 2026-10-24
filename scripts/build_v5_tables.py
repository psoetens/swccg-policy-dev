"""Plan context-overhaul: the context-version-5 table bundle (runs/knowledge-v3.1/tables-v5).

Same cards and question / option templates as tables-decipher (same batches, IDs and text priors); the context family
is replaced by the typed version-5 keys (StoredContextV5.keys(), written by its main) plus the shared battle.* /
drain.* rows, and the vocabulary carries context_version 5 (the Java tensor builder keeps only that version's rows).

    bash scripts/efficient_java_check.sh com.gempukku.swccgo.game.knowledge.StoredContextV5 \\
        reports/data/context-overhaul/context-v5-keys.json
    PYTHONPATH=scripts:src .venv/bin/python scripts/build_v5_tables.py

Plan actor-v6-inputs (tables-v6): --context-buckets 0 builds the context family without OOV rows, and --extra-keys
appends a last batch of question / option keys (a JSON {"q": {key: count}, "o": {...}}, e.g. the pool-v2 census):

    PYTHONPATH=scripts:src .venv/bin/python scripts/build_v5_tables.py --context-buckets 0 \
        --extra-keys runs/dojo/v6-census/unknown-qo.json --out runs/knowledge-v3.1/tables-v6
"""
import argparse
import json
from pathlib import Path

import numpy as np

from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import PRIOR_MODEL, SoftVocabulary, fingerprint, readable_text
from swccg_policy.v4_vocabulary import OOV_BUCKETS
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_vocabulary import FAMILIES

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_VERSION = 5
SHARED = ("battle.", "drain.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-decipher")
    parser.add_argument("--keys", type=Path, default=ROOT / "reports/data/context-overhaul/context-v5-keys.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-v5")
    parser.add_argument("--context-buckets", type=int, default=OOV_BUCKETS, help="context OOV rows (0 for tables-v6)")
    parser.add_argument("--extra-keys", type=Path, help="JSON of new q / o keys, appended as a last batch")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/data/context-overhaul/tables-v5.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = TableBundle.load(args.source)
    old = source.vocabulary
    keys = json.loads(args.keys.read_text())
    shared = sorted(key for key in old.payload["context"] if key.startswith(SHARED))
    context = sorted(set(keys) | set(shared))
    if any(not key.startswith(("now.", "mod.", "priv.") + SHARED) for key in context):
        raise ValueError("Unexpected context key family")

    # The question and option rows keep their priors exactly; only the new context keys are embedded.
    known = {(family, key): old.priors[family][index]
             for family in ("q", "o") for key, index in old.payload[family].items()}
    fresh = embed_texts([readable_text("context", key) for key in context], PRIOR_MODEL)
    known.update({("context", key): np.asarray(vector, np.float16) for key, vector in zip(context, fresh)})

    def batch_keys(batch, family):
        start, end = batch[family]
        return [key for key, index in old.payload[family].items() if start <= index < end]

    vocabulary = None
    for number, batch in enumerate(old.payload["batches"]):
        families = {family: batch_keys(batch, family) for family in ("q", "o")}
        families["context"] = context if number == 0 else []
        # build() embeds each family's sorted additions in FAMILIES order; answer from that family's own rows (a
        # question and an option can share a text but not a prior).
        pending = [family for family in FAMILIES if families[family]]

        def embed(texts, families=families, pending=pending):
            family = pending.pop(0)
            rows = {readable_text(family, key): known[(family, key)] for key in families[family]}
            return np.stack([rows[value] for value in texts])
        vocabulary = SoftVocabulary.build(families, embed, f"context-v5-{batch['source']}", vocabulary,
                                          context_buckets=args.context_buckets)
        if number == 0:
            payload = dict(vocabulary.payload, context_version=CONTEXT_VERSION)
            payload["sha"] = fingerprint(payload)
            vocabulary = SoftVocabulary(payload, vocabulary.priors)
    for family in ("q", "o"):
        if vocabulary.payload[family] != old.payload[family] or not np.array_equal(vocabulary.priors[family],
                                                                                     old.priors[family]):
            raise ValueError(f"Template family {family} differs from the source tables")
    added = {"q": 0, "o": 0}
    if args.extra_keys:
        extra = json.loads(args.extra_keys.read_text())
        families = {family: sorted(set(extra.get(family, {})) - set(vocabulary.payload[family]))
                    for family in ("q", "o")}
        families["context"] = []
        added = {family: len(families[family]) for family in ("q", "o")}
        vocabulary = SoftVocabulary.build(families, lambda texts: embed_texts(list(texts), PRIOR_MODEL),
                                          f"census-{args.extra_keys.parent.name}", vocabulary)
    bundle = TableBundle(source.cards, vocabulary)
    bundle.save(args.out)
    summary = {"table_sha": bundle.sha, "context_version": CONTEXT_VERSION, "context_buckets": args.context_buckets,
               "added_keys": added,
               "context_rows": len(vocabulary.payload["context"]), "source_context_rows": len(old.payload["context"]),
               "families": {prefix: sum(1 for key in context if key.startswith(prefix))
                            for prefix in ("now.action.", "now.play.", "now.target", "now.respond.", "now.result.",
                                           "mod.", "priv.", "battle.", "drain.")}}
    out = args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
