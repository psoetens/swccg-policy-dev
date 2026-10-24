"""Plan multiselect-count: grow a table bundle (and optionally a checkpoint) by the Done option template.

PolicyAi answers card multi-selections whose count is a choice step by step and offers an explicit Done option ("stop with
the current selection"); its option template is StateEncoder.DONE (Java) = DONE below. The template is appended to the
option family "o" through the existing growth path (SoftVocabulary.build with a text prior, an exact prefix extension;
v4_checkpoint.load grows the model's TextPriorEmbedding), so every existing ID, prior and weight is unchanged and a grown
checkpoint computes its parent's function on every decision without a Done row (tests/test_multiselect_count.py).
Without the growth the Done row would fall into a hashed OOV bucket shared with other unknown templates.

    # a checkpoint, grown into its own tables + Done (optimizer state dropped, as grow_checkpoint_tables.py):
    PYTHONPATH=scripts:src .venv/bin/python scripts/add_done_template.py --checkpoint CKPT.pt --checkpoint-out OUT.pt
    # a table bundle directory (e.g. the next run's tables):
    PYTHONPATH=scripts:src .venv/bin/python scripts/add_done_template.py --tables DIR --out DIR2
"""
import argparse
import json
from pathlib import Path

DONE = "<done>Finish selection"


def grow_done(parent, embed):
    """The child bundle of `parent` (TableBundle) with DONE appended to the option family (no-op error if present)."""
    from swccg_policy.soft_vocabulary import SoftVocabulary
    from swccg_policy.table_bundle import TableBundle
    if DONE in parent.vocabulary.payload["o"]:
        raise ValueError("the tables already hold the Done template")
    vocabulary = SoftVocabulary.build({"o": [DONE]}, embed, "multiselect-count-done", parent.vocabulary)
    grown = TableBundle(parent.cards, vocabulary, parent.sha, (*parent.ancestors, parent.sha))
    parent.assert_prefix_of(grown)
    return grown


def text_prior(texts):
    from swccg_policy.cards import embed_texts
    from swccg_policy.soft_vocabulary import PRIOR_MODEL
    return embed_texts(texts, PRIOR_MODEL)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, help="parent table bundle directory")
    parser.add_argument("--out", type=Path, help="grown table bundle directory")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint to grow (its own tables are the parent)")
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.tables) or bool(args.checkpoint) != bool(args.checkpoint_out) \
            or bool(args.tables) != bool(args.out):
        parser.error("either --tables/--out or --checkpoint/--checkpoint-out")
    import torch
    from swccg_policy.table_bundle import TableBundle
    from swccg_policy.v4_checkpoint import load, save, unpack_tables
    if args.tables:
        parent = TableBundle.load(args.tables)
        grown = grow_done(parent, text_prior)
        grown.save(args.out)
    else:
        if args.checkpoint_out.exists():
            raise FileExistsError(args.checkpoint_out)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        parent = unpack_tables(payload["tables"])
        grown = grow_done(parent, text_prior)
        actor, critic, grown, _ = load(args.checkpoint, tables=grown, payload=payload)
        save(args.checkpoint_out, actor, critic, grown,
             {**payload.get("rl_meta", {}), "done_template_parent_sha": parent.sha,
              "optimizer_reset_from": str(args.checkpoint)})
    print(json.dumps({"parent_sha": parent.sha, "sha": grown.sha, "done_id": grown.vocabulary.payload["o"][DONE],
                      "o_rows": len(grown.vocabulary.payload["o"])}))


if __name__ == "__main__":
    main()
