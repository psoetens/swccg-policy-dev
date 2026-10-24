"""Export for the in-JVM (native, libtorch) policy backend (plan .claude/plans/rollout-throughput.md, step 2).

A checkpoint becomes a TorchScript file (<name>.ts): the policy forward on the flat bucket tensors of rl.GraphForward (tok,
tokf, tmask, opt, omask, head[B, 22]) returning float32 logits [B, O]. The card embedding table (which depends only on the
weights) is a named buffer "card_table", so the JVM can swap weights in place (copy every named parameter and buffer of a newly
loaded .ts into the live module; CUDA graphs captured on the live module stay valid).

The bundle directory (vocab.json, cards.json) holds the maps the Java side needs to build the tensors (as the RL server's
parse_record + compact: card index from --cards, template vocabularies from the checkpoint).

    uv run python -m swccg_policy.native ts runs/rl-v4/ckpt/learner-0419.pt [OUT.ts]
    uv run python -m swccg_policy.native bundle runs/rl-v4/ckpt/learner-0419.pt DIR [--cards data/cards/card_table.npz]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class FlatPolicy(torch.nn.Module):
    """Policy forward on the flat bucket tensors, with the card table as the named buffer "card_table"."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("tensor_abi", torch.tensor(getattr(model.cfg,"tensor_abi",1)))
        with torch.no_grad():
            table = model.cards.table()
        self.register_buffer("card_table", table.detach().clone())

    def forward(self, tok, tokf, tmask, opt, omask, head):
        self.m._card_table = self.card_table
        b = {"tok": tok, "tokf": tokf, "tmask": tmask, "opt": opt, "omask": omask, "dtype": head[:, 0], "qtmpl": head[:, 1],
             "qbp": head[:, 2], "side": head[:, 3], "glob": head[:, 4:22]}
        return self.m(b).float()

    def forward_packed(self, tok, tokf, tmask, opt, omask, head, edge_rows, question_rows, option_rows):
        self.m._card_table = self.card_table
        b = {"tok":tok,"tokf":tokf,"tmask":tmask,"opt":opt,"omask":omask,
             "dtype":head[:,0],"qtmpl":head[:,1],"qbp":head[:,2],"side":head[:,3],"glob":head[:,4:22],
             "edge_rows":edge_rows,"question_rows":question_rows,"option_rows":option_rows}
        return self.m(b).float()


def _example(device) -> tuple:
    """A small valid batch for tracing (shapes are traced symbolically: any bucket works afterwards)."""
    B, T, O = 2, 32, 8
    tok = torch.zeros(B, T, 8, dtype=torch.int16, device=device)
    tok[:, :, 1] = 2
    tokf = torch.full((B, T, 4), float("nan"), device=device)
    tmask = torch.zeros(B, T, dtype=torch.bool, device=device)
    tmask[:, :10] = True
    opt = torch.zeros(B, O, 6, dtype=torch.int32, device=device)
    omask = torch.zeros(B, O, dtype=torch.bool, device=device)
    omask[:, :3] = True
    head = torch.zeros(B, 22, dtype=torch.int32, device=device)
    return tok, tokf, tmask, opt, omask, head


def export_ts(model, path: str | Path) -> None:
    """Trace a copy of the policy (eval mode, no grad) and save it atomically as path (.ts).

    The card table buffer is computed under the caller's TF32 setting: rl.py enables TF32 globally, so the learner's exports
    carry the same table as the Python server's cache_card_table(); an exact (TF32-off) export must disable TF32 first."""
    if getattr(model.cfg, "tensor_abi", 1) in (4, 5):
        from .v4_native import export_ts as export_v4
        export_v4(model, path)
        return
    path = Path(path)
    device = next(model.parameters()).device
    m = copy.deepcopy(model).eval()
    m._card_table = None
    flat = FlatPolicy(m).eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # Fused eval MHA materializes quadratic attention; SDPA uses memory-efficient kernels.
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            example=_example(device)
            traced = torch.jit.trace(flat, example, check_trace=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    tmp = path.with_name(path.name + ".tmp")
    traced.save(str(tmp))
    os.replace(tmp, path)


def export_checkpoint(ck: str | Path, out: str | Path | None = None, device: str = "cuda") -> Path:
    from .train import load_checkpoint
    model, _ = load_checkpoint(ck, device)
    out = Path(out) if out else Path(ck).with_suffix(".ts")
    export_ts(model, out)
    return out


def write_bundle(ck: str | Path, out_dir: str | Path, cards: str | Path) -> None:
    ckd = torch.load(ck, map_location="cpu", weights_only=False)
    out_dir = Path(out_dir)
    if ckd["cfg"].get("schema_version") == 4:
        from .v4_checkpoint import unpack_tables
        from .trajectory import card_fingerprint
        tables = unpack_tables(ckd["tables"])
        tables.save(out_dir)
        (out_dir / "cards.json").write_text(json.dumps(tables.cards.index))
        manifest = {"schema_version": 4, "tensor_abi": ckd["cfg"].get("tensor_abi", 4), "model": "knowledge_v3_1",
                    "checkpoint": str(ck), "table_sha": tables.sha,
                    "card_map_sha": card_fingerprint(tables.cards.index).hex(),
                    "vocabulary_sha": tables.vocabulary.sha, "n_cards_index": len(tables.cards.ids)}
        (out_dir / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.load(cards)["ids"]
    (out_dir / "cards.json").write_text(json.dumps({str(b): i for i, b in enumerate(ids)}))
    (out_dir / "vocab.json").write_text(json.dumps(ckd["vocab"]))
    (out_dir / "bundle.json").write_text(json.dumps({"checkpoint": str(ck), "cards": str(cards), "n_cards_index": len(ids), "tensor_abi": ckd["cfg"].get("tensor_abi",1), "model": ckd["cfg"].get("model","legacy"), "schema_version": ckd["cfg"].get("schema_version", 1)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("ts")
    t.add_argument("ckpt", nargs="+")
    t.add_argument("--out")
    b = sub.add_parser("bundle")
    b.add_argument("ckpt")
    b.add_argument("dir")
    b.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    a = ap.parse_args()
    if a.cmd == "ts":
        for c in a.ckpt:
            print(export_checkpoint(c, a.out if len(a.ckpt) == 1 else None))
    else:
        write_bundle(a.ckpt, a.dir, a.cards)
        print(a.dir)


if __name__ == "__main__":
    main()
