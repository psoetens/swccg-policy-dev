"""Browser-deployability check (plan rl-selfplay-v1 §1.4): export the policy (option logits) to ONNX and compare
onnxruntime (CPU) with PyTorch on recorded decisions.

    uv run python -m swccg_policy.onnx_check --checkpoint runs/rl-v1/ckpt/latest.pt --packed data/packed/v2-eval_seen \
        --out runs/rl-v1/policy.onnx --n 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import Packed, load_checkpoint

INPUTS = ["tok", "tokf", "tmask", "opt", "omask", "dtype", "qtmpl", "qbp", "side", "glob"]


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tok, tokf, tmask, opt, omask, dtype, qtmpl, qbp, side, glob):
        b = {"tok": tok, "tokf": tokf, "tmask": tmask, "opt": opt, "omask": omask, "dtype": dtype, "qtmpl": qtmpl,
             "qbp": qbp, "side": side, "glob": glob}
        return self.model(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--cards", default="data/cards/card_table_knowledge_v2.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--exporter", choices=["dynamo", "legacy"], default="dynamo")
    a = ap.parse_args()
    import onnxruntime as ort
    torch.set_num_threads(2)
    model, vocab = load_checkpoint(a.checkpoint, "cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    model.cache_card_table()
    w = Wrapper(model).eval()
    p = Packed(a.packed)
    ex = p.batch(np.arange(0, min(4,a.n)), "cpu")
    args = tuple(ex[k] for k in INPUTS)
    dyn = {"tok": {0: "B", 1: "T"}, "tokf": {0: "B", 1: "T"}, "tmask": {0: "B", 1: "T"}, "opt": {0: "B", 1: "O"},
           "omask": {0: "B", 1: "O"}, "dtype": {0: "B"}, "qtmpl": {0: "B"}, "qbp": {0: "B"}, "side": {0: "B"}, "glob": {0: "B"},
           "logits": {0: "B", 1: "O"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.exporter == "dynamo":
        B, T, O = torch.export.Dim("B", min=1, max=1024), torch.export.Dim("T", min=1, max=320), torch.export.Dim("O", min=1, max=100000)
        shapes = {"tok": {0: B, 1: T}, "tokf": {0: B, 1: T}, "tmask": {0: B, 1: T}, "opt": {0: B, 1: O}, "omask": {0: B, 1: O},
                  "dtype": {0: B}, "qtmpl": {0: B}, "qbp": {0: B}, "side": {0: B}, "glob": {0: B}}
        prog = torch.onnx.export(w, args, input_names=INPUTS, output_names=["logits"], dynamic_shapes=shapes, dynamo=True,
                                 opset_version=18)
        prog.save(a.out)
    else:
        torch.onnx.export(w, args, a.out, input_names=INPUTS, output_names=["logits"], dynamic_axes=dyn, opset_version=17,
                          dynamo=False)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    max_diff, max_rel, max_logit, agree, n = 0.0, 0.0, 0.0, 0, 0
    for i in range(a.n):
        b = p.batch(np.array([i]), "cpu")
        with torch.no_grad():
            ref = model(b).numpy()
        feed = {k: b[k].numpy() for k in INPUTS}
        got = sess.run(["logits"], feed)[0]
        m = b["omask"].numpy() & np.isfinite(ref)
        d = np.abs(ref[m] - got[m])
        max_diff = max(max_diff, float(d.max()))
        max_rel = max(max_rel, float((d / np.maximum(np.abs(ref[m]), 1.0)).max()))
        max_logit = max(max_logit, float(np.abs(ref[m]).max()))
        agree += int(np.argmax(np.where(m, ref, -1e30)) == np.argmax(np.where(m, got, -1e30)))
        n += 1
    res = {"checkpoint": a.checkpoint, "params": n_params, "onnx_bytes": Path(a.out).stat().st_size, "decisions": n,
           "max_abs_diff": max_diff, "max_rel_diff": max_rel, "max_abs_logit": max_logit, "argmax_agreement": agree / n,
           "pass_abs_1e-4": max_diff <= 1e-4, "pass_rel_1e-5": max_rel <= 1e-5}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
