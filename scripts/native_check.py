#!/usr/bin/env python3
"""Gate: in-JVM policy path (Java PolicyTensors + libtorch/CUDA-graph forward) vs Python (parse_record + compact + PyTorch eager,
unpadded, batch 1, TF32 off) on the same recorded v4 decisions, at several Java batch sizes.
Pass: argmax 100 %, max relative logit difference <= 1e-3 (TF32 off on both sides).

    uv run python scripts/native_check.py [--n 1000] [--ckpt runs/rl-v4/ckpt/learner-0419.pt]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import orjson
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import spike_backends as sb  # noqa: E402
from swccg_policy import rl  # noqa: E402
from swccg_policy.data import card_index, parse_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/rl-v4/ckpt/learner-0419.pt"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    ap.add_argument("--bundle", default=str(ROOT / "native/bundle"), help="written from --ckpt and --cards before the check")
    ap.add_argument("--six", action="store_true", help="only decisions showing one of the six v2-only cards")
    a = ap.parse_args()
    sb.set_tf32(False)
    model, _, vocab, _ = rl.load_rl(a.ckpt)
    model.eval()
    from swccg_policy.native import export_ts
    ts = str(ROOT / "native/spike/check-tf32off.ts")  # exported with TF32 off (card table exact), unlike the learner's
    export_ts(model, ts)
    bp_idx = card_index(a.cards)
    from swccg_policy.native import write_bundle
    write_bundle(a.ckpt, a.bundle, a.cards)
    recs = sb.load_decisions(10 ** 9 if a.six else a.n, 5)
    if a.six:
        six = {"1_245", "3_159", "3_42", "3_77", "4_122", "8_1"}
        recs = [r for r in recs if ({t[1] for t in r[8]} | {o[3] for o in r[9] if o[3]} | set(r[6])) & six][:a.n]
    ref = []
    with torch.no_grad():
        for r in recs:
            ref.append(model(rl.stack([rl.compact(parse_record(r, bp_idx), vocab)], "cuda")).float().cpu().numpy()[0])
    res = {"n": len(recs), "ckpt": a.ckpt, "by_batch": {}}
    with tempfile.TemporaryDirectory() as td:
        rf = Path(td) / "recs.txt"
        rf.write_text("\n".join(orjson.dumps(r).decode() for r in recs) + "\n")
        for B in [int(x) for x in a.batches.split(",")]:
            of = Path(td) / f"out-{B}.jsonl"
            env = {"TRAINGEN_JOPTS": f"-Dpolicy.native.lib={ROOT}/native/build/libswccgpolicy.so"}
            cp = subprocess.run([str(ROOT / "scripts/traingen_main.sh"), "com.gempukku.swccgo.bench.NativeCheck", ts,
                                 a.bundle, str(rf), str(B), str(of), "false"],
                                capture_output=True, text=True, env={**__import__("os").environ, **env})
            if cp.returncode:
                print(cp.stdout[-3000:], cp.stderr[-3000:])
                sys.exit(1)
            got = [np.array(json.loads(l), dtype=np.float32) for l in of.read_text().splitlines()]
            agree, max_rel, n_opt_mismatch = 0, 0.0, 0
            for g, r in zip(got, ref):
                if len(g) != len(r):
                    n_opt_mismatch += 1
                    continue
                agree += int(np.argmax(g) == np.argmax(r))
                m = np.isfinite(r)
                max_rel = max(max_rel, float((np.abs(g[m] - r[m]) / np.maximum(np.abs(r[m]), 1.0)).max()))
            res["by_batch"][B] = {"argmax_agree": agree, "total": len(ref), "option_count_mismatch": n_opt_mismatch,
                                  "max_rel_diff": max_rel}
            print(B, res["by_batch"][B], flush=True)
    res["pass"] = all(v["argmax_agree"] == v["total"] and v["max_rel_diff"] <= 1e-3 for v in res["by_batch"].values())
    out = ROOT / f"reports/data/rollout-throughput/native-check{'-six-' + Path(a.cards).stem if a.six else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print("pass:", res["pass"])


if __name__ == "__main__":
    main()
