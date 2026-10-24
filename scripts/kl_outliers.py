#!/usr/bin/env python3
"""Where does a high approx_kl_old come from? For the consumed batch of an RL run, per recorded step: d = old_logp (actor, as
recorded) - new_logp (the learner's recomputation with the weights the iteration started from), once in fp32 and once in bf16
autocast (as --bf16 does). Reports mean / tails, the steps that dominate the mean, by decision type, by the largest |logit| of
the decision, and by how many learner versions old the acting weights were (the trajectory's "learner" label vs the start).

    uv run python scripts/kl_outliers.py RUN_DIR START_CKPT [--cards data/cards/card_table_v2.npz]
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import orjson
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swccg_policy import rl  # noqa: E402
from swccg_policy.data import card_index, parse_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("start")
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table_v2.npz"))
    ap.add_argument("--out")
    x = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # as the learner runs (rl.py enables TF32)
    model, _, vocab, ck = rl.load_rl(x.start)
    model.eval()
    it_start = ck["rl_meta"]["iteration"]
    bp = card_index(x.cards)
    steps, labels = [], []
    for f in sorted((Path(x.run) / "traj-consumed").glob("part-*.jsonl.gz")):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                g = orjson.loads(line)
                lab = g.get("learner", "")
                v = int(lab.split(":")[1]) if ":" in lab else -1
                for s in g["steps"]:
                    steps.append(s)
                    labels.append(v)
    res = {"run": x.run, "start": x.start, "start_iteration": it_start, "steps": len(steps)}
    out = {}
    maxlogit = []
    for mode in ("fp32", "bf16"):
        lps = []
        for i in range(0, len(steps), 128):
            chunk = steps[i:i + 128]
            b = rl.stack([rl.compact(parse_record(s["rec"], bp), vocab) for s in chunk], "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "bf16")):
                lg = model(b)
            lg = lg.float()
            K = max(len(s["seq"]) for s in chunk)
            seq = torch.zeros(len(chunk), K, dtype=torch.long, device="cuda")
            sl = torch.tensor([len(s["seq"]) for s in chunk], device="cuda")
            ex = torch.zeros_like(b["omask"])
            for j, s in enumerate(chunk):
                seq[j, :len(s["seq"])] = torch.tensor(s["seq"])
                for o in s.get("x") or []:
                    if o < ex.shape[1]:
                        ex[j, o] = True
            lps += rl.seq_logprob(lg, b["omask"], b["opt"][:, :, 5] > 0, seq, sl, ex, sync_free=True).cpu().tolist()
            if mode == "fp32":
                maxlogit += lg.masked_fill(~b["omask"], 0).abs().max(1).values.cpu().tolist()
        out[mode] = np.array([s["logp"] for s in steps]) - np.array(lps)
    maxlogit = np.array(maxlogit)
    types = np.array([s["rec"][4] for s in steps])
    age = np.array([it_start - v if v >= 0 else -1 for v in labels])
    for mode, d in out.items():
        order = np.argsort(-d)
        top = order[: max(1, len(d) // 1000)]
        r = {"mean": float(d.mean()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max()),
             "share_of_mean_from_top_0.1pct": float(d[top].sum() / d.sum()) if d.sum() > 0 else None,
             "top10": [{"d": round(float(d[i]), 2), "type": str(types[i]), "max_abs_logit": round(float(maxlogit[i]), 1),
                        "age": int(age[i])} for i in order[:10]],
             "by_type_mean": {t: float(d[types == t].mean()) for t in sorted(set(types))},
             "by_max_logit_mean": {k: float(d[m].mean()) for k, m in (("<100", maxlogit < 100), ("100-1000", (maxlogit >= 100) & (maxlogit < 1000)),
                                                                       (">=1000", maxlogit >= 1000)) if m.any()},
             "by_age_mean": {int(a): float(d[age == a].mean()) for a in sorted(set(age.tolist()))}}
        res[mode] = r
    res["age_counts"] = {int(a): int((age == a).sum()) for a in sorted(set(age.tolist()))}
    res["max_logit_counts"] = {"<100": int((maxlogit < 100).sum()), "100-1000": int(((maxlogit >= 100) & (maxlogit < 1000)).sum()),
                               ">=1000": int((maxlogit >= 1000).sum())}
    print(json.dumps(res, indent=1))
    if x.out:
        Path(x.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
