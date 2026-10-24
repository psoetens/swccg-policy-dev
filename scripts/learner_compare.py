#!/usr/bin/env python3
"""Compare two learner results that started from the same checkpoint (plan learner-throughput, gates 3 and 4).

- update-relative parameter difference: ||theta_A - theta_B|| / ||theta_A - theta_start|| (policy, value head)
- policy divergence on held-out decisions (runs/bench/lt/heldout.txt): mean KL(pi_A || pi_B) and max, total variation, argmax
  agreement of the first pick, computed in fp32 with TF32 off.

    uv run python scripts/learner_compare.py START.pt A.pt B.pt [--n 5000] [--out FILE.json]
"""
import argparse
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


def heldout(n):
    recs = []
    for f in open(ROOT / "runs/bench/lt/heldout.txt").read().split():
        with gzip.open(f, "rb") as fh:
            for line in fh:
                recs += [s["rec"] for s in orjson.loads(line)["steps"]]
                if len(recs) >= n:
                    return recs[:n]
    return recs


def flat(state, prefix_skip=("cards.features", "cards.text")):
    return torch.cat([v.float().flatten() for k, v in sorted(state.items()) if not k.startswith(prefix_skip)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table_v2.npz"))
    ap.add_argument("--out")
    x = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cks = {k: torch.load(p, map_location="cpu", weights_only=False) for k, p in (("s", x.start), ("a", x.a), ("b", x.b))}
    res = {"start": x.start, "a": x.a, "b": x.b}
    for part in ("state", "value_state"):
        s, a, b = (flat(cks[k][part]) for k in ("s", "a", "b"))
        res[f"{part}_update_norm_a"] = float((a - s).norm())
        res[f"{part}_update_norm_b"] = float((b - s).norm())
        res[f"{part}_rel_diff"] = float((a - b).norm() / max((a - s).norm(), 1e-12))
    bp = card_index(x.cards)
    ma, _, vocab, _ = rl.load_rl(x.a)
    mb, _, _, _ = rl.load_rl(x.b)
    ma.eval(); mb.eval()
    recs = heldout(x.n)
    kls, tvs, agree = [], [], 0
    with torch.no_grad():
        for i in range(0, len(recs), 128):
            items = [rl.compact(parse_record(r, bp), vocab) for r in recs[i:i + 128]]
            bt = rl.stack(items, "cuda")
            la = torch.log_softmax(ma(bt).float(), 1)
            lb = torch.log_softmax(mb(bt).float(), 1)
            pa = la.exp()
            m = bt["omask"]
            kl = (pa * (la - lb)).masked_fill(~m, 0).sum(1)
            tv = 0.5 * (pa - lb.exp()).abs().masked_fill(~m, 0).sum(1)
            kls += kl.cpu().tolist(); tvs += tv.cpu().tolist()
            agree += int((la.argmax(1) == lb.argmax(1)).sum())
    kls, tvs = np.array(kls), np.array(tvs)
    res.update(decisions=len(recs), kl_mean=float(kls.mean()), kl_p99=float(np.percentile(kls, 99)), kl_max=float(kls.max()),
               tv_mean=float(tvs.mean()), argmax_agree=agree / len(recs))
    print(json.dumps(res, indent=1))
    if x.out:
        Path(x.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
