"""Golden fixture for the Java port of the policy input (Gemp: ai/brainiac/BrainiacTensors): records from TrainGen shards and
the tensors the training code builds from them (data.parse_record + rl.compact), plus the vocabularies and card index.

    uv run python scripts/brainiac_golden.py --checkpoint runs/rl-v1/final/rl-final.pt --shards data/shards/v2-eval_seen \
        --n 100 --out /tmp/golden [--subset]

--subset writes only the vocabulary and card entries the records use (a small fixture for the Gemp unit test); the lookups of
all other entries give the same defaults either way.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
from pathlib import Path

import numpy as np
import orjson
import torch

from swccg_policy.data import card_index, parse_record
from swccg_policy.rl import compact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--cards", default="data/cards/card_table.npz")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--stride", type=int, default=97, help="take every stride-th record, for variety")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", action="store_true")
    a = ap.parse_args()
    vocab = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["vocab"]
    bp_idx = card_index(a.cards)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, i = [], 0
    for f in sorted(glob.glob(f"{a.shards}/*.jsonl.gz")):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if i % a.stride == 0:
                    lines.append(line.decode().rstrip("\n"))
                i += 1
                if len(lines) >= a.n:
                    break
        if len(lines) >= a.n:
            break
    # V1 truncation regression; knowledge schemas instead preserve a large choice
    # list and valid token routes without relabeling or truncating their states.
    r = orjson.loads(lines[0])
    if r[0]>=2:
        r[8]=[t for t in r[8] if not (t[0]==10 or t[0]==9 and t[8]>0)]
        r[9]=[['<pass>',[],-1,'','',None] for _ in range(1031)]
        r[10]=[]
    else:
        toks = [list(t) for t in (r[8] * (330 // max(1, len(r[8])) + 1))[:330]]
        toks[5][5], toks[6][5], toks[7][5] = 325, 319, 320
        r[8] = toks
        r[9] = [list(o) for o in r[9]] + [["<card>", [], 325, None, "x", None], ["<card>", [], 319, None, "y", None]]
    lines.append(orjson.dumps(r).decode())
    used_q, used_o, used_c = set(), set(), set()
    with open(out / "records.jsonl", "w") as rf, open(out / "expected.jsonl", "w") as ef:
        for line in lines:
            r = orjson.loads(line)
            c = compact(parse_record(r, bp_idx), vocab)
            # rl.stack for a batch of one: parent references beyond the kept tokens become 0
            c["tok"][:, 5][c["tok"][:, 5] > max(1, c["tok"].shape[0])] = 0
            tokf = [[None if math.isnan(x) else float(x) for x in row] for row in c["tokf"].astype(np.float32).tolist()]
            rf.write(line + "\n")
            ef.write(json.dumps({"tok": c["tok"].tolist(), "tokf": tokf, "opt": c["opt"].tolist(), "head": c["head"].tolist()}) + "\n")
            used_q.add(r[5])
            used_c.update(x for x in r[6] if x)
            for t in r[8]:
                used_c.add(t[1])
            for o in r[9]:
                used_o.add(o[0])
                used_c.update(x for x in o[1] if x)
                if o[3]:
                    used_c.add(o[3])
    q, o = vocab["q"], vocab["o"]
    cards = bp_idx
    if a.subset:
        q = {k: v for k, v in q.items() if k in used_q}
        o = {k: v for k, v in o.items() if k in used_o}
        cards = {k: v for k, v in bp_idx.items() if k in used_c}
    (out / "vocab.json").write_text(json.dumps({"q": q, "o": o}))
    (out / "cards.json").write_text(json.dumps(cards))
    print(f"{len(lines)} records -> {out} (q {len(q)}, o {len(o)}, cards {len(cards)})")


if __name__ == "__main__":
    main()
