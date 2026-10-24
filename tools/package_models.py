#!/usr/bin/env python3
"""Package lab checkpoints into models/<name>/policy.pt (plan dev-distribution).

    python tools/package_models.py --name D --source ~/src/swccg-policy/runs/critic-noise/D.pt --slim

--slim drops the optimizer state (Adam moments, ~172 MB): RL continues from the weights with fresh moments.
Every other entry (weights, embedded tables / vocabulary, rl_meta) is kept and checked tensor-for-tensor against
the source. Appends name, sha256 and size to models/models.tsv.
"""
import argparse
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return True  # non-tensor leaves are pickled unchanged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--slim", action="store_true")
    a = ap.parse_args()
    out = ROOT / "models" / a.name / "policy.pt"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    if a.slim:
        ck.pop("optimizer_state", None)
    torch.save(ck, out)
    back = torch.load(out, map_location="cpu", weights_only=False)
    for key in ("state", "value_state"):
        if not same(ck[key], back[key]):
            raise SystemExit(f"{key} differs after save")
    src = torch.load(a.source, map_location="cpu", weights_only=False)
    for key in ("state", "value_state"):
        if not same(src[key], back[key]):
            raise SystemExit(f"{key} differs from the source")
    if back["table_sha"] != src["table_sha"]:
        raise SystemExit("table_sha differs")
    row = [a.name, "slim" if a.slim else "full", sha256(out), str(out.stat().st_size), sha256(a.source),
           str(src.get("rl_meta", {}).get("games", "")), src["table_sha"]]
    with open(ROOT / "models/models.tsv", "a") as stream:
        stream.write("\t".join(row) + "\n")
    print("\t".join(row))


if __name__ == "__main__":
    main()
