"""Behaviour-cloning training and evaluation.

    uv run python -m swccg_policy.train --train data/packed/v1-train --eval data/packed/v1-eval_seen \
        --eval data/packed/v1-eval_heldout --out checkpoints/bc-v1 --epochs 4
    uv run python -m swccg_policy.train --evaluate-only --checkpoint checkpoints/bc-v1/model.pt --eval data/packed/...
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import DTYPES
from .model import ModelConfig, Policy, imitation_loss
from .train_step import TrainingConfig, TrainingStep

ROOT = Path(__file__).resolve().parents[2]


class Packed:
    """Memory-mapped packed split."""

    def __init__(self, path: str | Path):
        p = Path(path)
        self.path = p
        self.rec = np.load(p / "rec.npy", mmap_mode="r")
        self.tok = np.load(p / "tok.npy", mmap_mode="r")
        self.tokf = np.load(p / "tokf.npy", mmap_mode="r")
        self.opt = np.load(p / "opt.npy", mmap_mode="r")
        self.chosen = np.load(p / "chosen.npy", mmap_mode="r")
        self.vocab = json.load(open(p / "vocab.json"))
        self.n = self.rec.shape[0]

    def batch(self, idx: np.ndarray, device: str, max_tokens: int = 320) -> dict:
        rec = np.asarray(self.rec[np.sort(idx)])
        B = len(rec)
        tn = np.minimum(rec[:, 5], max_tokens)
        on = rec[:, 7]
        T, O = int(tn.max()), int(on.max())
        tok = np.zeros((B, T, 8), dtype=np.int16)
        tokf = np.full((B, T, 4), np.nan, dtype=np.float16)
        tmask = np.zeros((B, T), dtype=bool)
        opt = np.zeros((B, O, 6), dtype=np.int32)
        ch = np.zeros((B, O), dtype=bool)
        om = np.zeros((B, O), dtype=bool)
        for i, r in enumerate(rec):
            t0, n_t, o0, n_o = r[4], tn[i], r[6], r[7]
            tok[i, :n_t] = self.tok[t0:t0 + n_t]
            tokf[i, :n_t] = self.tokf[t0:t0 + n_t]
            tmask[i, :n_t] = True
            o = np.asarray(self.opt[o0:o0 + n_o])
            # truncated token lists: drop references to tokens beyond the cut
            o = o.copy()
            o[o[:, 2] > n_t, 2] = 0
            opt[i, :n_o] = o
            ch[i, :n_o] = self.chosen[o0:o0 + n_o] > 0
            om[i, :n_o] = True
        tok[:, :, 5][tok[:, :, 5] > T] = 0
        t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
        return {"tok": t(tok), "tokf": t(tokf.astype(np.float32)), "tmask": t(tmask), "opt": t(opt), "chosen": t(ch), "omask": t(om),
                "dtype": t(rec[:, 0]), "qtmpl": t(rec[:, 1]), "qbp": t(rec[:, 2]), "side": t(rec[:, 3]), "glob": t(rec[:, 10:28]),
                "_rec": rec}


def prefetch(split: "Packed", index_lists, device: str, depth: int = 6, workers: int = 4):
    """Assemble batches in background threads (numpy slicing releases the GIL for the copies) and yield them in order."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(workers)
    futures = queue.Queue()
    it = iter(index_lists)

    def submit():
        try:
            idx = next(it)
        except StopIteration:
            return False
        futures.put(pool.submit(split.batch, idx, "cpu"))
        return True

    for _ in range(depth):
        if not submit():
            break
    while not futures.empty():
        b = futures.get().result()
        submit()
        yield {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    pool.shutdown()


def load_cards(path: str | Path):
    z = np.load(path)
    return z["features"], z["text_emb"], z["ids"], z["titles"]


def build_model(train_vocab: dict, cards_path: str | Path, **kw) -> Policy:
    feats, text, _, _ = load_cards(cards_path)
    cfg = ModelConfig(n_cards=feats.shape[0], n_card_features=feats.shape[1], text_dim=text.shape[1],
                      n_qtmpl=len(train_vocab["q"]), n_otmpl=len(train_vocab["o"]), **kw)
    return Policy(cfg, feats, text)


def baseline_tables(train: Packed) -> dict:
    """Most frequent chosen option per (decision type, question template). The option's identity is its template plus
    value ("t": INTEGER values count), or template + value + blueprint ("tb")."""
    by_t = collections.defaultdict(collections.Counter)
    by_tb = collections.defaultdict(collections.Counter)
    rec = np.asarray(train.rec)
    opt = np.asarray(train.opt)
    ch = np.asarray(train.chosen)
    for r in rec:
        key = (int(r[0]), int(r[1]))
        o = opt[r[6]:r[6] + r[7]]
        c = ch[r[6]:r[6] + r[7]]
        for row, flag in zip(o, c):
            if flag:
                by_t[key][(int(row[0]), int(row[4]))] += 1
                by_tb[key][(int(row[0]), int(row[4]), int(row[3]))] += 1
    return {"t": by_t, "tb": by_tb}


@torch.no_grad()
def evaluate(model: Policy | None, split: Packed, device: str, tables: dict | None = None, batch: int = 512,
             limit: int | None = None, indices: np.ndarray | None = None) -> dict:
    idx_all = np.arange(split.n) if indices is None else np.asarray(indices)
    if limit is not None:
        idx_all = idx_all[:limit]
    n = len(idx_all)
    per = collections.defaultdict(lambda: collections.Counter())
    if model is not None:
        model.eval()
    for s in range(0, n, batch):
        idx = idx_all[s:s + batch]
        b = split.batch(idx, device)
        rec = b["_rec"]
        ch = b["chosen"].cpu().numpy()
        om = b["omask"].cpu().numpy()
        opt = b["opt"].cpu().numpy()
        if model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b).float().cpu().numpy()
            order = np.argsort(-logits, axis=1)
        for i, r in enumerate(rec):
            dt = DTYPES[int(r[0])]
            no = int(om[i].sum())
            chosen = set(np.nonzero(ch[i][:no])[0].tolist())
            for key in (dt, "ALL"):
                c = per[key]
                c["n"] += 1
                c["random"] += len(chosen) / no
                if model is not None:
                    top = order[i][:no]
                    c["top1"] += int(top[0] in chosen)
                    c["top3"] += int(any(t in chosen for t in top[:3]))
                if tables is not None:
                    # Frequency baselines: ties among equally frequent options are broken uniformly at random
                    # (expected accuracy). "first" = the first option in the engine's order: positional information
                    # the order-invariant policy does not get (AdvancedAi picks card selections by position).
                    k = (int(r[0]), int(r[1]))
                    for name, tab, keyf in (("freq_t", tables["t"], lambda row: (int(row[0]), int(row[4]))),
                                            ("freq_tb", tables["tb"], lambda row: (int(row[0]), int(row[4]), int(row[3])))):
                        cnt = tab.get(k)
                        scores = [cnt.get(keyf(opt[i][j]), 0) for j in range(no)] if cnt else [0] * no
                        top = max(scores)
                        best = [j for j in range(no) if scores[j] == top]
                        c[name] += sum(1 for j in best if j in chosen) / len(best)
                    c["first"] += int(0 in chosen)
    out = {}
    for key, c in per.items():
        n_ = c["n"]
        out[key] = {"n": n_, **{m: c[m] / n_ for m in ("random", "top1", "top3", "freq_t", "freq_tb", "first") if m in c or m == "random"}}
    return out


def train(a) -> None:
    device = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tr = Packed(a.train)
    evals = [Packed(p) for p in a.eval]
    model = build_model(tr.vocab, a.cards, d=a.d, layers=a.layers, heads=a.heads, ff=a.ff).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params / 1e6:.1f} M; train records {tr.n}")
    training = TrainingStep(model, TrainingConfig(optimizer="adamw", lr=a.lr, betas=(0.9, 0.98),
                                                 weight_decay=0.01, delta_weight_decay=a.delta_wd,
                                                 precision="bf16"))
    optim = training.optimizer
    steps_per_epoch = tr.n // a.batch
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    configuration = {"training_step": training.config.to_dict(), "scheduler": "500-step warmup then cosine"}
    log.write(json.dumps(configuration) + "\n")
    log.flush()
    rng = np.random.default_rng(a.seed)
    step = 0
    t0 = time.time()
    deadline = t0 + a.max_hours * 3600
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr.n)
        batches = prefetch(tr, [perm[s * a.batch:(s + 1) * a.batch] for s in range(steps_per_epoch)], device)
        for s in range(steps_per_epoch):
            b = next(batches)
            update = training(b, lambda logits, smoothing: imitation_loss(logits, b["chosen"], b["omask"]))
            loss = update["loss"]
            sched.step()
            step += 1
            if step % 200 == 0:
                rec = {"step": step, "epoch": ep, "loss": float(loss), "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if time.time() > deadline:
                print("time budget reached", flush=True)
                break
        res = {Path(e.path).name: evaluate(model, e, device, limit=a.eval_limit)["ALL"] for e in evals}
        rec = {"epoch": ep, "step": step, "eval": res, "elapsed_s": time.time() - t0}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"cfg": model.cfg.to_dict(), "state": model.state_dict(), "vocab": tr.vocab, "cards": a.cards}, out / "model.pt")
        if time.time() > deadline:
            break
    print(f"training done in {(time.time() - t0) / 60:.1f} min")


def load_checkpoint(path: str | Path, device: str = "cuda") -> tuple[Policy, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    if cfg.get("schema_version") == 4:
        from .v4_checkpoint import load
        actor, _, tables, _ = load(path, device, payload=ck)
        return actor, tables.vocabulary.payload
    # Plan remove-old-generations: V1 (schema 1) and V3.1 (schema 4, above) only; schema 2 / 3 checkpoints are refused.
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("unsupported checkpoint schema")
    model = build_model(ck["vocab"], ck["cards"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], ff=cfg["ff"])
    if ck["vocab"].get("schema_version", 1) != 1:
        raise ValueError("checkpoint schema mismatch")
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["vocab"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/packed/v1-train"))
    ap.add_argument("--eval", action="append", default=[])
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints/bc-v1"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--delta-wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--eval-limit", type=int, default=100_000)
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--report", help="write full evaluation JSON here (evaluate-only)")
    a = ap.parse_args()
    if a.evaluate_only:
        model, vocab = load_checkpoint(a.checkpoint)
        tables = baseline_tables(Packed(a.train))
        res = {Path(p).name: evaluate(model, Packed(p), "cuda", tables) for p in a.eval}
        print(json.dumps(res, indent=1))
        if a.report:
            json.dump(res, open(a.report, "w"), indent=1)
    else:
        train(a)


if __name__ == "__main__":
    main()
