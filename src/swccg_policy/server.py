"""Policy server for the Gemp PolicyAi bot: one JSON line per decision in (StateEncoder format), one JSON array of option
indices out (best first). Connections are served by threads; a single GPU worker batches pending requests.

    uv run python -m swccg_policy.server --checkpoint checkpoints/bc-v1/model.pt --port 47000
    uv run python -m swccg_policy.server --selftest data/shards/v1-eval_seen --packed data/packed/v1-eval_seen \
        --checkpoint ... --port 47000     # client: server choices == offline inference on packed data
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import queue
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np
import orjson
import torch

from .data import DTYPE_INDEX, card_index, parse_record
from .train import Packed, load_checkpoint


def records_to_batch(recs: list, vocab: dict, bp_idx: dict, device: str, max_tokens: int = 320) -> dict:
    if vocab.get("schema_version", 1) >= 2:
        raise ValueError("the V1 server accepts schema-1 records only (plan remove-old-generations)")
    parsed = [parse_record(r, bp_idx) for r in recs]
    B = len(parsed)
    T = max(1, min(max_tokens, max(len(p[1]) for p in parsed)))
    O = max(1, max(len(p[3]) for p in parsed))
    tok = np.zeros((B, T, 8), dtype=np.int16)
    tokf = np.full((B, T, 4), np.nan, dtype=np.float32)
    tmask = np.zeros((B, T), dtype=bool)
    opt = np.zeros((B, O, 6), dtype=np.int32)
    om = np.zeros((B, O), dtype=bool)
    rec = np.zeros((B, 28), dtype=np.int32)
    qv, ov = vocab["q"], vocab["o"]
    for i, (r, ti, tf, orow, och, qt, otmpl, gid) in enumerate(parsed):
        n_t = min(len(ti), T)
        if n_t:
            tok[i, :n_t] = np.asarray(ti[:n_t], dtype=np.int16)
            tokf[i, :n_t] = np.asarray(tf[:n_t], dtype=np.float32)
        tmask[i, :n_t] = True
        o = np.asarray(orow, dtype=np.int32)
        o[:, 0] = [ov.get(s, 0) for s in otmpl]
        o[o[:, 2] > n_t, 2] = 0
        opt[i, :len(o)] = o
        om[i, :len(o)] = True
        dtype, _, qbp, side, dec, gl = r
        rec[i, :4] = (dtype, qv.get(qt, 0), qbp, side)
        rec[i, 10:28] = gl
    tok[:, :, 5][tok[:, :, 5] > T] = 0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
    return {"tok": t(tok), "tokf": t(tokf), "tmask": t(tmask), "opt": t(opt), "omask": t(om), "dtype": t(rec[:, 0]),
            "qtmpl": t(rec[:, 1]), "qbp": t(rec[:, 2]), "side": t(rec[:, 3]), "glob": t(rec[:, 10:28])}


class Batcher:
    def __init__(self, model, vocab, bp_idx, device="cuda", max_batch=256, wait_ms=1.0, fp32=True):
        self.model, self.vocab, self.bp_idx, self.device, self.fp32 = model, vocab, bp_idx, device, fp32
        self.q: queue.Queue = queue.Queue()
        self.max_batch, self.wait = max_batch, wait_ms / 1000
        self.batches = self.requests = 0
        threading.Thread(target=self.loop, daemon=True).start()

    def rank(self, rec) -> list[int]:
        slot = {"ev": threading.Event()}
        self.q.put((rec, slot))
        slot["ev"].wait()
        return slot["out"]

    @torch.no_grad()
    def loop(self):
        while True:
            items = [self.q.get()]
            deadline = time.perf_counter() + self.wait
            while len(items) < self.max_batch:
                try:
                    items.append(self.q.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            try:
                b = records_to_batch([it[0] for it in items], self.vocab, self.bp_idx, self.device)
                # fp32: bf16 rounding flips near-ties (e.g. duplicate cards) depending on batch composition.
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.fp32):
                    logits = self.model(b).float()
                order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
                for i, (rec, slot) in enumerate(items):
                    slot["out"] = [int(j) for j in order[i][:len(rec[9])]]
                    slot["ev"].set()
            except Exception as e:  # never leave a game thread hanging
                print("batch failed:", repr(e), flush=True)
                for rec, slot in items:
                    slot["out"] = list(range(len(rec[9])))
                    slot["ev"].set()
            self.batches += 1
            self.requests += len(items)


def serve(a) -> None:
    model, vocab = load_checkpoint(a.checkpoint)
    bp_idx = card_index(a.cards)
    batcher = Batcher(model, vocab, bp_idx, max_batch=a.max_batch, wait_ms=a.wait_ms)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for line in self.rfile:
                rec = orjson.loads(line)
                out = batcher.rank(rec)
                self.wfile.write(orjson.dumps(out) + b"\n")
                self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Server(("127.0.0.1", a.port), Handler)
    print(f"policy server on 127.0.0.1:{a.port} ({a.checkpoint})", flush=True)
    def stats():
        while True:
            time.sleep(60)
            if batcher.batches:
                print(f"requests {batcher.requests}, batches {batcher.batches}, mean batch {batcher.requests / batcher.batches:.1f}", flush=True)
    threading.Thread(target=stats, daemon=True).start()
    srv.serve_forever()


def selftest(a) -> None:
    """Client: send the first K records of a shard directory (same filtering as the packer) and compare the server's top
    choice with offline inference on the packed split."""
    lines = []
    for shard in sorted(glob.glob(str(Path(a.selftest) / "shard-*.jsonl.gz"))):
        with gzip.open(shard, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                if r[10] and len(r[9]) >= 2:
                    lines.append(line.strip())
                if len(lines) >= a.k:
                    break
        if len(lines) >= a.k:
            break
    s = socket.create_connection(("127.0.0.1", a.port))
    f = s.makefile("rwb")
    server_top = []
    t0 = time.perf_counter()
    for line in lines:
        f.write(line + b"\n")
        f.flush()
        server_top.append(orjson.loads(f.readline())[0])
    dt = time.perf_counter() - t0
    model, vocab = load_checkpoint(a.checkpoint)
    p = Packed(a.packed)
    offline = []
    with torch.no_grad():
        for s0 in range(0, len(lines), 256):
            b = p.batch(np.arange(s0, min(len(lines), s0 + 256)), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                offline += torch.argmax(model(b).float(), dim=1).cpu().tolist()
    agree = sum(int(x == y) for x, y in zip(server_top, offline))
    print(json.dumps({"records": len(lines), "top1_agreement": agree / len(lines), "round_trip_ms": 1000 * dt / len(lines)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cards", default=str(Path(__file__).resolve().parents[2] / "data/cards/card_table.npz"))
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--wait-ms", type=float, default=1.0)
    ap.add_argument("--selftest")
    ap.add_argument("--packed")
    ap.add_argument("--k", type=int, default=2000)
    a = ap.parse_args()
    selftest(a) if a.selftest else serve(a)


if __name__ == "__main__":
    main()
