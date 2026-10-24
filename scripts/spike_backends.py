#!/usr/bin/env python3
"""Inference-backend spike (plan rollout-throughput, step 1 / gate G1).

Same 1,000 recorded v4 decisions for every candidate:
  eager      PyTorch eager forward (reference: unpadded, batch 1, TF32 off)
  graphs     today's server path (GraphForward: padded buckets, CUDA graphs), TF32 on (as served) and off
  aoti       torch.export + AOTInductor (Triton-fused kernels) on the same padded buckets, replayed as CUDA graphs
Per candidate and batch size: latency per forward (host fill + copy + replay + copy back + sync), host CPU per forward, argmax
agreement and max relative logit difference vs the reference.

    uv run python scripts/spike_backends.py --out reports/data/rollout-throughput/spike.json [--only eager,graphs]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import orjson
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swccg_policy import rl  # noqa: E402
from swccg_policy.data import card_index, parse_record  # noqa: E402


def load_decisions(n: int, seed: int) -> list:
    recs = []
    for f in sorted((ROOT / "runs/rl-v4/traj-consumed").glob("part-*.jsonl.gz")):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                g = orjson.loads(line)
                recs.extend(s["rec"] for s in g["steps"])
    random.Random(seed).shuffle(recs)
    return recs[:n]


def set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


class Flat(torch.nn.Module):
    """The policy forward on the flat bucket tensors (the GraphForward device layout), for torch.export."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tok, tokf, tmask, opt, omask, head):
        b = {"tok": tok, "tokf": tokf, "tmask": tmask, "opt": opt, "omask": omask, "dtype": head[:, 0], "qtmpl": head[:, 1],
             "qbp": head[:, 2], "side": head[:, 3], "glob": head[:, 4:22]}
        return self.m(b).float()


class AotiGraphForward(rl.GraphForward):
    """GraphForward whose captured forward is an AOTInductor-compiled module instead of the eager model."""

    def __init__(self, model, runner):
        super().__init__(model)
        self.runner = runner

    @torch.no_grad()
    def _build(self, key):
        lay, nbytes = self._layout(key)
        host = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)
        dev = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        hbuf = host.numpy()
        host_np = {k: hbuf[o:o + n].view(dt).reshape(sh) for k, (o, n, dt, sh) in lay.items()}
        tdt = {np.int16: torch.int16, np.float32: torch.float32, np.bool_: torch.bool, np.int32: torch.int32}
        dv = {k: dev[o:o + n].view(tdt[dt]).view(sh) for k, (o, n, dt, sh) in lay.items()}
        # AOTI wants contiguous inputs of its own: copy the views into dedicated tensors inside the graph
        args = [dv[k] for k in ("tok", "tokf", "tmask", "opt", "omask", "head")]
        host_np["tokf"].fill(np.nan)
        dev.copy_(host)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.runner(*[a.contiguous() for a in args])
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            out = self.runner(*[a.contiguous() for a in args])
            out = out[0] if isinstance(out, (list, tuple)) else out
        self.pool = graph.pool()
        out_host = torch.empty(out.shape, dtype=torch.float32, pin_memory=True)
        return {"host": host, "host_np": host_np, "dev": dev, "graph": graph, "out": out, "out_host": out_host}


def compile_aoti(model, items, out_dir: Path):
    from torch.export import Dim
    flat = Flat(model).eval()
    ex_items = items[:4]
    T = -(-max(it["tok"].shape[0] for it in ex_items) // 16) * 16
    O = -(-max(it["opt"].shape[0] for it in ex_items) // 8) * 8
    ex = rl.stack(ex_items, "cuda", pad=(4, T, O))
    args = (ex["tok"], ex["tokf"], ex["tmask"], ex["opt"], ex["omask"], torch.cat(
        [ex["dtype"][:, None], ex["qtmpl"][:, None], ex["qbp"][:, None], ex["side"][:, None], ex["glob"]], 1).int())
    # Dim.AUTO: export derives the ranges itself (named Dims trip on guards the solver cannot prove, e.g. min(2+T, 16+8T) == 2+T)
    B = T = O = Dim.AUTO
    ds = ({0: B, 1: T}, {0: B, 1: T}, {0: B, 1: T}, {0: B, 1: O}, {0: B, 1: O}, {0: B})
    t0 = time.time()
    ep = torch.export.export(flat, args, dynamic_shapes=ds, strict=False)
    t1 = time.time()
    pkg = torch._inductor.aoti_compile_and_package(ep, package_path=str(out_dir / "policy.pt2"))
    t2 = time.time()
    runner = torch._inductor.aoti_load_package(pkg)
    return runner, {"export_s": t1 - t0, "compile_s": t2 - t1, "package": pkg}


def bench(name, fwd, batches, ref_argmax, ref_logits, reps):
    """fwd(items) -> numpy logits [B, >=O]; returns latency stats + agreement."""
    for items in batches[:5]:
        fwd(items)  # warm-up / graph capture for the buckets in use
    torch.cuda.synchronize()
    lat, cpu = [], []
    agree = total = 0
    max_rel = 0.0
    for r in range(reps):
        for bi, items in enumerate(batches):
            c0, t0 = time.thread_time(), time.perf_counter()
            lg = fwd(items)
            lat.append(time.perf_counter() - t0)
            cpu.append(time.thread_time() - c0)
            if r == 0:
                for i, it in enumerate(items):
                    n = it["opt"].shape[0]
                    k = it["_idx"]
                    got = lg[i, :n]
                    agree += int(np.argmax(got)) == ref_argmax[k]
                    total += 1
                    ref = ref_logits[k]
                    fin = np.isfinite(ref)
                    rel = np.abs(got[fin] - ref[fin]) / np.maximum(np.abs(ref[fin]), 1.0)
                    max_rel = max(max_rel, float(rel.max()) if rel.size else 0.0)
    lat = np.array(lat) * 1e3
    return {"name": name, "p50_ms": float(np.median(lat)), "mean_ms": float(lat.mean()), "p90_ms": float(np.percentile(lat, 90)),
            "cpu_ms": float(np.mean(cpu) * 1e3), "argmax_agree": agree, "argmax_total": total, "max_rel_diff": max_rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/rl-v4/ckpt/learner-0419.pt"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="eager,graphs,graphs_tf32,aoti")
    ap.add_argument("--out", default=str(ROOT / "reports/data/rollout-throughput/spike.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))

    model, _, vocab, _ = rl.load_rl(a.ckpt)
    model.eval()
    model.cache_card_table()
    bp_idx = card_index(ROOT / "data/cards/card_table.npz")
    recs = load_decisions(a.n, 1)
    items = []
    for k, r in enumerate(recs):
        it = rl.compact(parse_record(r, bp_idx), vocab)
        it["_idx"] = k
        items.append(it)
    res = {"ckpt": a.ckpt, "n": len(items), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": []}

    # reference: eager, unpadded, batch 1, TF32 off
    set_tf32(False)
    ref_logits, ref_argmax = [], []
    with torch.no_grad():
        for it in items:
            lg = model(rl.stack([it], "cuda")).float().cpu().numpy()[0]
            ref_logits.append(lg)
            ref_argmax.append(int(np.argmax(lg)))
    rng = random.Random(2)
    sizes = [int(x) for x in a.batches.split(",")]
    batch_sets = {B: [rng.sample(items, B) for _ in range(max(20, 400 // B))] for B in sizes}

    def eager_fwd(its):
        with torch.no_grad():
            return model(rl.stack(its, "cuda")).float().cpu().numpy()

    gf = rl.GraphForward(model)

    def graph_fwd(its):
        key = gf.bucket(its)
        o = gf.launch(its, key)
        torch.cuda.synchronize()
        return o.numpy()

    runs = []
    if "eager" in only:
        runs.append(("eager", False, eager_fwd))
    if "graphs" in only:
        runs.append(("graphs", False, graph_fwd))
    if "graphs_tf32" in only:
        gf32 = rl.GraphForward(model)

        def graph32_fwd(its):
            o = gf32.launch(its, gf32.bucket(its))
            torch.cuda.synchronize()
            return o.numpy()
        runs.append(("graphs_tf32", True, graph32_fwd))
    if "aoti" in only:
        set_tf32(False)
        try:
            runner, info = compile_aoti(model, items, out.parent)
            res["aoti_compile"] = info
            gf_pad = rl.GraphForward(model)  # only for its bucket() shapes

            def aoti_eager_fwd(its):
                bb = rl.stack(its, "cuda", pad=gf_pad.bucket(its))
                head = torch.stack([bb["dtype"], bb["qtmpl"], bb["qbp"], bb["side"]], 1)
                head = torch.cat([head, bb["glob"]], 1).int().contiguous()
                out = runner(bb["tok"], bb["tokf"], bb["tmask"], bb["opt"], bb["omask"], head)
                out = out[0] if isinstance(out, (list, tuple)) else out
                return out.float().cpu().numpy()
            runs.append(("aoti_eager", False, aoti_eager_fwd))
            if "aoti_graphs" in only:
                ag = AotiGraphForward(model, runner)

                def aoti_fwd(its):
                    o = ag.launch(its, ag.bucket(its))
                    torch.cuda.synchronize()
                    return o.numpy()
                runs.append(("aoti_graphs", False, aoti_fwd))
        except Exception as e:  # report, don't hide
            res["aoti_error"] = repr(e)[:2000]
            print("AOTI failed:", repr(e)[:2000], flush=True)
    if "script" in only:
        # TorchScript trace of the flat forward (what libtorch C++ loads), replayed as CUDA graphs per bucket
        set_tf32(False)
        ex_items = items[:4]
        T = -(-max(it["tok"].shape[0] for it in ex_items) // 16) * 16
        O = -(-max(it["opt"].shape[0] for it in ex_items) // 8) * 8
        ex = rl.stack(ex_items, "cuda", pad=(4, T, O))
        head = torch.cat([torch.stack([ex["dtype"], ex["qtmpl"], ex["qbp"], ex["side"]], 1), ex["glob"]], 1).int()
        with torch.no_grad():
            traced = torch.jit.trace(Flat(model).eval(), (ex["tok"], ex["tokf"], ex["tmask"], ex["opt"], ex["omask"], head),
                                     check_trace=False)
        traced = torch.jit.freeze(traced.eval()) if False else traced
        traced.save(str(out.parent / "policy_traced.pt"))
        loaded = torch.jit.load(str(out.parent / "policy_traced.pt"), map_location="cuda")
        sgf = AotiGraphForward(model, loaded)

        def script_fwd(its):
            o = sgf.launch(its, sgf.bucket(its))
            torch.cuda.synchronize()
            return o.numpy()
        runs.append(("script_graphs", False, script_fwd))
    if "compile" in only:
        # torch.compile (Inductor, the same fused kernels as AOTI) + CUDA graphs per bucket shape, as a reference for C++
        set_tf32(False)
        cflat = torch.compile(Flat(model).eval(), mode="max-autotune-no-cudagraphs", dynamic=False)
        cgf = AotiGraphForward(model, cflat)

        def compile_fwd(its):
            o = cgf.launch(its, cgf.bucket(its))
            torch.cuda.synchronize()
            return o.numpy()
        runs.append(("compile_graphs", False, compile_fwd))
    for name, tf32, fwd in runs:
        set_tf32(tf32)
        for B in sizes:
            r = bench(name, fwd, batch_sets[B], ref_argmax, ref_logits, a.reps)
            r["batch"] = B
            r["tf32"] = tf32
            res["results"].append(r)
            print(json.dumps(r), flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
