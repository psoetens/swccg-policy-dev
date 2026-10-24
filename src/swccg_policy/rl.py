"""Self-play reinforcement learning (PPO) for the SWCCG policy (plan: .claude/plans/rl-selfplay-v1.md).

Three processes:

  server   RL policy server for the Gemp harness (TrainGen agents "policy:KEY", envelope protocol, see PolicyAi.java).
           Model keys: "learner" (current weights, sampled, trajectories recorded), "bc" (frozen bc-v1, sampled),
           "snap" (a random past learner snapshot per game, sampled), "<key>@greedy" (argmax, not recorded) for learner/bc,
           and "ck:<checkpoint path>@greedy" for evaluation. Learner weights and snapshots are reloaded when they change.
  learn    PPO learner: reads finished trajectories written by the server, trains policy + value head on the GPU, writes
           checkpoints (latest.pt is what the server serves), and logs one JSON line per iteration.
  eval     Seeded evaluation games through TrainGen (single-threaded JVMs, --seed) against the running server.

    uv run python -m swccg_policy.rl server --run runs/rl-v1 --port 47100
    uv run python -m swccg_policy.rl learn  --run runs/rl-v1
    uv run python -m swccg_policy.rl eval   --schedule data/schedules/eval_vs_bc.tsv --agent policy:ck:PATH@greedy \
        --opponent policy:bc@greedy --name NAME --port 47100
    uv run python -m swccg_policy.rl selfcheck --packed data/packed/v2-eval_seen --port 47100   # greedy == offline argmax
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
import math
import os
import queue
import random
import shutil
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import card_index, parse_record
from .train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
MAX_TOKENS = 320


def use_tf32():
    """TF32 CUDA matmuls / convolutions, the precision every V3.1 learner, evaluation and probe has run with. Set on
    import of this module; tools that load the model without importing it (the probe scripts) call it in main()."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


use_tf32()


# ----------------------------------------------------------------------------------------------------------------- model

class ValueHead(nn.Module):
    """V(s) from the (detached) question + global token outputs of the policy encoder."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def load_rl(path: str | Path, device: str = "cuda"):
    """Policy (+ value head if the checkpoint has one) and the raw checkpoint dict."""
    model, vocab = load_checkpoint(path, device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if getattr(model.cfg, "schema_version", 1) != 1:
        raise ValueError("this learner runs V1 (schema 1) checkpoints; V3.1 uses scripts/v31_stage.py")
    value = ValueHead(model.cfg.d).to(device)
    if "value_state" in ck:
        value.load_state_dict(ck["value_state"])
    return model, value, vocab, ck


def save_rl(path: Path, base_ck: dict, model, value, meta: dict, optimizers=None) -> None:
    ck = {k: v for k, v in base_ck.items() if k not in ("state", "value_state", "rl_meta", "optimizer_state")}
    ck["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck["value_state"] = {k: v.detach().cpu() for k, v in value.state_dict().items()}
    ck["rl_meta"] = meta
    if optimizers is not None:
        ck["optimizer_state"] = [o.state_dict() for o in optimizers]
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------ batch building

def compact(parsed, vocab) -> dict:
    """One parsed record (data.parse_record) -> compact numpy arrays (kept per step by the learner)."""
    r, ti, tf, orow, och, qt, otmpl, gid = parsed
    version = r[1]
    if version != vocab.get("schema_version", 1):
        raise ValueError("observation/checkpoint schema mismatch")
    if version != 1:
        raise ValueError("compact handles V1 (schema 1) records only (plan remove-old-generations)")
    n_t = min(len(ti), MAX_TOKENS)
    tok = np.asarray(ti[:n_t], dtype=np.int16).reshape(n_t, 8)
    tokf = np.asarray(tf[:n_t], dtype=np.float16).reshape(n_t, 4)
    o = np.asarray(orow, dtype=np.int32).reshape(len(orow), 6)
    o[:, 0] = [vocab["o"].get(s, 0) for s in otmpl]
    o[o[:, 2] > n_t, 2] = 0
    qid = vocab["q"].get(qt, 0)
    dtype, _, qbp, side, dec, gl = r
    head = np.zeros(22, dtype=np.int32)
    head[:4] = (dtype, qid, qbp, side)
    head[4:22] = gl
    item = {"tok": tok, "tokf": tokf, "opt": o, "head": head, "schema_version": version}
    return item


def stack(items: list[dict], device: str, pad: tuple[int, int, int] | None = None, pin: bool = False) -> dict:
    """Batch compacted items; pad = (B, T, O) pads to a larger (CUDA-graph bucket) shape with masked rows/tokens/options."""
    B = len(items)
    T = max(1, max(it["tok"].shape[0] for it in items))
    O = max(1, max(it["opt"].shape[0] for it in items))
    Bp, Tp, Op = pad if pad is not None else (B, T, O)
    tok = np.zeros((Bp, Tp, 8), dtype=np.int16)
    tokf = np.full((Bp, Tp, 4), np.nan, dtype=np.float32)
    tmask = np.zeros((Bp, Tp), dtype=bool)
    opt = np.zeros((Bp, Op, 6), dtype=np.int32)
    om = np.zeros((Bp, Op), dtype=bool)
    head = np.zeros((Bp, 22), dtype=np.int32)
    for i, it in enumerate(items):
        n_t, n_o = it["tok"].shape[0], it["opt"].shape[0]
        tok[i, :n_t] = it["tok"]
        tokf[i, :n_t] = it["tokf"]
        tmask[i, :n_t] = True
        opt[i, :n_o] = it["opt"]
        om[i, :n_o] = True
        head[i] = it["head"]
    tok[:, :, 5][tok[:, :, 5] > T] = 0  # parent indices beyond the real token count (not the padded one)
    if pin:  # pinned host memory: the copies are really asynchronous
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).pin_memory().to(device, non_blocking=True)
    else:
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
    return {"tok": t(tok), "tokf": t(tokf), "tmask": t(tmask), "opt": t(opt), "omask": t(om), "dtype": t(head[:, 0]),
            "qtmpl": t(head[:, 1]), "qbp": t(head[:, 2]), "side": t(head[:, 3]), "glob": t(head[:, 4:22])}


def none_flags(item: dict) -> np.ndarray:
    return item["opt"][:, 5] > 0


# ------------------------------------------------------------------------------------------------------------ sampling

def sample_sequence(logits: np.ndarray, is_none: np.ndarray, k: int, temperature: float, eps: float,
                    rng: random.Random, excluded: list[int] | None = None) -> tuple[list[int], float]:
    """Sample the option(s) PolicyAi will answer with and their behaviour log-probability.

    Mirrors PolicyAi: a "none"/"pass" option first means answering nothing; otherwise the first k non-none options are
    the answer. First pick over all legal options, later picks over the remaining non-none options (Plackett-Luce).
    Behaviour distribution per pick: (1 - eps) * softmax(logits / T) + eps * uniform.
    excluded: options that may not be picked (the preselected cards of a step-by-step selection that must add a card)."""
    n = len(logits)
    avail = np.ones(n, dtype=bool)
    if excluded:
        avail[[i for i in excluded if 0 <= i < n]] = False
    seq, logp = [], 0.0
    for j in range(max(1, k)):
        mask = avail.copy()
        if j > 0:
            mask &= ~is_none
        if not mask.any():
            break
        z = logits[mask].astype(np.float64) / max(temperature, 1e-6)
        p = np.exp(z - z.max())
        normalizer = p.sum()
        p /= normalizer
        if eps > 0:
            p = (1 - eps) * p + eps / len(p)
        idx = np.flatnonzero(mask)
        c = rng.choices(range(len(idx)), weights=p.tolist())[0]
        choice = int(idx[c])
        seq.append(choice)
        logp += math.log(p[c]) if eps > 0 else float(z[c] - z.max() - math.log(normalizer))
        avail[choice] = False
        if j == 0 and is_none[choice]:
            break
    return seq, logp


def seq_logprob(logits: torch.Tensor, omask: torch.Tensor, is_none: torch.Tensor, seq: torch.Tensor, seq_len: torch.Tensor,
                excluded: torch.Tensor | None = None, sync_free: bool = False):
    """Log-probability (under softmax(logits)) of each row's pick sequence, same rules as sample_sequence (eps=0, T=1).
    logits/omask/is_none [B, O]; seq [B, K] (padded with 0), seq_len [B]; excluded [B, O] bool (options sampling skipped)."""
    B, K = seq.shape
    avail = omask.clone() if excluded is None else omask & ~excluded
    total = torch.zeros(B, device=logits.device, dtype=torch.float32)
    ar = torch.arange(B, device=logits.device)
    for j in range(K):
        active = seq_len > j
        # sync_free: K is the longest sequence of the batch, so some row is active at every j (the check is a GPU sync)
        if not sync_free and not active.any():
            break
        mask = avail & (~is_none if j > 0 else torch.ones_like(avail))
        lp = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=1)
        pick = seq[:, j]
        got = lp[ar, pick]
        total = total + torch.where(active, got, torch.zeros_like(got))
        avail = avail.clone()
        avail[ar[active], pick[active]] = False
    return total


# --------------------------------------------------------------------------------------------------------------- server

# Serialises CUDA-graph capture (GPU loop thread) with model loading in other server threads: a capture must not overlap
# other CUDA work of the process.
CUDA_CAPTURE_LOCK = threading.Lock()


class GraphForward:
    """CUDA-graph replay of a model's forward (inference only) on padded bucket shapes: one graph per (B, T, O) bucket, captured
    on first use; larger shapes use the eager forward. Per bucket, all inputs live in one pinned host buffer and one device
    buffer, so a call is: write the batch into pinned memory, one host-to-device copy, one graph replay, one device-to-host copy
    of the logits into pinned memory (all asynchronous; the caller synchronizes once)."""
    B_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
    T_STEP, O_STEP, O_MAX = 16, 8, 64
    FIELDS = (("tok", np.int16, 8), ("tokf", np.float32, 4), ("tmask", np.bool_, 0), ("opt", np.int32, 6),
              ("omask", np.bool_, 0), ("head", np.int32, 22))

    def __init__(self, model):
        self.model = model
        self.graphs: dict[tuple, dict] = {}
        self.pool = None

    def bucket(self, items: list[dict]) -> tuple[int, int, int] | None:
        B = len(items)
        T = max(1, max(it["tok"].shape[0] for it in items))
        O = max(1, max(it["opt"].shape[0] for it in items))
        Bb = next((x for x in self.B_BUCKETS if x >= B), None)
        Tb, Ob = -(-T // self.T_STEP) * self.T_STEP, -(-O // self.O_STEP) * self.O_STEP
        if Bb is None or Tb > MAX_TOKENS or Ob > self.O_MAX:
            return None
        return Bb, Tb, Ob

    def launch(self, items: list[dict], key: tuple) -> torch.Tensor:
        """Queue the forward of the compacted items on the bucket `key`. Returns the pinned host logits [B, O] of the bucket,
        valid after the next torch.cuda.synchronize() and until the next launch() of this key."""
        g = self.graphs.get(key)
        if g is None:
            g = self.graphs[key] = self._build(key)
        self._fill(g["host_np"], items, key)
        g["dev"].copy_(g["host"], non_blocking=True)
        g["graph"].replay()
        g["out_host"].copy_(g["out"], non_blocking=True)
        return g["out_host"]

    def _layout(self, key):
        Bb, Tb, Ob = key
        shapes = {"tok": (Bb, Tb, 8), "tokf": (Bb, Tb, 4), "tmask": (Bb, Tb), "opt": (Bb, Ob, 6), "omask": (Bb, Ob), "head": (Bb, 22)}
        off, lay = 0, {}
        for name, dt, _ in self.FIELDS:
            n = int(np.prod(shapes[name])) * np.dtype(dt).itemsize
            lay[name] = (off, n, dt, shapes[name])
            off += -(-n // 16) * 16
        return lay, off

    @staticmethod
    def _fill(host_np: dict, items: list[dict], key: tuple) -> None:
        # same layout and padding as stack(); parent indices are clamped to the real token count
        T = max(1, max(it["tok"].shape[0] for it in items))
        tok, tokf, tmask, opt, om, head = (host_np[k] for k in ("tok", "tokf", "tmask", "opt", "omask", "head"))
        tok.fill(0); tokf.fill(np.nan); tmask.fill(False); opt.fill(0); om.fill(False); head.fill(0)
        for i, it in enumerate(items):
            n_t, n_o = it["tok"].shape[0], it["opt"].shape[0]
            tok[i, :n_t] = it["tok"]
            tokf[i, :n_t] = it["tokf"]
            tmask[i, :n_t] = True
            opt[i, :n_o] = it["opt"]
            om[i, :n_o] = True
            head[i] = it["head"]
        col = tok[:, :, 5]
        col[col > T] = 0

    @torch.no_grad()
    def _build(self, key: tuple) -> dict:
        lay, nbytes = self._layout(key)
        host = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)
        dev = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        hbuf = host.numpy()
        host_np = {k: hbuf[o:o + n].view(dt).reshape(sh) for k, (o, n, dt, sh) in lay.items()}
        tdt = {np.int16: torch.int16, np.float32: torch.float32, np.bool_: torch.bool, np.int32: torch.int32}
        dv = {k: dev[o:o + n].view(tdt[dt]).view(sh) for k, (o, n, dt, sh) in lay.items()}
        head = dv["head"]
        b = {"tok": dv["tok"], "tokf": dv["tokf"], "tmask": dv["tmask"], "opt": dv["opt"], "omask": dv["omask"],
             "dtype": head[:, 0], "qtmpl": head[:, 1], "qbp": head[:, 2], "side": head[:, 3], "glob": head[:, 4:22]}
        host_np["tokf"].fill(np.nan)
        with CUDA_CAPTURE_LOCK:
            dev.copy_(host)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.model(b)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                out = self.model(b).float()
            self.pool = graph.pool()
        out_host = torch.empty(out.shape, dtype=torch.float32, pin_memory=True)
        return {"host": host, "host_np": host_np, "dev": dev, "graph": graph, "out": out, "out_host": out_host}


class ModelSlot:
    """A model served by the RL server (inference only): its card embedding table is computed once, not per forward; with
    graphs=True its forward replays CUDA graphs."""

    def __init__(self, model, vocab, name, graphs: bool = False):
        self.model, self.vocab, self.name = model, vocab, name
        model.cache_card_table()
        self.graph = GraphForward(model) if graphs else None


class RLServer:
    def __init__(self, a):
        self.a = a
        self.run = Path(a.run)
        self.traj_dir = self.run / "traj"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        (self.run / "snaps").mkdir(parents=True, exist_ok=True)
        self.bp_idx = card_index(a.cards)
        self.lock = threading.Lock()
        self.rng = random.Random(a.seed)
        bc, _, bc_vocab, _ = load_rl(a.bc)
        self.slots = {"bc": ModelSlot(bc, bc_vocab, "bc", a.graphs)}
        self.learner_mtime = None
        self.snaps: dict[str, ModelSlot] = {}
        self.eval_slots: dict[str, ModelSlot] = {}
        self.reload()
        self.game_snap: dict[str, str] = {}
        self.game_keys: dict[str, dict] = collections.defaultdict(dict)
        self.traj: dict[tuple, list] = {}
        self.out_q: queue.Queue = queue.Queue(maxsize=32)
        self.stats = collections.Counter()
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self.gpu_loop, daemon=True).start()
        threading.Thread(target=self.writer_loop, daemon=True).start()
        threading.Thread(target=self.reload_loop, daemon=True).start()

    # -- weights
    def reload(self):
        latest = self.run / "ckpt" / "latest.pt"
        src = latest if latest.exists() else Path(self.a.bc)
        m = src.stat().st_mtime
        if m != self.learner_mtime:
            with CUDA_CAPTURE_LOCK:
                model, _, vocab, ck = load_rl(src)
                slot = ModelSlot(model, vocab, f"learner:{ck.get('rl_meta', {}).get('iteration', 0)}", self.a.graphs)
            with self.lock:
                self.slots["learner"] = slot
                self.learner_mtime = m
            print(f"learner weights <- {src} ({self.slots['learner'].name})", flush=True)
        files = sorted(glob.glob(str(self.run / "snaps" / "*.pt")))[-self.a.max_snaps:]
        for f in files:
            if f not in self.snaps:
                with CUDA_CAPTURE_LOCK:
                    model, _, vocab, _ = load_rl(f)
                    slot = ModelSlot(model, vocab, Path(f).stem, self.a.graphs)
                with self.lock:
                    self.snaps[f] = slot
                print(f"snapshot loaded {f}", flush=True)
        with self.lock:
            for f in list(self.snaps):
                if f not in files:
                    del self.snaps[f]

    def reload_loop(self):
        while True:
            time.sleep(15)
            try:
                self.reload()
            except Exception as e:  # a half-written file: try again next round
                print("reload failed:", repr(e), flush=True)

    def slot_for(self, key: str, game: str) -> tuple[ModelSlot, bool, bool]:
        """(slot, greedy, record)"""
        greedy = key.endswith("@greedy")
        base = key[:-len("@greedy")] if greedy else key
        with self.lock:
            if base == "learner":
                return self.slots["learner"], greedy, not greedy
            if base == "bc":
                return self.slots["bc"], greedy, False
            if base == "snap":
                if not self.snaps:
                    return self.slots["bc"], greedy, False
                f = self.game_snap.get(game)
                if f not in self.snaps:
                    f = self.rng.choice(list(self.snaps))
                    self.game_snap[game] = f
                return self.snaps[f], greedy, False
        if base.startswith("ck:"):
            path = base[3:]
            with CUDA_CAPTURE_LOCK:  # also keeps concurrent first requests from loading the same checkpoint twice
                if path not in self.eval_slots:
                    model, _, vocab, _ = load_rl(path)
                    self.eval_slots[path] = ModelSlot(model, vocab, path, self.a.graphs)
            return self.eval_slots[path], greedy, False
        raise KeyError(key)

    # -- requests
    def handle(self, env: dict) -> list[int]:
        if "end" in env:
            self.finish(env)
            return []
        slot, greedy, record = self.slot_for(env["m"], env["g"])
        self.game_keys[env["g"]][env["s"]] = env["m"]
        item = {"env": env, "slot": slot, "greedy": greedy, "record": record, "ev": threading.Event()}
        self.q.put(item)
        item["ev"].wait()
        return item["out"]

    @torch.no_grad()
    def gpu_loop(self):
        rng = random.Random(self.a.seed + 1)
        while True:
            items = [self.q.get()]
            deadline = time.perf_counter() + self.a.wait_ms / 1000
            while len(items) < self.a.max_batch:
                try:
                    items.append(self.q.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            groups = collections.defaultdict(list)
            for it in items:
                groups[id(it["slot"])].append(it)
            # Phase 1: launch every slot's forward (asynchronous); phase 2: one synchronization, then sampling per request.
            launched = []
            for grp in groups.values():
                slot = grp[0]["slot"]
                try:
                    comp = [compact(parse_record(it["env"]["rec"], self.bp_idx), slot.vocab) for it in grp]
                    key = slot.graph.bucket(comp) if slot.graph is not None else None
                    if key is not None:
                        launched.append((grp, comp, slot.graph.launch(comp, key)))
                    else:
                        launched.append((grp, comp, slot.model(stack(comp, "cuda")).float().to("cpu", non_blocking=True)))
                except Exception as e:  # never leave a game thread hanging
                    self._fail(grp, e)
            if launched:
                torch.cuda.synchronize()
            for grp, comp, logits_cpu in launched:
                try:
                    logits = logits_cpu.numpy()
                    for i, it in enumerate(grp):
                        n = comp[i]["opt"].shape[0]
                        lg = logits[i, :n]
                        order = [int(j) for j in np.argsort(-lg, kind="stable")]
                        if it["greedy"]:
                            it["out"] = order
                        else:
                            env = it["env"]
                            seq, logp = sample_sequence(lg, none_flags(comp[i]), int(env.get("k", 1)), self.a.temperature, 0.0, rng,
                                                        env.get("x"))
                            it["out"] = seq + [j for j in order if j not in seq]
                            # Beyond --guard-threshold occurrences of the identical situation, PolicyAi's loop guard answers
                            # with an untried option instead of this sample: such steps are not recorded.
                            # Steps PolicyAi answers with a rule instead of the sample ("skip": undoing a step-by-step
                            # selection) are not recorded either.
                            if it["record"] and env.get("r", 0) <= self.a.guard_threshold and not env.get("skip") and seq:
                                key = (env["g"], env["s"])
                                step = {"rec": env["rec"], "seq": seq, "logp": logp, "r": env.get("r", 0), "k": int(env.get("k", 1)), "policy": it["slot"].name}
                                if env.get("x"):
                                    step["x"] = env["x"]
                                self.traj.setdefault(key, []).append(step)
                        it["ev"].set()
                except Exception as e:  # never leave a game thread hanging
                    self._fail(grp, e)
                self.stats["requests"] += len(grp)
            self.stats["batches"] += 1

    def _fail(self, grp: list, e: Exception) -> None:
        print("batch failed:", repr(e), flush=True)
        for it in grp:
            if not it["ev"].is_set():
                it["out"] = list(range(len(it["env"]["rec"][9])))
                it["ev"].set()

    def finish(self, env: dict):
        g, s, outcome, winner = env["end"], env["s"], env.get("o", ""), env.get("w", "")
        steps = self.traj.pop((g, s), None)
        keys = self.game_keys.get(g, {})
        other = "ls" if s == "ds" else "ds"
        # a policy opponent is known by its model key; the harness names other opponents in "op" (advanced -> "bot", as before)
        op = env.get("op")
        opp = keys.get(other) or ("bot" if op in (None, "advanced") else op)
        if steps is not None:
            if outcome == "exception":
                self.stats["dropped_exception"] += 1
            else:
                won = outcome == "finished" and winner == s
                reward = (1.0 if won else -1.0) if outcome == "finished" and winner in ("ds", "ls") else 0.0
                # Standard Gameplay reward (plan objective-dojo removed the weekend-plan A7 life-force term).
                lf = env.get("lf")
                self.out_q.put({"g": g, "s": s, "o": outcome, "w": winner, "reward": reward, "won": int(won), "lf": lf, "opp": opp,
                                "learner": self.slots["learner"].name, "steps": steps})
                self.stats["games_recorded"] += 1
        # forget per-game state once both sides reported (or the other side was a bot)
        done = keys.setdefault("_done", set()) if isinstance(keys, dict) else set()
        done.add(s)
        if other in done or other not in keys:
            self.game_keys.pop(g, None)
            self.game_snap.pop(g, None)

    def writer_loop(self):
        """Batch finished trajectories into gzip files: traj/part-<time>-<n>.jsonl.gz (written as .tmp, then renamed)."""
        buf, n_file = [], 0
        last = time.time()
        while True:
            try:
                buf.append(self.out_q.get(timeout=5))
            except queue.Empty:
                pass
            if buf and (len(buf) >= self.a.games_per_file or time.time() - last > 60):
                n_file += 1
                name = self.traj_dir / f"part-{int(time.time())}-{n_file:05d}.jsonl.gz"
                tmp = name.with_suffix(".tmp")
                with gzip.open(tmp, "wb", compresslevel=3) as f:
                    for t in buf:
                        f.write(orjson.dumps(t) + b"\n")
                os.replace(tmp, name)
                buf, last = [], time.time()


def serve(a) -> None:
    srv_state = RLServer(a)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for line in self.rfile:
                env = orjson.loads(line)
                out = srv_state.handle(env)
                self.wfile.write(orjson.dumps(out) + b"\n")
                self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = Server(("127.0.0.1", a.port), Handler)
    print(f"RL server on 127.0.0.1:{a.port} (run {a.run})", flush=True)

    def stats():
        while True:
            time.sleep(60)
            st = srv_state.stats
            print(json.dumps({"t": int(time.time()), **st, "open_traj": len(srv_state.traj),
                              "mean_batch": st["requests"] / max(1, st["batches"])}), flush=True)
    threading.Thread(target=stats, daemon=True).start()
    srv.serve_forever()


# -------------------------------------------------------------------------------------------------------------- learner

from .trajectory import trajectory_files

_LOAD_CTX = {}


def _load_init(vocab, bp_idx):
    _LOAD_CTX["vocab"], _LOAD_CTX["bp_idx"] = vocab, bp_idx


def _load_one(f):
    return load_games([f], _LOAD_CTX["vocab"], _LOAD_CTX["bp_idx"])


def load_games_parallel(pool, files: list[Path]) -> list[dict]:
    """load_games over worker processes (one file per task), same order as load_games."""
    return [g for part in pool.map(_load_one, files) for g in part]


def load_games(files: list[Path], vocab: dict, bp_idx: dict) -> list[dict]:
    games = []
    for f in files:
        if f.suffix == ".traj":
            raise ValueError(f"binary trajectories were V2 / V3 only (plan remove-old-generations): {f}")
        with gzip.open(f, "rb") as fh:
            for line in fh:
                t = orjson.loads(line)
                items, critic_items, seqs, logps, xs = [], [], [], [], []
                t["types"] = []
                t["policy_versions"] = []
                for st in t["steps"]:
                    t["types"].append(st["rec"][4])
                    items.append(compact(parse_record(st["rec"], bp_idx), vocab))
                    seqs.append(st["seq"])
                    logps.append(st["logp"])
                    xs.append(st.get("x"))
                t["items"], t["seqs"], t["old_logp"], t["excluded"] = items, seqs, np.asarray(logps, dtype=np.float32), xs
                t["critic_items"] = critic_items
                del t["steps"]
                games.append(t)
    return games


def learn(a) -> None:
    run = Path(a.run)
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    (run / "snaps").mkdir(parents=True, exist_ok=True)
    consumed_dir = run / "traj-consumed"
    consumed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    if a.seed_all:  # reproducible minibatch order (benchmarks / A-B studies)
        np.random.seed(a.seed)
        random.seed(a.seed)
    latest = run / "ckpt" / "latest.pt"
    start_ck = latest if latest.exists() else Path(a.bc)
    model, value, vocab, base_ck = load_rl(start_ck)
    bc, _, _, _ = load_rl(a.bc)
    bc.eval()
    bp_idx = card_index(a.cards)
    meta = dict(base_ck.get("rl_meta", {}))
    it0 = int(meta.get("iteration", 0))
    games_total = int(meta.get("games", 0))
    steps_total = int(meta.get("steps", 0))
    gpu_seconds = float(meta.get("gpu_seconds", 0.0))
    fused = {"fused": True} if a.fused_adam else {}
    opt_pol = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0, **fused)
    opt_val = torch.optim.AdamW(value.parameters(), lr=a.value_lr, weight_decay=0.0, **fused)
    pool = None
    if a.load_workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.load_workers, initializer=_load_init, initargs=(vocab, bp_idx))
    log = open(run / "learner.jsonl", "a")
    t_start = time.time()
    print(f"learner from {start_ck} (iteration {it0})", flush=True)
    it = it0
    while True:
        # ---- collect
        files, n_steps = [], 0
        while n_steps < a.batch_steps:
            if (run / "STOP").exists() or (a.max_hours and time.time()-t_start >= a.max_hours*3600):
                print("collection stopped at time/stop boundary", flush=True)
                return
            new = [p for p in trajectory_files(run / "traj") if p not in files]
            for p in new:
                n_steps += int(p.stat().st_size / a.bytes_per_step_estimate)
                files.append(p)
                if n_steps >= a.batch_steps:
                    break
            if n_steps < a.batch_steps:
                time.sleep(2)
        t_iter = time.time()
        phases = {}
        phase_memory = {}
        if a.profile_phases: torch.cuda.reset_peak_memory_stats()

        def phase(name, t0=[t_iter]):  # --profile-phases: wall time per phase, GPU synchronised at the boundary
            if a.profile_phases:
                torch.cuda.synchronize()
                now = time.time()
                phases[name] = round(now - t0[0], 3)
                phase_memory[name] = {"allocated_peak": torch.cuda.max_memory_allocated(), "reserved_peak": torch.cuda.max_memory_reserved()}
                torch.cuda.reset_peak_memory_stats()
                t0[0] = now
        games = load_games_parallel(pool, files) if pool is not None else load_games(files, vocab, bp_idx)
        phase("load")
        for p in files:  # keep the last few for inspection, delete older consumed files (transient rollout data)
            shutil.move(str(p), consumed_dir / p.name)
            sidecar = p.with_name(p.name + ".meta.json")
            if sidecar.exists():
                shutil.move(str(sidecar), consumed_dir / sidecar.name)
        for old in trajectory_files(consumed_dir)[:-a.keep_consumed]:
            old.unlink()
            old.with_name(old.name + ".meta.json").unlink(missing_ok=True)
        discarded_stale = 0
        items = [x for g in games for x in g["items"]]
        critic_items = items
        if not items:
            continue
        if a.kl_diag:  # per step: decision type, acting learner version (trajectory label), for the KL diagnostic
            diag_type = [st_ for g in games for st_ in g.get("types", [None] * len(g["items"]))]
            diag_ver = [g.get("learner", "") for g in games for _ in g["items"]]
        # ---- values and GAE
        model.eval()
        value.eval()
        values = np.zeros(len(items), dtype=np.float32)
        critic_lens = np.asarray([x["tok"].shape[0] for x in critic_items])
        by_len = np.argsort(critic_lens, kind="stable")
        value_batches=[]
        start=0
        while start<len(items):
            end=min(start+a.infer_batch,len(items))
            value_batches.append(by_len[start:end]);start=end
        with torch.no_grad():
            for sel in value_batches:
                b = stack([critic_items[i] for i in sel], "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    _, hid = model(b, return_hidden=True)
                    predicted = value(hid.float())
                values[sel] = predicted.float().cpu().numpy()
        adv = np.zeros(len(items), dtype=np.float32)
        ret = np.zeros(len(items), dtype=np.float32)
        pos = 0
        for g in games:
            n = len(g["items"])
            v = values[pos:pos + n]
            last_gae = 0.0
            for t in range(n - 1, -1, -1):
                next_v = v[t + 1] if t + 1 < n else 0.0
                r = g["reward"] if t == n - 1 else 0.0
                delta = r + a.gamma * next_v - v[t]
                last_gae = delta + a.gamma * a.lam * last_gae
                adv[pos + t] = last_gae
            ret[pos:pos + n] = adv[pos:pos + n] + v
            pos += n
        phase("values_gae")
        old_logp = np.concatenate([g["old_logp"] for g in games])
        seqs = [s for g in games for s in g["seqs"]]
        excl = [x for g in games for x in g.get("excluded", [None] * len(g["seqs"]))]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        # ---- PPO (eval mode: no dropout, so ratios compare like with like against the rollout log-probabilities)
        model.eval()
        policy_on = it >= a.value_warmup
        kl_coef = max(a.kl_min, a.kl_start * (a.kl_decay ** max(0, it - a.value_warmup)))
        stats = collections.defaultdict(list)
        lens = np.asarray([max(x["tok"].shape[0], c["tok"].shape[0]) for x,c in zip(items,critic_items)])
        for ep in range(a.epochs):
            # Length-bucketed minibatches: shuffle, sort within chunks of 16 minibatches by token count, shuffle the batches.
            perm = np.random.permutation(len(items))
            chunk = a.minibatch * 16
            batches = []
            for c0 in range(0, len(perm), chunk):
                part = perm[c0:c0 + chunk]
                part = part[np.argsort(lens[part], kind="stable")]
                batches += [part[i:i + a.minibatch] for i in range(0,len(part),a.minibatch)]
            random.shuffle(batches)
            microbatches=[]
            for logical in batches:
                start=0
                while start<len(logical):
                    end=len(logical)
                    microbatches.append((logical[start:end],start==0,end==len(logical),(end-start)/len(logical)))
                    start=end
            for n_mb, (mb, first_micro, last_micro, micro_weight) in enumerate(microbatches):
                b = stack([items[i] for i in mb], "cuda", pin=a.fast_batch)
                K = max(1, max(len(seqs[i]) for i in mb))
                if a.fast_batch:
                    # built on the CPU, one pinned host-to-device copy each (no per-row / per-element GPU writes)
                    seq_np = np.zeros((len(mb), K), dtype=np.int64)
                    len_np = np.zeros(len(mb), dtype=np.int64)
                    for r_, i in enumerate(mb):
                        seq_np[r_, :len(seqs[i])] = seqs[i]
                        len_np[r_] = len(seqs[i])
                    pin = lambda x: torch.from_numpy(np.ascontiguousarray(x)).pin_memory().to("cuda", non_blocking=True)
                    seq, seq_len = pin(seq_np), pin(len_np)
                    excluded = None
                    if any(excl[i] for i in mb):
                        ex_np = np.zeros(tuple(b["omask"].shape), dtype=bool)
                        for r_, i in enumerate(mb):
                            for o in excl[i] or []:
                                if o < ex_np.shape[1]:
                                    ex_np[r_, o] = True
                        excluded = pin(ex_np)
                    olp, A, R = pin(old_logp[mb]), pin(adv_n[mb]), pin(ret[mb])
                else:
                    seq = torch.zeros((len(mb), K), dtype=torch.long, device="cuda")
                    seq_len = torch.zeros(len(mb), dtype=torch.long, device="cuda")
                    for r_, i in enumerate(mb):
                        s = seqs[i]
                        seq[r_, :len(s)] = torch.tensor(s, device="cuda")
                        seq_len[r_] = len(s)
                    excluded = None
                    if any(excl[i] for i in mb):
                        excluded = torch.zeros_like(b["omask"])
                        for r_, i in enumerate(mb):
                            for o in excl[i] or []:
                                if o < excluded.shape[1]:
                                    excluded[r_, o] = True
                    olp = torch.from_numpy(old_logp[mb]).cuda()
                    A = torch.from_numpy(adv_n[mb]).cuda()
                    R = torch.from_numpy(ret[mb]).cuda()
                is_none = b["opt"][:, :, 5] > 0
                # --bf16: the policy forward (and so its backward) in bf16 autocast; weights, optimizer, value head,
                # log-probabilities and losses stay fp32
                with torch.set_grad_enabled(policy_on), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    logits, hid = model(b, return_hidden=True)
                v_pred = value(hid.detach().float())
                # --skip-bc: the BC forward only feeds kl_coef * KL; with kl_coef == 0 it is computed on every 16th
                # minibatch, for the kl_bc diagnostic only (no gradient)
                need_bc = bc is not None and not (a.skip_bc and kl_coef == 0)
                diag_bc = bc is not None and (need_bc or n_mb % 16 == 0)
                if diag_bc:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                        bc_logits = bc(b)
                logits = logits.float()
                new_lp = seq_logprob(logits, b["omask"], is_none, seq, seq_len, excluded, sync_free=a.fast_batch)
                ratio = torch.exp(new_lp - olp)
                pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A).mean()
                lp_all = torch.log_softmax(logits, dim=1)
                p_all = lp_all.exp().nan_to_num(0.0)
                ent = -(p_all * lp_all.nan_to_num(0.0)).sum(1).mean()
                kl = None
                if diag_bc:
                    bc_lp = torch.log_softmax(bc_logits.float(), dim=1)
                    bc_p = bc_lp.exp().nan_to_num(0.0)
                    lp_for_kl = lp_all if need_bc else lp_all.detach()
                    kl = (bc_p * (bc_lp - lp_for_kl).nan_to_num(0.0)).sum(1).mean()   # KL(pi_bc || pi)
                vloss = F.mse_loss(v_pred.float(), R)
                loss_pol = pg + (kl_coef * kl if need_bc else 0.0) - a.entropy * ent
                if not torch.isfinite(vloss + loss_pol):
                    raise FloatingPointError("nonfinite PPO/value loss; checkpoint not advanced")
                if first_micro:
                    opt_val.zero_grad(set_to_none=True)
                    opt_pol.zero_grad(set_to_none=True)
                (micro_weight * (a.value_coef * vloss + (loss_pol if policy_on else 0.0))).backward()
                if last_micro:
                    torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0, error_if_nonfinite=True)
                    opt_val.step()
                    if policy_on:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm, error_if_nonfinite=True)
                        opt_pol.step()
                if a.kl_diag:
                    with torch.no_grad():
                        dmb = (olp - new_lp).detach()
                        if float(dmb.mean()) > a.kl_diag:  # a sync, only with --kl-diag
                            top = torch.argsort(dmb, descending=True)[:10].tolist()
                            mx = logits.detach().float().masked_fill(~b["omask"], 0).abs().max(1).values
                            with open(run / "kl-diag.jsonl", "a") as fdiag:
                                fdiag.write(json.dumps({"iteration": it + 1, "epoch": ep, "minibatch": n_mb, "mb_mean": float(dmb.mean()),
                                    "mb_size": len(mb), "top": [{"d": float(dmb[j]), "old": float(olp[j]), "new": float(new_lp[j]),
                                    "type": diag_type[mb[j]], "seq": seqs[mb[j]], "max_abs_logit": float(mx[j]),
                                    "n_opt": int(b["omask"][j].sum()), "acted_by": diag_ver[mb[j]], "adv": float(A[j])} for j in top]}) + "\n")
                with torch.no_grad():
                    # --gpu-stats: keep the per-minibatch values on the GPU (one synchronisation per iteration)
                    val = (lambda t: t.detach().float()) if a.gpu_stats else (lambda t: t.item())
                    stats["pg"].append(val(pg))
                    stats["vloss"].append(val(vloss))
                    if kl is not None:
                        stats["kl_bc"].append(val(kl))
                    stats["entropy"].append(val(ent))
                    stats["clipfrac"].append(val(((ratio - 1).abs() > a.clip).float().mean()))
                    stats["approx_kl_old"].append(val((olp - new_lp).mean()))
                    stats["approx_kl_nonnegative"].append(val((ratio - 1 - (new_lp-olp)).mean()))
                    stats["max_abs_logit"].append(val(logits.masked_fill(~b["omask"],0).abs().max()))
        if a.gpu_stats:
            stats = {k: torch.stack(v).cpu().tolist() for k, v in stats.items()}
        phase("ppo")
        it += 1
        n_games = len(games)
        games_total += n_games
        steps_total += len(items)
        gpu_seconds += time.time() - t_iter
        by_opp = collections.defaultdict(lambda: [0, 0, 0])
        for g in games:
            o = g["opp"].replace("@greedy", "")
            by_opp[o][0] += 1
            by_opp[o][1] += g["won"] if "won" in g else g["reward"] > 0
            by_opp[o][2] += g["o"] != "finished"
        rec = {"iteration": it, "time": time.time(), "wall_hours": (time.time() - t_start) / 3600, "games": n_games,
               "games_total": games_total, "steps": len(items), "steps_total": steps_total, "policy_on": policy_on,
               "kl_coef": kl_coef, "mean_reward": float(np.mean([g["reward"] for g in games])),
               "win_by_opp": {k: {"games": v[0], "win_rate": v[1] / max(1, v[0]), "unfinished": v[2]} for k, v in by_opp.items()},
               "discarded_stale_steps": discarded_stale, "value_mean": float(values.mean()), "return_mean": float(ret.mean()),
               **{k: float(np.mean(v)) for k, v in stats.items()}}
        meta = {"iteration": it, "games": games_total, "steps": steps_total, "gpu_seconds": gpu_seconds,
                "start": str(start_ck), "args": vars(a)}
        path = run / "ckpt" / f"learner-{it:04d}.pt"
        save_rl(path, base_ck, model, value, meta, None)
        shutil.copyfile(path, run / "ckpt" / "latest.tmp")
        os.replace(run / "ckpt" / "latest.tmp", run / "ckpt" / "latest.pt")
        if a.native_ts:
            # TorchScript copies for the in-JVM policy (plan rollout-throughput): latest.ts (+ its iteration) and snapshots
            from .native import export_ts
            export_path = run / "ckpt" / "latest.ts"
            export_ts(model, export_path)
            publication={"iteration": it}
            (run / "ckpt" / "latest.ts.json.tmp").write_text(json.dumps(publication))
            os.replace(run / "ckpt" / "latest.ts.json.tmp", run / "ckpt" / "latest.ts.json")
        if it % a.snap_every == 0:
            shutil.copyfile(path, run / "snaps" / f"snap-{it:04d}.pt")
            if a.native_ts:
                export_ts(model, run / "snaps" / f"snap-{it:04d}.ts")
        for old in sorted((run / "ckpt").glob("learner-*.pt"))[:-a.keep_ckpt]:
            if int(old.stem.split("-")[1]) % a.keep_every != 0:
                old.unlink()
        phase("checkpoint")
        if a.profile_phases:
            rec["phases"] = phases
            rec["phase_memory"] = phase_memory
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: rec.get(k) for k in ("iteration", "games", "steps", "policy_on", "mean_reward", "kl_bc", "entropy",
                                                  "clipfrac", "vloss", "wall_hours")}), json.dumps(rec["win_by_opp"]), flush=True)
        if a.max_iterations and it >= a.max_iterations:
            break
        if a.max_hours and (time.time() - t_start) / 3600 >= a.max_hours:
            print("max hours reached", flush=True)
            break


# ----------------------------------------------------------------------------------------------------------------- eval

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def evaluate(a) -> None:
    out = ROOT / "reports/data/rl"
    out.mkdir(parents=True, exist_ok=True)
    per = (a.games + a.jvms - 1) // a.jvms
    procs, parts = [], []
    stack_log = Path(a.run) / f"exceptions-eval-{a.name}.log"
    for k in range(a.jvms):
        off, n = k * per, min(per, a.games - k * per)
        if n <= 0:
            break
        part = out / f"{a.name}.part{k}.csv"
        parts.append(part)
        cmd = [str(ROOT / "scripts/traingen.sh"), "--schedule", a.schedule, "--games", str(n), "--gameOffset", str(off), "--threads", "1",
               "--seed", str(a.seed), "--agent", a.agent, "--opponent", a.opponent, "--agentSide", "alternate",
               "--server", f"127.0.0.1:{a.port}", "--cap", str(a.cap), "--maxGameMs", str(a.max_game_ms), "--tag", a.name,
               "--stackLog", str(stack_log), "--out", str(part)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out / f"{a.name}.part{k}.log", "w"),
                                      env={**os.environ, "TRAINGEN_HEAP": "3g"}))
    for p in procs:
        p.wait()
    rows = []
    for part in parts:
        rows += list(csv.DictReader(open(part)))
    merged = out / f"{a.name}.csv"
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for part in parts:
        part.unlink()
    print(json.dumps(summarize(merged, a.agent), indent=1))


def summarize(path: Path, agent: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    c = collections.Counter()
    side = collections.defaultdict(collections.Counter)
    for r in rows:
        g = int(r["game"].rsplit("-", 1)[1])
        a_side = "ds" if g % 2 == 0 else "ls"
        fin = r["outcome"] == "finished"
        win = fin and r["winner"] == a_side
        for cc in (c, side[a_side]):
            cc["games"] += 1
            cc["wins"] += win
            cc["finished"] += fin
            cc["errors"] += r["outcome"] not in ("finished", "cap", "timeout")
    lo, hi = wilson(c["wins"], c["games"])
    return {"file": path.name, "agent": agent, "games": c["games"], "wins": c["wins"], "win_rate": c["wins"] / max(1, c["games"]),
            "ci95": [round(lo, 4), round(hi, 4)], "completion": c["finished"] / max(1, c["games"]), "errors": c["errors"],
            "by_agent_side": {k: {"games": v["games"], "win_rate": v["wins"] / max(1, v["games"])} for k, v in side.items()}}


def selfcheck(a) -> None:
    """Greedy keys of the RL server must reproduce offline argmax on packed records (plan §3 #2)."""
    from .train import Packed
    lines = []
    for shard in sorted(glob.glob(str(Path(a.shards) / "shard-*.jsonl.gz"))):
        with gzip.open(shard, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                if r[10] and len(r[9]) >= 2:
                    lines.append(r)
                if len(lines) >= a.k:
                    break
        if len(lines) >= a.k:
            break
    s = socket.create_connection(("127.0.0.1", a.port))
    f = s.makefile("rwb")
    tops = []
    for i, r in enumerate(lines):
        f.write(orjson.dumps({"m": "bc@greedy", "g": "selfcheck", "s": "ds", "k": 1, "r": 0, "rec": r}) + b"\n")
        f.flush()
        tops.append(orjson.loads(f.readline())[0])
    model, vocab = load_checkpoint(a.bc)
    p = Packed(a.packed)
    offline = []
    with torch.no_grad():
        for s0 in range(0, len(lines), 256):
            b = p.batch(np.arange(s0, min(len(lines), s0 + 256)), "cuda")
            offline += torch.argmax(model(b).float(), dim=1).cpu().tolist()
    agree = sum(int(x == y) for x, y in zip(tops, offline))
    print(json.dumps({"records": len(lines), "greedy_vs_offline_top1": agree / len(lines)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", default=str(ROOT / "runs/rl-v1"))
    common.add_argument("--bc", default=str(ROOT / "checkpoints/bc-v1/model.pt"))
    common.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    common.add_argument("--port", type=int, default=47100)
    common.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("server", parents=[common])
    s.add_argument("--max-batch", type=int, default=256)
    s.add_argument("--wait-ms", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--max-snaps", type=int, default=6)
    s.add_argument("--games-per-file", type=int, default=20)
    s.add_argument("--graphs", action="store_true", help="replay CUDA graphs for the forward (bucketed padded shapes)")
    s.add_argument("--guard-threshold", type=int, default=10,
                   help="PolicyAi loop-guard threshold K (Java property policy.loopGuardThreshold); steps with r > K are not recorded")
    l = sub.add_parser("learn", parents=[common])
    l.add_argument('--packed-attention',action='store_true',help='V2 training FlashAttention over all valid tokens')
    l.add_argument('--micro-tokens',type=int,default=32768,help='V2 padded token budget per training/value microbatch')
    l.add_argument("--profile-phases", action="store_true", help="log wall time per phase (GPU synchronised at phase ends)")
    l.add_argument("--fast-batch", action="store_true", help="minibatch tensors built on the CPU, one pinned copy each; no GPU syncs in seq_logprob")
    l.add_argument("--gpu-stats", action="store_true", help="per-minibatch statistics stay on the GPU (one sync per iteration)")
    l.add_argument("--skip-bc", action="store_true", help="skip the BC forward when kl_coef == 0 (kl_bc diagnostic on every 16th minibatch)")
    l.add_argument("--fused-adam", action="store_true", help="fused AdamW kernels")
    l.add_argument("--kl-diag", type=float, default=0.0,
                   help="log the top steps of every minibatch whose mean(old - new log-prob) exceeds this to kl-diag.jsonl (0 = off)")
    l.add_argument("--bf16", action="store_true", help="bf16 autocast for the policy forward/backward (fp32 weights, log-probs, losses)")
    l.add_argument("--load-workers", type=int, default=1, help="processes parsing trajectory files (1 = in the learner)")
    l.add_argument("--seed-all", action="store_true", help="also seed numpy and random (reproducible minibatch order)")
    l.add_argument("--native-ts", action="store_true",
                   help="also write ckpt/latest.ts and snaps/snap-N.ts (TorchScript) for the in-JVM policy backend")
    l.add_argument("--max-policy-lag", type=int, default=3)
    l.add_argument("--keep-snapshots", type=int, default=6)
    l.add_argument("--batch-steps", type=int, default=150_000)
    l.add_argument("--bytes-per-step-estimate", type=float, default=900.0)
    l.add_argument("--epochs", type=int, default=1)  # user 2026-10-03: one PPO pass per batch
    l.add_argument("--minibatch", type=int, default=256)
    l.add_argument("--infer-batch", type=int, default=256)
    l.add_argument("--lr", type=float, default=1e-5)
    l.add_argument("--value-lr", type=float, default=3e-4)
    l.add_argument("--value-coef", type=float, default=1.0)
    l.add_argument("--value-warmup", type=int, default=2)
    l.add_argument("--clip", type=float, default=0.2)
    l.add_argument("--gamma", type=float, default=1.0)
    l.add_argument("--lam", type=float, default=0.95)
    l.add_argument("--entropy", type=float, default=0.003)
    l.add_argument("--kl-start", type=float, default=0.2)
    l.add_argument("--kl-decay", type=float, default=0.9)
    l.add_argument("--kl-min", type=float, default=0.02)
    l.add_argument("--max-grad-norm", type=float, default=0.5)
    l.add_argument("--snap-every", type=int, default=5)
    l.add_argument("--keep-ckpt", type=int, default=4)
    l.add_argument("--keep-every", type=int, default=5)
    l.add_argument("--keep-consumed", type=int, default=20)
    l.add_argument("--max-iterations", type=int, default=0)
    l.add_argument("--max-hours", type=float, default=0.0)
    e = sub.add_parser("eval", parents=[common])
    e.add_argument("--schedule", required=True)
    e.add_argument("--agent", required=True)
    e.add_argument("--opponent", required=True)
    e.add_argument("--games", type=int, default=600)
    e.add_argument("--jvms", type=int, default=7)
    e.add_argument("--cap", type=int, default=6000)
    e.add_argument("--max-game-ms", type=int, default=900000)
    e.add_argument("--name", required=True)
    e.set_defaults(seed=881000)
    c = sub.add_parser("selfcheck", parents=[common])
    c.add_argument("--shards", default=str(ROOT / "data/shards/v2-eval_seen"))
    c.add_argument("--packed", default=str(ROOT / "data/packed/v2-eval_seen"))
    c.add_argument("--k", type=int, default=1000)
    a = ap.parse_args()
    {"server": serve, "learn": learn, "eval": evaluate, "selfcheck": selfcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
