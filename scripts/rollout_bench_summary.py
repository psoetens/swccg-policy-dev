#!/usr/bin/env python3
"""Summarise one scripts/rollout_bench.sh run directory as JSON: games/s and policy latency (TrainGen), CPU (machine and per
process, cores), GPU utilisation (nvidia-smi dmon), perf counters per game, outcomes."""
import csv
import json
import math
import re
import sys
from pathlib import Path


def perf(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) >= 3 and p[0] and not line.startswith("#"):
            try:
                out[p[2].split(":")[0]] = float(p[0])
            except ValueError:
                pass
    return out


def main(d: Path) -> dict:
    s = {"name": d.name, "config": (d / "config.txt").read_text().strip()}
    tg = (d / "traingen.log").read_text()
    m = re.search(r"(\d+) games on (\d+) threads in ([\d.]+) s = ([\d.]+) games/s.*policy queries (\d+) \(([\d.]+) ms avg\), "
                  r"fallbacks (\d+), loop-guard (\d+), real-loops (\d+), rejected (\d+), decisions (\d+)", tg)
    if m:
        g, th, secs, gps, q, qms, fb, lg, rl, rj, dec = m.groups()
        s.update(games=int(g), threads=int(th), seconds=float(secs), games_per_s=float(gps), policy_queries=int(q),
                 query_ms_avg=float(qms), fallbacks=int(fb), loop_guard=int(lg), decisions=int(dec))
        s["thread_wait_fraction"] = int(q) * float(qms) / 1000 / int(th) / float(secs)
    m = re.search(r"native policy: (\d+) requests, (\d+) batches \(([\d.]+) per batch\), (\d+) forwards \(([\d.]+) ms avg\), (\d+) failures", tg)
    if m:
        s["native"] = {"requests": int(m[1]), "batches": int(m[2]), "per_batch": float(m[3]), "forwards": int(m[4]),
                       "forward_ms": float(m[5]), "failures": int(m[6])}
    m = re.search(r"policy latency ms: p50 ([\d.]+) p90 ([\d.]+) p99 ([\d.]+) p99.9 ([\d.]+)", tg)
    if m:
        s["latency_ms"] = dict(zip(["p50", "p90", "p99", "p999"], map(float, m.groups())))
    rows = list(csv.DictReader(open(d / "games.csv")))
    s["outcomes"] = {}
    for r in rows:
        s["outcomes"][r["outcome"]] = s["outcomes"].get(r["outcome"], 0) + 1
    t0, t1 = float((d / "t_start").read_text()), float((d / "t_end").read_text())
    samp = [r for r in csv.DictReader(open(d / "samples.csv")) if t0 + 20 <= float(r["t"]) <= t1 - 5]  # skip JIT warm-up
    if len(samp) > 2:
        s["machine_busy"] = sum(float(r["machine_busy"]) for r in samp) / len(samp)
        span = float(samp[-1]["t"]) - float(samp[0]["t"])
        for k in samp[0]:
            if k in ("t", "machine_busy", "machine_cores"):
                continue
            a, b = float(samp[0][k]), float(samp[-1][k])
            if not (math.isnan(a) or math.isnan(b)):
                s[f"cores_{k}"] = (b - a) / span
    gpu = []
    for line in (d / "gpu.txt").read_text().splitlines():
        p = line.split()
        if p and p[0].isdigit() and len(p) > 1 and p[1].isdigit():
            gpu.append(int(p[1]))
    if gpu:
        gpu = gpu[20:-5] or gpu
        s["gpu_util_avg"] = sum(gpu) / len(gpu)
    for who in ("java", "server"):
        c = perf(d / f"perf-{who}.csv")
        if c and s.get("games_per_s") and (d / "t_perf").exists():
            # perf covers t_perf..t_end; per-game figures = rate / games per second (TrainGen's rate over the whole run)
            span = t1 - float((d / "t_perf").read_text())
            rate = {k: v / span for k, v in c.items()}
            gps = s["games_per_s"]
            s[f"perf_{who}"] = {"instructions_per_game": rate.get("instructions", 0) / gps,
                                "cycles_per_game": rate.get("cycles", 0) / gps,
                                "ipc": c.get("instructions", 0) / max(1, c.get("cycles", 1)),
                                "cores": rate.get("task-clock", 0) / 1000,
                                "cpu_s_per_game": rate.get("task-clock", 0) / 1000 / gps,
                                "ctx_switches_per_s": rate.get("context-switches", 0)}
            if who == "java" and s.get("decisions"):
                s["perf_java"]["cpu_us_per_decision"] = s["perf_java"]["cpu_s_per_game"] * 1e6 * s["games"] / s["decisions"]
    return s


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
