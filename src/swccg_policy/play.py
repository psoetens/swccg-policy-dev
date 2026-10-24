"""Play-strength runs through the Gemp harness (TrainGen) with the policy server.

Each pairing plays the same seeded game set (game index -> deck pair, agent side, shuffle seed), split over parallel
single-threaded JVMs (seeded shuffles need one thread per JVM). The agent alternates sides (even games DS).

    uv run python -m swccg_policy.play --schedule data/schedules/play_bench.tsv --agent policy --opponent beginner \
        --games 300 --name bench-policy-vs-beginner --port 47000
    uv run python -m swccg_policy.play --summarize reports/data/play/*.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(a) -> Path:
    out = ROOT / "reports/data/play"
    out.mkdir(parents=True, exist_ok=True)
    per = (a.games + a.jvms - 1) // a.jvms
    procs, parts = [], []
    for k in range(a.jvms):
        off, n = k * per, min(per, a.games - k * per)
        if n <= 0:
            break
        part = out / f"{a.name}.part{k}.csv"
        parts.append(part)
        cmd = [str(ROOT / "scripts/traingen.sh"), "--schedule", a.schedule, "--games", str(n), "--gameOffset", str(off), "--threads", "1",
               "--seed", str(a.seed), "--agent", a.agent, "--opponent", a.opponent, "--agentSide", "alternate",
               "--server", f"127.0.0.1:{a.port}", "--cap", str(a.cap), "--maxGameMs", str(a.max_game_ms), "--tag", a.name,
               "--out", str(part)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out / f"{a.name}.part{k}.log", "w"),
                                      env={**__import__("os").environ, "TRAINGEN_HEAP": "3g"}))
    for p in procs:
        p.wait()
    merged = out / f"{a.name}.csv"
    rows = []
    for part in parts:
        rows += list(csv.DictReader(open(part)))
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for part in parts:
        part.unlink()
    print(json.dumps(summarize_file(merged, a.agent), indent=1))
    return merged


def summarize_file(path: Path, agent: str | None = None) -> dict:
    rows = list(csv.DictReader(open(path)))
    if agent is None:  # the agent plays DS in even-indexed games (--agentSide alternate)
        agent = next(r["ds_ai"] for r in rows if int(r["game"].rsplit("-", 1)[1]) % 2 == 0)
    tot = collections.Counter()
    by_side = collections.defaultdict(collections.Counter)
    by_match = collections.defaultdict(collections.Counter)
    for r in rows:
        agent_side = "ds" if r["ds_ai"] == agent else "ls"
        win = r["winner"] == agent_side
        fin = r["outcome"] == "finished"
        for c in (tot, by_side[agent_side], by_match[f"{Path(r['ds_deck']).stem} vs {Path(r['ls_deck']).stem}"]):
            c["games"] += 1
            c["wins"] += win
            c["finished"] += fin
            c["cap"] += r["outcome"] in ("cap", "timeout")
            c["error"] += r["outcome"] not in ("finished", "cap", "timeout")
    f = lambda c: {"games": c["games"], "win_rate": c["wins"] / c["games"], "completion": c["finished"] / c["games"],
                   "cap_or_timeout": c["cap"], "errors": c["error"],
                   "win_rate_finished": c["wins"] / c["finished"] if c["finished"] else None}
    return {"file": path.name, "agent": agent, "all": f(tot), "by_agent_side": {k: f(v) for k, v in by_side.items()},
            "by_matchup": {k: f(v) for k, v in sorted(by_match.items())} if len(by_match) <= 12 else f"{len(by_match)} matchups"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule")
    ap.add_argument("--agent", default="policy")
    ap.add_argument("--opponent", default="beginner")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--jvms", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777000)
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--max-game-ms", type=int, default=900000)
    ap.add_argument("--name")
    ap.add_argument("--summarize", nargs="*")
    a = ap.parse_args()
    if a.summarize:
        print(json.dumps([summarize_file(Path(p)) for p in a.summarize], indent=1))
    else:
        run(a)


if __name__ == "__main__":
    main()
