"""Learning-curve plot for an RL run: greedy win rate vs greedy bc-v1 (curve.jsonl, with 95 % CIs) over games played and
GPU hours, plus the rollout (sampled) win rates by opponent from learner.jsonl.

    uv run python -m swccg_policy.rl_plot --run runs/rl-v1 --out reports/data/rl/learning_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rl-v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    learner = [json.loads(l) for l in open(run / "learner.jsonl")]
    by_it = {r["iteration"]: r for r in learner}
    curve = sorted((json.loads(l) for l in open(run / "curve.jsonl")), key=lambda d: d["iteration"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ink, accent, muted = "#1b2130", "#0d6b73", "#8a93a3"
    for ax, xkey, xlabel in ((axes[0], "games_total", "rollout games played (game-sides recorded)"),
                             (axes[1], "wall_hours", "wall-clock hours (GPU shared with the rollout server)")):
        xs = [0] + [by_it[d["iteration"]][xkey] for d in curve if d["iteration"] in by_it]
        ys = [0.5] + [d["win_rate"] for d in curve if d["iteration"] in by_it]
        lo = [0.5] + [d["ci95"][0] for d in curve if d["iteration"] in by_it]
        hi = [0.5] + [d["ci95"][1] for d in curve if d["iteration"] in by_it]
        ax.fill_between(xs, lo, hi, color=accent, alpha=0.18, linewidth=0, label="95 % CI")
        ax.plot(xs, ys, color=accent, marker="o", markersize=4, linewidth=2, label="greedy RL vs greedy bc-v1")
        for opp, color, label in (("bc", muted, "rollouts: sampled vs sampled bc"),
                                  ("bot", "#b5651d", "rollouts: sampled vs AdvancedAi"),
                                  ("snap", "#6b4c9a", "rollouts: sampled vs past snapshots")):
            roll = [(r[xkey], r["win_by_opp"][opp]["win_rate"]) for r in learner if opp in r["win_by_opp"]]
            if roll:
                ax.plot([x for x, _ in roll], [y for _, y in roll], color=color, linewidth=1, alpha=0.6, label=label)
        ax.axhline(0.5, color=ink, linewidth=0.8, linestyle="--")
        ax.axhline(0.55, color=ink, linewidth=0.6, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("win rate")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Self-play PPO from bc-v1: win rate against the frozen imitation policy", fontsize=11)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
