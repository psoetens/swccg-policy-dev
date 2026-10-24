"""Summarise RL runs from their learner.jsonl: whole-run speed and waste, and learning signals per game window.

    learner_windows.py --out JSON NAME=LEARNER_JSONL [NAME=LEARNER_JSONL ...] [--windows 0,2000,6000,12000,18000,24200]

Whole run: updates, games, decisions trained on, decisions per game, stale-discarded decisions and their share of all
decisions produced, hours, median seconds per update, decisions per second, the learner's median wait for data, median
KL and clipfrac (policy updates), last entropy and maximum logit, median and last explained variance.
Per window (by games_total): medians of KL, clipfrac, entropy, maximum logit and explained variance, and the mean of the
per-update first-decision and 0-20% critic AUCs where the log has them (plan actor-v7-recipe).
"""
import argparse
import json
import statistics
from pathlib import Path


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summary(rows):
    policy = [row for row in rows if row.get("policy_on")]
    decisions = sum(row["steps"] for row in rows)
    stale = sum(row.get("discarded_stale_steps", 0) for row in rows)
    games = rows[-1]["games_total"]
    seconds = rows[-1]["elapsed"]
    return {"updates": len(rows), "games": games, "decisions": decisions,
            "decisions_per_game": round(decisions / games, 1), "stale": stale,
            "stale_share": round(stale / (decisions + stale), 3), "hours": round(seconds / 3600, 2),
            "update_s_median": round(median(row["update_seconds"] for row in rows), 2),
            "decisions_per_s": round(decisions / seconds, 1),
            "queue_wait_median": round(median(row.get("queue_wait_seconds") for row in rows), 2),
            "kl_median": median(row["approx_kl_old"] for row in policy),
            "clipfrac_median": median(row["clipfrac"] for row in policy),
            "entropy_last": round(rows[-1]["entropy"], 3), "max_logit_last": round(rows[-1]["max_abs_logit"], 2),
            "ev_median": round(median(row.get("value_explained_variance") for row in rows), 3),
            "ev_last": round(rows[-1]["value_explained_variance"], 3)}


def window(rows, low, high):
    part = [row for row in rows if low < row["games_total"] <= high]
    policy = [row for row in part if row.get("policy_on")]
    return {"updates": len(part), "decisions": sum(row["steps"] for row in part),
            "kl": median(row["approx_kl_old"] for row in policy), "clipfrac": median(row["clipfrac"] for row in policy),
            "entropy": median(row["entropy"] for row in part), "max_logit": median(row["max_abs_logit"] for row in part),
            "ev": median(row.get("value_explained_variance") for row in part),
            "first_auc": mean(row.get("value_first_auc") for row in part),
            "early_auc": mean(row.get("value_early_auc") for row in part)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="NAME=path/to/learner.jsonl")
    parser.add_argument("--windows", default="0,2000,6000,12000,18000,24200")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    edges = [int(edge) for edge in args.windows.split(",")]
    report = {}
    for run in args.runs:
        name, path = run.split("=", 1)
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        report[name] = {"summary": summary(rows),
                        "windows": {f"{low}-{high}": window(rows, low, high) for low, high in zip(edges, edges[1:])}}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
