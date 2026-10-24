"""Green/red health check for a V3.1 RL pilot before continuing to the eight-hour stage (Amendment 10b (c)).

Exit 0 only if the pilot completed its duration and its learner metrics stay inside conservative bounds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

LIMITS = {
    "min_iterations": 20,
    "entropy_min": 0.5,        # no policy collapse
    "entropy_drop_max": 0.35,  # late mean may not fall more than this below the early mean
    "kl_p95_max": 0.05,        # approx_kl_old
    "clipfrac_p95_max": 0.2,
    "max_abs_logit_max": 50.0,  # runaway confidence
    "vloss_max": 0.5,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = json.loads((args.evidence / "result.json").read_text())
    rows = [json.loads(line) for line in (args.evidence / "learner.jsonl").read_text().splitlines()]
    trained = [row for row in rows if row.get("policy_on")]
    if not trained:
        # A learner that crashed before any policy update is RED, with a report instead of a traceback.
        report = {"evidence": str(args.evidence), "limits": LIMITS, "green": False, "iterations": len(rows),
                  "checks": {"learner_exit_0": result["learner_returncode"] == 0, "policy_updates": False},
                  "reason": result["reason"]}
        (args.evidence / "health.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        raise SystemExit(1)
    entropy = np.array([row["entropy"] for row in trained])
    kl = np.array([row["approx_kl_old"] for row in trained])
    clipfrac = np.array([row["clipfrac"] for row in trained])
    quarter = max(1, len(entropy) // 4)
    checks = {
        "duration_complete": result["reason"] == "duration_complete",
        "learner_exit_0": result["learner_returncode"] == 0,
        "iterations": len(rows) >= LIMITS["min_iterations"],
        "entropy_floor": bool(entropy.min() >= LIMITS["entropy_min"]),
        "entropy_stable": bool(entropy[:quarter].mean() - entropy[-quarter:].mean() <= LIMITS["entropy_drop_max"]),
        "kl_p95": bool(np.quantile(kl, 0.95) <= LIMITS["kl_p95_max"]),
        "clipfrac_p95": bool(np.quantile(clipfrac, 0.95) <= LIMITS["clipfrac_p95_max"]),
        "max_abs_logit": max(row["max_abs_logit"] for row in rows) <= LIMITS["max_abs_logit_max"],
        "vloss": max(row["vloss"] for row in rows) <= LIMITS["vloss_max"],
        "finite": all(np.isfinite([row[key] for key in ("entropy", "approx_kl_old", "vloss", "max_abs_logit")]).all()
                      for row in rows),
    }
    report = {"evidence": str(args.evidence), "limits": LIMITS, "checks": checks, "green": all(checks.values()),
              "iterations": len(rows), "decisions": rows[-1]["steps_total"], "games": rows[-1]["games_total"],
              "entropy": [float(entropy.min()), float(entropy.mean()), float(entropy.max())],
              "kl_p95": float(np.quantile(kl, 0.95)), "clipfrac_p95": float(np.quantile(clipfrac, 0.95)),
              "max_abs_logit": max(row["max_abs_logit"] for row in rows),
              "max_vloss": max(row["vloss"] for row in rows)}
    (args.evidence / "health.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["green"] else 1)


if __name__ == "__main__":
    main()
