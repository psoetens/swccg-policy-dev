"""Full imitation evaluation for the report: accuracy vs baselines per decision type on
  - eval_seen:    held-out games (disjoint from training games) between training decks, all decisions;
  - eval_heldout: games with a held-out deck, only the decisions of the player using the held-out deck.

    uv run python -m swccg_policy.report_eval --checkpoint checkpoints/bc-v1/model.pt --train data/packed/v1-train \
        --seen data/packed/v1-eval_seen --heldout data/packed/v1-eval_heldout --out reports/data/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train import Packed, baseline_tables, evaluate, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def heldout_indices(split: Packed, schedule: Path) -> np.ndarray:
    """Records whose deciding side plays the held-out deck (per game, from the eval_heldout schedule and pool.json)."""
    held = set(json.load(open(ROOT / "data/schedules/pool.json"))["heldout"])
    lines = [l.rstrip("\n").split("\t") for l in open(schedule)]
    games = json.load(open(split.path / "games.json"))
    side_of_game = []
    for gid in games:
        g = int(gid.rsplit("-", 1)[1])
        ds, ls = (Path(p).stem for p in lines[g])
        side_of_game.append(0 if ds in held else 1 if ls in held else -1)
    rec = np.asarray(split.rec)
    want = np.array(side_of_game)[rec[:, 8]]
    return np.nonzero(rec[:, 3] == want)[0]


def era_indices(split: Packed, schedule: Path) -> dict[str, np.ndarray]:
    """Record indices per era of the deciding player's deck (from <schedule>.games.json written by decks.py)."""
    meta_path = schedule.with_suffix(".games.json")
    if not meta_path.exists():
        return {}
    meta = json.load(open(meta_path))
    games = json.load(open(split.path / "games.json"))
    eras = [(meta[int(g.rsplit("-", 1)[1])]["ds_era"], meta[int(g.rsplit("-", 1)[1])]["ls_era"]) for g in games]
    rec = np.asarray(split.rec)
    era_of = np.array([eras[g][side] for g, side in zip(rec[:, 8], rec[:, 3])])
    return {e: np.nonzero(era_of == e)[0] for e in sorted(set(era_of))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--seen", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--schedule", default=str(ROOT / "data/schedules/eval_heldout.tsv"))
    ap.add_argument("--seen-schedule", default=str(ROOT / "data/schedules/eval_seen.tsv"))
    ap.add_argument("--out", default=str(ROOT / "reports/data/eval.json"))
    a = ap.parse_args()
    model, _ = load_checkpoint(a.checkpoint)
    tables = baseline_tables(Packed(a.train))
    seen = Packed(a.seen)
    held = Packed(a.heldout)
    hi = heldout_indices(held, Path(a.schedule))
    res = {"eval_seen": evaluate(model, seen, "cuda", tables),
           "eval_heldout_decks": evaluate(model, held, "cuda", tables, indices=hi)}
    for era, idx in era_indices(seen, Path(a.seen_schedule)).items():
        res[f"eval_seen_era_{era}"] = {"ALL": evaluate(model, seen, "cuda", tables, indices=idx)["ALL"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k)
        for t, m in sorted(v.items(), key=lambda x: -x[1]["n"]):
            print(f"  {t:20s} n={m['n']:7d} top1={m['top1']:.3f} top3={m['top3']:.3f} random={m['random']:.3f} "
                  f"freq_t={m['freq_t']:.3f} freq_tb={m['freq_tb']:.3f} first(positional)={m['first']:.3f}")


if __name__ == "__main__":
    main()
