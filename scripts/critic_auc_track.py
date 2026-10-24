"""How well a run's critic predicts the winner, per checkpoint: rank AUC of the value against the game result.

    critic_auc_track.py --out JSON TRAJECTORY_DIR CHECKPOINT [CHECKPOINT ...]

Every checkpoint scores the same decided games (terminal reward +-1) of TRAJECTORY_DIR (a run's kept trajectories,
written with its tables). Reported: AUC over all decisions, at each seat's first decision, and per fifth of the game
(decision index / game length), plus the mean value per fifth. A late checkpoint has trained on the run's last games
(one PPO pass), so its numbers on them are slightly in-sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swccg_policy.ppo_step import PPOConfig
from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_rl import predict_values, rank_auc, win_labels
from swccg_policy.v4_trajectory import load_binary


def score(checkpoint, files):
    _, critic, tables, payload = load(checkpoint, "cuda")
    games = [game for path in files for game in load_binary(path, tables)]
    games = [game for game in games if float(game["reward"]) in (1.0, -1.0)]
    items = [item for game in games for item in game["critic_items"]]
    labels = win_labels(games)
    progress = np.concatenate([np.arange(len(game["critic_items"])) / len(game["critic_items"]) for game in games])
    first = np.concatenate([np.arange(len(game["critic_items"])) == 0 for game in games])
    values = predict_values(critic, items, PPOConfig(), 256)
    fifths = {}
    for part in range(5):
        mask = (progress >= part / 5) & (progress < (part + 1) / 5)
        fifths[f"{20 * part}-{20 * part + 20}%"] = {"auc": rank_auc(values[mask], labels[mask] == 1),
                                                    "decisions": int(mask.sum())}
    meta = payload.get("rl_meta") or {}
    return {"iteration": meta.get("iteration", 0), "games_trained": meta.get("games"), "games_scored": len(games),
            "decisions": len(items), "auc_all": rank_auc(values, labels == 1),
            "auc_first_decision": rank_auc(values[first], labels[first] == 1), "by_fifth": fifths}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="plan critic-counterfactuals: score every N-th file (sorted), so a 600k probe fits in RAM")
    args = parser.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    files = sorted(args.trajectories.glob("*.traj"))[::args.every]
    report = {}
    for checkpoint in args.checkpoints:
        report[str(checkpoint)] = score(checkpoint, files)
        print(json.dumps({"checkpoint": str(checkpoint), **report[str(checkpoint)]}), flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
