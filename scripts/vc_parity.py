"""Plan v-cards §1.2 gate: the grown checkpoint acts exactly like its parent on the parent's own decisions.
Reads decision records the parent's run wrote (binary trajectories, parent tables), runs actor and critic of both
checkpoints on them, and reports the largest absolute logit and value differences. Gate (Amendment 1): the same
greedy option on every decision, and differences <= 2x the parent's own float noise floor (batch 1 vs batch 32 on the
first 256 decisions, measured in the same run).
    PYTHONPATH=scripts:src .venv/bin/python scripts/vc_parity.py PARENT GROWN TRAJ_DIR OUT_JSON [--files 4]
"""
import argparse
import json
from pathlib import Path

import torch

from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import stack
from swccg_policy.v4_trajectory import TrajectoryReader


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", type=Path)
    parser.add_argument("grown", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--files", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    parent_actor, parent_critic, parent_tables, _ = load(args.parent, device=device)
    grown_actor, grown_critic, grown_tables, _ = load(args.grown, device=device)
    parent_tables.assert_prefix_of(grown_tables)
    reader = TrajectoryReader(parent_tables)
    files = sorted(args.trajectories.glob("*.traj"))[:args.files]
    games = [game for path in files for game in reader.load(path)]
    items = [item for game in games for item in game["items"]]
    critic_items = [item for game in games for item in game["critic_items"]]
    worst = {"logit": 0.0, "value": 0.0}
    floor = {"logit": 0.0, "value": 0.0}
    choices_changed = 0

    def first(output):
        return output[0] if isinstance(output, tuple) else output
    with torch.no_grad():
        for model in (parent_actor, grown_actor):
            if hasattr(model, "cache_card_table"):
                model.cache_card_table()
        for model in (parent_critic, grown_critic):
            if hasattr(model.encoder, "cache_card_table"):
                model.encoder.cache_card_table()
        for start in range(0, len(items), args.batch):
            batch = stack(items[start:start + args.batch], device)
            a, b = first(parent_actor(batch)), first(grown_actor(batch))
            choices_changed += int((a.argmax(-1) != b.argmax(-1)).sum())
            finite = torch.isfinite(a) & torch.isfinite(b)
            if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
                raise ValueError("masked options differ")
            worst["logit"] = max(worst["logit"], float((a[finite] - b[finite]).abs().max()))
            batch = stack(critic_items[start:start + args.batch], device)
            v, w = first(parent_critic(batch)), first(grown_critic(batch))
            worst["value"] = max(worst["value"], float((v - w).abs().max()))
        # the parent's own noise floor: the same decisions one at a time
        for start in range(0, min(256, len(items)), args.batch):
            stop = min(start + args.batch, len(items))
            a = first(parent_actor(stack(items[start:stop], device)))
            v = first(parent_critic(stack(critic_items[start:stop], device)))
            for k in range(start, stop):
                one = first(parent_actor(stack([items[k]], device)))[0]
                row = a[k - start][:one.shape[0]]
                mask = torch.isfinite(row) & torch.isfinite(one)
                floor["logit"] = max(floor["logit"], float((row[mask] - one[mask]).abs().max()))
                single = first(parent_critic(stack([critic_items[k]], device))).reshape(-1)
                floor["value"] = max(floor["value"], float((v[k - start].reshape(-1) - single).abs().max()))
    result = {"parent": str(args.parent), "grown": str(args.grown), "files": [str(p) for p in files],
              "games": len(games), "decisions": len(items), "max_abs_logit_difference": worst["logit"],
              "max_abs_value_difference": worst["value"], "greedy_choices_changed": choices_changed,
              "noise_floor": floor,
              "passed": choices_changed == 0 and worst["logit"] <= 2 * floor["logit"]
              and worst["value"] <= 2 * floor["value"]}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
