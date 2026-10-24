"""Plan v-cards §1.2: zero the structured card-feature weight columns no trained card ever set.
Columns that are 0 on every row of the parent checkpoint's card table never received a gradient, so their weights are
the random initialisation; V cards that set them would get a meaningless shift. Zeroing them leaves the function on the
parent's cards unchanged (their inputs there are 0). Applies to every '*cards.structured.weight' in the actor and critic.
    PYTHONPATH=scripts:src .venv/bin/python scripts/vc_zero_untrained.py PARENT GROWN OUT
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", type=Path, help="the checkpoint before growth (defines the trained cards)")
    parser.add_argument("grown", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    grown = torch.load(args.grown, map_location="cpu", weights_only=False)
    rows = len(parent["tables"]["cards"]["ids"])
    names = [str(n) for n in parent["tables"]["cards"]["feature_names"]]
    zeroed = {}
    for part in ("state", "value_state"):
        state = grown.get(part) or {}
        for key in [k for k in state if k.endswith("cards.structured.weight")]:
            prefix = key[:-len("structured.weight")]
            features = state[prefix + "features"]
            structured = state[prefix + "structured_indices"]
            used = (features[:rows][:, structured] != 0).any(0)
            weight = state[key].clone()
            weight[:, ~used] = 0
            state[key] = weight
            zeroed[f"{part}.{key}"] = [names[int(i)] for i in structured[~used]]
    meta = grown.setdefault("rl_meta", {})
    meta["untrained_feature_columns_zeroed"] = {key: len(value) for key, value in zeroed.items()}
    torch.save(grown, args.out)
    print(json.dumps({"out": str(args.out), "zeroed": {key: len(value) for key, value in zeroed.items()}}))
    report = args.out.with_suffix(".zeroed.json")
    report.write_text(json.dumps(zeroed, indent=1) + "\n")


if __name__ == "__main__":
    main()
