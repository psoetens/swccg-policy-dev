"""Plan option-inputs-and-balance-probe §1.3: an extended checkpoint computes exactly what its parent did.

    check_option_extension.py PARENT.pt EXTENDED.pt --records DIR --out JSON

Both actors score the decisions of D4's fixed set (exported AdvancedAI-game records, at least two options); reported:
decisions, the maximum |logit difference|, and the parameters the extension added (every branch's output zero).
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_diagnostics import records  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import parse, stack  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", type=Path)
    parser.add_argument("extended", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent, _, tables, _ = load(args.parent, "cuda")
    extended, _, extended_tables, _ = load(args.extended, "cuda")
    if extended.cfg.option_inputs != "v2" or parent.cfg.option_inputs != "legacy":
        raise ValueError("expected a legacy parent and a v2 extension")
    width = 8 if parent.cfg.tensor_abi == 5 else 4
    added = {name: float(value.abs().max()) for name, value in extended.named_parameters()
             if name not in dict(parent.named_parameters())}
    for model in (parent, extended):
        model.eval()
        model.cache_card_table()
    worst, count = 0.0, 0
    with torch.no_grad():
        for record in records(args.records):
            if len(record[9]) < 2:
                continue
            batch = stack([parse(record, tables, width=width)], "cuda")
            difference = (parent(batch).float() - extended(batch).float()).abs()[batch["omask"]]
            worst = max(worst, float(difference.max()))
            count += 1
            if count >= args.limit:
                break
    result = {"decisions": count, "max_abs_logit_difference": worst, "added_parameters": added,
              # Every new branch ends in a zero layer; a hidden layer inside a branch (option_source_location.0)
              # keeps its normal initialisation.
              "passed": worst <= 1e-5 and all(value == 0.0 for name, value in added.items()
                                              if not name.startswith("option_source_location.0"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
