"""Real decipher-fixture logits/value parity across the complete virtual table extension."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import new_models, save
from swccg_policy.v4_data import parse, stack


def records(directory):
    for path in sorted(directory.glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/knowledge-v3.1/decipher-growth-candidate.pt"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/data/knowledge-v3.1/decipher-model-growth.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path("runs/knowledge-v3.1")
    old = TableBundle.load(root / "tables-decipher")
    new = TableBundle.load(root / "tables-decipher-virtual-extension")
    actor, critic = new_models(old)
    actor.to(args.device).eval()
    critic.to(args.device).eval()
    checkpoint = args.checkpoint
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    save(checkpoint, actor, critic, old, {"purpose": "untrained table-growth check; D selection pending"})
    source = root / "grid-decipher/fixtures"
    fixtures = list(records(source))
    inputs = [(parse(record, old), parse(record, old, critic=True)) for record in fixtures]
    before = []
    with torch.no_grad():
        actor.cache_card_table()
        critic.encoder.cache_card_table()
        for acting, privileged in inputs:
            before.append((actor(stack([acting], args.device)).cpu(),
                           critic(stack([privileged], args.device)).cpu()))
        actor.grow(old, new)
        critic.encoder.grow(old, new)
        actor.cache_card_table()
        critic.encoder.cache_card_table()
        for index, ((acting, privileged), (logits, value)) in enumerate(zip(inputs, before, strict=True)):
            grown_logits = actor(stack([acting], args.device)).cpu()
            grown_value = critic(stack([privileged], args.device)).cpu()
            if not torch.equal(logits, grown_logits) or not torch.equal(value, grown_value):
                raise AssertionError(f"Full-policy growth changed old-table observation {index}")
        virtual_checked = 0
        for record in records(root / "grid-final/fixtures"):
            virtual = any(token[1].split("_", 1)[0].isdigit()
                          and int(token[1].split("_", 1)[0]) >= 200 for token in record[8])
            if not virtual:
                continue
            try:
                parse(record, old)
            except ValueError as error:
                if "Virtual blueprint" not in str(error):
                    raise
            else:
                raise AssertionError("Decipher-only model admitted virtual observation")
            output = actor(stack([parse(record, new)], args.device))
            if not torch.isfinite(output).all():
                raise AssertionError("Grown virtual observation produced nonfinite logits")
            virtual_checked += 1
            if virtual_checked == 16:
                break
    if not inputs or virtual_checked == 0:
        raise ValueError("Required decipher/virtual fixtures missing")
    report = {"passed": True, "device": args.device, "decipher_decisions": len(inputs),
              "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
              "actor_logits_bit_identical": True, "critic_values_bit_identical": True,
              "virtual_observations_refused_before_accepted_after": virtual_checked,
              "old_table_sha": old.sha, "new_table_sha": new.sha, "cfg": actor.cfg.to_dict(),
              "comparison": "Identical old-table tensors; newly admitted text keys may change future re-encodings.",
              "fixture_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source.glob("*.jsonl.gz")},
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                [Path(__file__), *Path("src/swccg_policy").glob("*.py")]}}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("sha256")}))


if __name__ == "__main__":
    main()
