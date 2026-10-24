"""Export the candidate ABI-4/5 actor and check ONNX CPU inference against PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_onnx import INPUTS, export
from swccg_policy.v4_parity import greedy_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/knowledge-v3.1/decipher-onnx-check"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/knowledge-v3.1/decipher-growth-candidate.pt"))
    parser.add_argument("--records", type=Path, default=Path("runs/knowledge-v3.1/decipher-abi-check-r3/records.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/data/knowledge-v3.1/decipher-onnx-check.json"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    root = Path("runs/knowledge-v3.1")
    model, _, tables, _ = load(args.checkpoint)
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    width = 8 if model.cfg.tensor_abi == 5 else 4  # plan brainiac-v6: ABI 5 carries 8 floats per token
    items = [parse(record, tables, width=width) for record in records]
    path = args.out / "model.onnx"
    export(model, stack(items[:2]), str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    live_inputs = {value.name for value in session.get_inputs()}
    errors = []
    near_ties = []
    with torch.no_grad():
        model.cache_card_table()
        for index, item in enumerate(items):
            batch = stack([item])
            expected = model(batch).numpy()[0]
            actual = session.run(["logits"], {key: batch[key].numpy() for key in INPUTS if key in live_inputs})[0][0]
            np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-4)
            match = greedy_match(expected, actual)
            if not match["matched"]:
                raise AssertionError(f"ONNX greedy mismatch: {index}")
            if match["used_near_tie"]:
                near_ties.append({"record": index, "options": len(expected), **match})
            errors.append(float(np.abs(expected - actual).max()))
    report = {"passed": True, "records": len(items), "max_absolute": max(errors), "table_sha": tables.sha,
              "near_tie_count": len(near_ties), "near_ties": near_ties,
              "strict_greedy_match": len(items) - len(near_ties),
              "inputs": sorted(live_inputs), "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path("src/swccg_policy/v4_onnx.py").read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
