"""Check real schema-4 records through Java packing, SWTRJ003, NumPy and JNI."""

import copy
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from swccg_policy.native import export_ts, write_bundle
from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_trajectory import load_binary
from swccg_policy.v4_parity import greedy_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/knowledge-v3.1/decipher-abi-check"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/knowledge-v3.1/decipher-growth-candidate.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/data/knowledge-v3.1/decipher-abi-check.json"))
    parser.add_argument("--fixtures", type=Path, default=Path("runs/knowledge-v3.1/grid-decipher/fixtures"),
                        help="exported schema-4 records (plan context-overhaul: records of context version 5)")
    parser.add_argument("--traced", type=Path, help="Validate an existing cached weight-update export")
    parser.add_argument("--native-lib", type=Path, default=Path("native/build-v3.1/libswccgpolicy.so"),
                        help="the JNI library under test (default: the installed build)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path("runs/knowledge-v3.1")
    run = args.run
    run.mkdir(exist_ok=False)
    checkpoint = args.checkpoint
    model, _, tables, _ = load(checkpoint, "cuda")
    if args.traced:
        shutil.copyfile(args.traced, run / "actor.ts")
    else:
        export_ts(model, run / "actor.ts")
    write_bundle(checkpoint, run / "bundle", "unused-owned-cards")
    available = []
    for path in sorted(args.fixtures.glob("*.gz")):
        with gzip.open(path, "rt") as stream:
            available.extend(json.loads(line) for line in stream)
    records = [available[index] for index in np.linspace(0, len(available) - 1, 48, dtype=int)]
    for marker in ("inPlay", "rules"):
        record = copy.deepcopy(next(row for row in available if row[9]))
        record[9][0][3] = marker
        records.append(record)
    # Distinct integer choices test complete tails without a mathematically tied argmax.
    for count in (65, 513, 4097):
        record = copy.deepcopy(next(record for record in available if all(token[0] != 10 for token in record[8])))
        record[4] = "INTEGER"
        record[9] = [["<n>", [], -1, "", str(index), index] for index in range(count)]
        records.append(record)
    if model.cfg.tensor_abi == 4:
        # Today's exports carry the ABI-5 location state on location rows; an ABI-4 model reads four floats.
        for record in records:
            for index in (8, 12):
                if len(record) > index:
                    record[index] = [token[:12] for token in record[index]]
    record_path = run / "records.jsonl"
    record_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    width = 8 if model.cfg.tensor_abi == 5 else 4  # plan objective-dojo Amendment 2
    actors = [parse(record, tables, width=width) for record in records]
    critics = [parse(record, tables, critic=True, width=width) for record in records]
    with torch.no_grad():
        model.cache_card_table()
        expected = [model(stack([actor], "cuda")).cpu().numpy()[0] for actor in actors]
    command = ["bash", "scripts/efficient_java_check.sh",
               f"-Dpolicy.native.lib={args.native_lib.resolve()}",
               "com.gempukku.swccgo.bench.AbiV4Check", str(record_path), str(run / "bundle"),
               str(run / "java"), str(run / "actor.ts")]
    environment = {**os.environ, "SWCCG_TENSOR_ABI": str(model.cfg.tensor_abi), "SWCCG_GRAPH_T": "512",
                   "SWCCG_V31_CLASSES": str(run / "classes")}
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    (run / "java.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-6000:])
    binary = {}
    for level in (0, 1):
        files = list((run / f"java/gzip{level}/traj").glob("*.traj"))
        games = [game for path in files for game in load_binary(path, tables)]
        assert len(games) == 1
        game = games[0]
        for originals, decoded in ((actors, game["items"]), (critics, game["critic_items"])):
            for original, actual in zip(originals, decoded, strict=True):
                for key in ("tok", "tokf", "opt", "head", "kgrid"):
                    np.testing.assert_array_equal(original[key], actual[key])
        assert game["knowledge_sequences"] == [record[13] for record in records]
        np.testing.assert_array_equal(game["old_logp"], np.full(len(records), -0.25, np.float32))
        binary[str(level)] = {"exact": True, "records": len(records)}
    native = {}
    for path in sorted((run / "java").glob("logits-*.jsonl")):
        actual = [np.asarray(json.loads(line), np.float32) for line in path.read_text().splitlines()]
        errors = []
        near_ties = []
        for index, (reference, candidate) in enumerate(zip(expected, actual, strict=True)):
            np.testing.assert_allclose(reference, candidate, atol=2e-5, rtol=2e-4)
            match = greedy_match(reference, candidate)
            if not match["matched"]:
                raise AssertionError(f"Greedy mismatch in {path.name}, record {index}")
            if match["used_near_tie"]:
                near_ties.append({"record": index, "options": len(reference), **match})
            errors.append(float(np.abs(reference - candidate).max()))
        native[path.stem] = {"greedy_match": len(actual), "strict_greedy_match": len(actual) - len(near_ties),
                             "near_tie_count": len(near_ties), "near_ties": near_ties,
                             "max_absolute": max(errors)}
    report = {"passed": True, "binary": binary, "native": native, "table_sha": tables.sha,
              "tails": [65, 513, 4097], "command": command, "existing_traced_module": str(args.traced),
              "scope": "Java binary/NumPy and JNI parity; browser and sampled log-prob reconstruction pending.",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                [Path(__file__), Path("native/src/policy_jni.cpp"),
                                 *Path("src/swccg_policy").glob("v4_*.py"),
                                 *Path("engine/java/com/gempukku/swccgo/bench").glob("*.java")]}}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"binary": binary, "native": native}))


if __name__ == "__main__":
    main()
