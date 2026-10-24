"""Plan brainiac-v6: build a Brainiac release from a v6 checkpoint, pulled from the training server if needed.

    .venv/bin/python scripts/brainiac_release.py --checkpoint models/shc-r1/policy.pt --name shc-r1-2026-10-24
    .venv/bin/python scripts/brainiac_release.py --checkpoint runs/dojo/my-run/final.pt --name my-run-YYYY-MM-DD

- "<host>:<path>" (host = $SWCCG_PULL_HOST, an ssh alias) is a path under ~/src/swccg-policy on a training machine. It
  is copied (read only there, rsync over ssh) with its configuration.json to runs/brainiac/pull/<run>/.
- Only v6 checkpoints: tensor ABI 5, context version 5 (BrainiacRelease refuses anything else).
- Writes models/brainiac/releases/<name>/ (--out): model.onnx, the table bundle (vocab.json, cards.json, bundle.json, ...)
  the frozen title table (titles.tsv) and manifest.json (encoder 5, tensor ABI 5, table SHA, model SHA-256, checkpoint source and SHA-256, the export
  check). An existing release directory is never overwritten.
- Export check: ONNX (CPU) against PyTorch on --records (atol 1e-5, greedy match with the near-tie rule); the full
  browser check is scripts/check_v31_browser.py.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import os
import subprocess

import numpy as np
import onnxruntime as ort
import torch

from swccg_policy.native import write_bundle
from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_onnx import INPUTS, export
from swccg_policy.v4_parity import greedy_match

ROOT = Path(__file__).resolve().parents[1]
SERVER = os.environ.get("SWCCG_PULL_HOST", "swccg-server")
RUNTIME = "onnxruntime-web-1.30.0"
TITLES = ROOT / "runs/knowledge-v3.1/decipher/titles.tsv"  # the frozen title table training encodes with
ENCODER = 5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pull(source):
    """A local checkpoint path; "swccg-server:<path>" is copied from the server first."""
    if not source.startswith(SERVER + ":"):
        return Path(source), None
    remote = Path(source.split(":", 1)[1])
    local = ROOT / "runs/brainiac/pull" / remote.parent.name
    local.mkdir(parents=True, exist_ok=True)
    files = [f"{SERVER}:src/swccg-policy/{remote}"]
    configuration = remote.parent / "configuration.json"
    if subprocess.run(["ssh", SERVER, f"test -f src/swccg-policy/{configuration}"]).returncode == 0:
        files.append(f"{SERVER}:src/swccg-policy/{configuration}")
    subprocess.run(["rsync", "-a", *files, f"{local}/"], check=True)
    return local / remote.name, source


def check(model, items, path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    live = {value.name for value in session.get_inputs()}
    errors, near_ties = [], 0
    with torch.no_grad():
        for index, item in enumerate(items):
            batch = stack([item])
            expected = model(batch).numpy()[0]
            actual = session.run(["logits"], {key: batch[key].numpy() for key in INPUTS if key in live})[0][0]
            np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-4)
            match = greedy_match(expected, actual)
            if not match["matched"]:
                raise AssertionError(f"ONNX greedy mismatch on record {index}")
            near_ties += match["used_near_tie"]
            errors.append(float(np.abs(expected - actual).max()))
    return {"decisions": len(items), "max_abs_diff": max(errors), "near_ties": near_ties,
            "argmax_agreement": 1.0, "inputs": sorted(live)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local path or <SWCCG_PULL_HOST>:<path under the repo>")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "models/brainiac")
    parser.add_argument("--records", type=Path, default=ROOT / "data/brainiac/export-records.jsonl",
                        help="schema-4 v6 records for the export sample and the ONNX check")
    parser.add_argument("--notes", default="")
    # Plan v-cards: a grown checkpoint's title table (default: the frozen Decipher one training encodes with).
    parser.add_argument("--titles", type=Path, default=TITLES)
    args = parser.parse_args()
    release = args.out / "releases" / args.name
    if release.exists():
        raise SystemExit(f"{release} exists; releases are never overwritten")
    if not (args.out / "runtime" / RUNTIME).is_dir():
        raise SystemExit(f"browser runtime {args.out / 'runtime' / RUNTIME} missing")
    checkpoint, source = pull(args.checkpoint)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    model, _, tables, payload = load(checkpoint)
    rl_meta = payload.get("rl_meta") or {}
    model.eval()
    if model.cfg.tensor_abi != 5 or tables.vocabulary.payload.get("context_version") != 5:
        raise SystemExit("Brainiac serves v6 policies only: tensor ABI 5 with context version 5")
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    items = [parse(record[:12], tables, width=8) for record in records]
    staging = release.with_name(release.name + ".partial")
    if staging.exists():
        raise SystemExit(f"{staging} exists (a failed build); remove it by hand")
    write_bundle(checkpoint, staging, "owned")
    shutil.copy2(args.titles, staging / "titles.tsv")
    export(model, stack(items[:2]), str(staging / "model.onnx"))
    model.cache_card_table()
    export_check = check(model, items, staging / "model.onnx")
    configuration = checkpoint.parent / "configuration.json"
    run = json.loads(configuration.read_text()) if configuration.exists() else {}
    manifest = {
        "name": args.name, "model": "model.onnx", "sha256": sha256(staging / "model.onnx"),
        "bytes": (staging / "model.onnx").stat().st_size, "runtime": RUNTIME, "encoder": ENCODER, "tensor_abi": 5,
        "table_sha": tables.sha, "titles": "titles.tsv", "titles_sha256": sha256(staging / "titles.tsv"),
        "inputs": INPUTS, "output": "logits float32[B,O]",
        "source": {"checkpoint": source or str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                   "run": run.get("args", {}).get("tag"), "rl_meta": rl_meta},
        "export_check": export_check,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": args.notes,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    staging.rename(release)
    print(json.dumps({"release": str(release), "sha256": manifest["sha256"], "bytes": manifest["bytes"],
                      "table_sha": tables.sha, "export_check": export_check}))


if __name__ == "__main__":
    main()
