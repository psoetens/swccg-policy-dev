"""Plan brainiac-v6: a release's production Java wire format (SWT5, tensor ABI 5) and actual browser WASM/WebGPU parity.

    .venv/bin/python scripts/check_v31_browser.py --release releases-staging/releases/<name> \\
        --checkpoint <the checkpoint it was built from> --records runs/brainiac-v6/records.jsonl --run <new dir> --out <json>
"""

import argparse
import base64
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import socketserver
import struct
import subprocess
import threading

import numpy as np
import torch

from browser_equality import PAGE
from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_parity import greedy_match


def payload(item):
    header = struct.pack("<4i", 0x35545753, len(item["tok"]), len(item["opt"]), len(item["kgrid"]))
    return header + b"".join(np.asarray(item[key], dtype=dtype).tobytes()
                            for key, dtype in (("head", "<i4"), ("tok", "<i4"), ("tokf", "<f4"),
                                               ("opt", "<i4"), ("kgrid", "u1")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True, help="a release from scripts/brainiac_release.py")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    model, _, tables, _ = load(args.checkpoint)
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    items = [parse(record[:12], tables, width=8) for record in records]
    release = args.release.resolve()
    manifest = json.loads((release / "manifest.json").read_text())
    onnx = release / manifest["model"]
    digest = hashlib.sha256(onnx.read_bytes()).hexdigest()
    if digest != manifest["sha256"] or manifest["table_sha"] != tables.sha:
        raise AssertionError("Release model/table does not match the checkpoint")
    java_path = args.run / "java-payloads.json"
    command = ["bash", "scripts/efficient_java_check.sh", "-Dpolicy.knowledge=true", "-Dpolicy.context=5",
               "com.gempukku.swccgo.bench.BrowserV4Check", str(args.records), str(release), str(java_path)]
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, "SWCCG_V31_CLASSES": str(args.run / "classes")})
    (args.run / "java.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Production browser packing failed; see java.log")
    java = json.loads(java_path.read_text())
    for item, row in zip(items, java, strict=True):
        if payload(item) != base64.b64decode(row["payload"]):
            raise AssertionError("Java browser bytes differ from NumPy")
    with torch.no_grad():
        model.cache_card_table()
        reference = [model(stack([item])).numpy()[0] for item in items]
    site = args.run / "site"
    site.mkdir()
    (site / "index.html").write_text(PAGE.replace("SHA", json.dumps(digest)))
    shutil.copyfile("engine/production/web/brainiacWorker.js", site / "brainiacWorker.js")
    (site / "model.onnx").symlink_to(onnx)
    (site / "runtime").symlink_to(Path("releases-staging/runtime/onnxruntime-web-1.30.0").resolve())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {"release": manifest["name"], "records": len(items), "java_bytes_exact": True, "privileged_envelopes_rejected": True,
              "onnx_sha256": digest, "table_sha": tables.sha, "backends": {},
              "reference": "Python CPU FP32; approved top-two-only near-tie rule"}
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan",
                                                       "--use-angle=vulkan"])
            for backend in ("wasm", "webgpu"):
                page = browser.new_page()
                page.goto(f"http://localhost:{server.server_address[1]}/index.html")
                try:
                    output = page.evaluate("([b,p]) => Promise.race([go(b,p), new Promise((_,reject) => "
                                           "setTimeout(() => reject(new Error('Browser gate timeout')),360000))])",
                                           [[backend], [row["payload"] for row in java]])
                    checks = []
                    for index, (expected, got) in enumerate(zip(reference, output["out"], strict=True)):
                        actual = np.asarray(got["logits"], np.float32)
                        same_shape = actual.shape == expected.shape
                        finite = same_shape and np.isfinite(actual).all()
                        close = finite and np.allclose(expected, actual, atol=2e-5, rtol=2e-4)
                        match = greedy_match(expected, actual) if finite else {"matched": False}
                        # How far the browser's choice is below the best option in PyTorch (0 for a tie).
                        regret = float(expected.max() - expected[int(actual.argmax())]) if finite else None
                        checks.append({"record": index, "options": len(expected), "close": bool(close),
                                       "max_absolute": float(np.abs(expected - actual).max()) if finite else None,
                                       "choice_regret": regret, **match})
                    report["backends"][backend] = {"available": True, "backend_used": output["backend"],
                        "checks": checks, "passed": output["backend"] == backend
                        and all(check["close"] and check["matched"] for check in checks)}
                except Exception as error:
                    report["backends"][backend] = {"available": False, "passed": False, "error": str(error)}
                finally:
                    page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    report["passed"] = all(row["passed"] for row in report["backends"].values())
    paths = [Path(__file__), Path("engine/production/web/brainiacWorker.js"),
             *Path("engine/production/com/gempukku/swccgo/ai/brainiac").glob("*.java"),
             Path("engine/java/com/gempukku/swccgo/game/knowledge/ActorTensors.java")]
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "backends": {
        name: {key: value for key, value in result.items() if key != "checks"}
        for name, result in report["backends"].items()}}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
