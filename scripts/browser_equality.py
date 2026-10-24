"""Browser equality check for Brainiac: the Gemp client's worker (brainiacWorker.js, onnxruntime-web) in headless Chromium must
rank the options like onnxruntime on CPU, on recorded decisions.

    uv run --with playwright python scripts/browser_equality.py --release releases-staging/releases/rl-final-2026-09-25 \
        --golden /tmp/golden-full --worker ~/src/gemp-swccg-private/src/gemp-swccg-async/src/main/web/js/brainiac/brainiacWorker.js \
        --out reports/browser-bot-equality.json

The inputs are the golden records' tensors (scripts/brainiac_golden.py; equal to the Gemp server's BrainiacTensors by its unit
test) in the server's binary layout. Each backend (wasm; webgpu if the headless browser has an adapter) is tested separately.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import math
import shutil
import socketserver
import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

PAGE = """<!doctype html><html><body><script>
async function go(backends, payloads) {
  const w = new Worker("brainiacWorker.js");
  const pending = {};
  const ready = new Promise((res, rej) => {
    w.onmessage = ev => {
      const m = ev.data;
      if (m.type === "ready") res(m.backend);
      else if (m.type === "result") pending[m.id].res(m);
      else if (m.type === "error") (m.id && pending[m.id] ? pending[m.id].rej : rej)(new Error(m.message));
    };
  });
  const t0 = performance.now();
  w.postMessage({type: "init", modelUrl: location.origin + "/model.onnx", sha256: SHA, runtimeBase: location.origin + "/runtime/", backends});
  const backend = await ready;
  const initMs = performance.now() - t0;
  const out = [];
  for (let i = 0; i < payloads.length; i++) {
    const r = await new Promise((res, rej) => { pending[i + 1] = {res, rej}; w.postMessage({type: "run", id: i + 1, t: payloads[i], withLogits: true}); });
    out.push({ranking: r.ranking, logits: r.logits, ms: r.ms});
  }
  w.terminate();
  return {backend, initMs, out};
}
</script></body></html>"""


def to_bytes(e: dict) -> bytes:
    """BrainiacTensors.toBytes(): int32 T, O, head[22], int16 tok[T*8], float32 tokf[T*4], int32 opt[O*6], little endian."""
    tok, tokf, opt, head = e["tok"], e["tokf"], e["opt"], e["head"]
    b = struct.pack("<ii", len(tok), len(opt)) + struct.pack("<22i", *head)
    b += struct.pack(f"<{8 * len(tok)}h", *[v for row in tok for v in row])
    b += struct.pack(f"<{4 * len(tok)}f", *[math.nan if v is None else v for row in tokf for v in row])
    b += struct.pack(f"<{6 * len(opt)}i", *[v for row in opt for v in row])
    return b


def reference(model: Path, expected: list[dict]) -> list[np.ndarray]:
    import onnxruntime as ort
    s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out = []
    for e in expected:
        T = max(1, len(e["tok"]))
        tok = np.zeros((1, T, 8), np.int32)
        tokf = np.full((1, T, 4), np.nan, np.float32)
        tmask = np.zeros((1, T), bool)
        if e["tok"]:
            tok[0, :len(e["tok"])] = e["tok"]
            tokf[0, :len(e["tok"])] = np.array([[np.nan if v is None else v for v in r] for r in e["tokf"]], np.float32)
            tmask[0, :len(e["tok"])] = True
        O = len(e["opt"])
        h = np.array(e["head"], np.int32)
        feed = {"tok": tok, "tokf": tokf, "tmask": tmask, "opt": np.array([e["opt"]], np.int32), "omask": np.ones((1, O), bool),
                "dtype": h[0:1], "qtmpl": h[1:2], "qbp": h[2:3], "side": h[3:4], "glob": h[4:22][None]}
        out.append(s.run(["logits"], feed)[0][0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backends", default="wasm,webgpu")
    ap.add_argument("--required-backends", default="wasm")
    a = ap.parse_args()
    rel = Path(a.release)
    manifest = json.loads((rel / "manifest.json").read_text())
    runtime = rel.parent.parent / "runtime" / manifest["runtime"]
    expected = [json.loads(l) for l in open(Path(a.golden) / "expected.jsonl")]
    payloads = [base64.b64encode(to_bytes(e)).decode() for e in expected]
    ref = reference(rel / manifest["model"], expected)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "index.html").write_text(PAGE.replace("SHA", json.dumps(manifest["sha256"])))
        shutil.copy(a.worker, d / "brainiacWorker.js")
        (d / "model.onnx").symlink_to((rel / manifest["model"]).resolve())
        (d / "runtime").symlink_to(runtime.resolve())
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
        handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                                  ".wasm": "application/wasm", ".js": "text/javascript"}
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        from playwright.sync_api import sync_playwright
        results = {"release": manifest["name"], "model_sha256":manifest['sha256'], "decisions": len(expected), "backends": {}}
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan"])
            for backend in a.backends.split(","):
                page = browser.new_page()
                page.goto(f"http://localhost:{port}/index.html")
                try:
                    r = page.evaluate("([b, p]) => go(b, p)", [[backend], payloads])
                except Exception as ex:  # e.g. no WebGPU adapter in headless Chromium
                    results["backends"][backend] = {"available": False, "error": str(ex).splitlines()[0][:300]}
                    page.close()
                    continue
                agree, max_rel, ms, near_ties = 0, 0.0, [], []
                for index,(got, want) in enumerate(zip(r["out"], ref, strict=True)):
                    g = np.array(got["logits"], np.float64)
                    if g.shape != want.shape or not np.isfinite(g).all():
                        raise AssertionError("invalid browser logits")
                    agree += int(int(np.argmax(g)) == int(np.argmax(want)))
                    if np.argmax(g)!=np.argmax(want):
                        gap=float(want.max()-want[np.argmax(g)])
                        if gap<=2e-7:near_ties.append({'index':index,'reference_gap':gap})
                    max_rel = max(max_rel, float((np.abs(g - want) / np.maximum(np.abs(want), 1.0)).max()))
                    ms.append(got["ms"])
                results["backends"][backend] = {"available": True, "backend_used": r["backend"], "argmax_agreement": agree / len(ref),
                                                "argmax_equal": agree, "max_rel_diff": max_rel, "init_ms": round(r["initMs"]),
                                                "near_ties": near_ties, "passed": agree+len(near_ties)==len(ref) and max_rel<=2e-5,
                                                "decision_ms_median": round(float(np.median(ms)), 1),
                                                "decision_ms_p95": round(float(np.percentile(ms, 95)), 1)}
                page.close()
            browser.close()
        httpd.shutdown()
    results["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['passed']=all(results['backends'].get(k,{}).get('passed',False) for k in a.required_backends.split(',')) and all(
        not value['available'] or value['passed'] for value in results['backends'].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    if not results['passed']:raise SystemExit(1)


if __name__ == "__main__":
    main()
