"""Plan client-logs §1: browser console lines of both clients reach the beta's collector.

    .venv/bin/python deploy/beta/clientlog_check.py                     # rehearsal (http://localhost:18080), all checks
    .venv/bin/python deploy/beta/clientlog_check.py --base https://brainiac-beta.duckdns.org   # live: §1.1-1.2 only

1.1 a classic page and the GempArena page each send a marker console.error line and an uncaught exception;
1.2 with ?debug=0 nothing is sent; 1.3 (rehearsal only, it writes 20,000 lines) a request over 64 KB is refused (413),
a session is cut at 20,000 lines, and more than 600 requests a minute from one address get 429. The 14-day clean-up
is not exercised (no deletion tests). Lines are read from the rehearsal's folder, or live with deploy.sh clientlogs.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
results: list[tuple[bool, str]] = []


def check(ok: bool, text: str) -> None:
    results.append((ok, text))
    print(("ok   " if ok else "FAIL ") + text, flush=True)


def lines_with(base: str, needle: str) -> list[dict]:
    if "localhost" in base:
        out = []
        for f in sorted((ROOT / ".cache/beta-rehearsal/clientlogs").glob("*.jsonl")):
            out += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if needle in l]
        return out
    text = subprocess.run([str(ROOT / "deploy/beta/deploy.sh"), "clientlogs", "1000", needle], capture_output=True,
                          text=True, check=True).stdout
    return [json.loads(l) for l in text.splitlines() if l.startswith("{")]


def visit(browser, url: str, nonce: str, tag: str, wait: float) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto(url, wait_until="load")
    page.wait_for_timeout(wait * 1000)
    page.evaluate("""([n, t]) => { console.error("clientlog-check " + n + " " + t);
                                   setTimeout(() => { throw new Error("clientlog-uncaught " + n + " " + t); }, 0); }""",
                  [nonce, tag])
    page.wait_for_timeout(4000)
    page.close()


def post(base: str, body: bytes) -> int:
    req = urllib.request.Request(base + "/clientlog", data=body, headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:18080")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    nonce = "%08x" % random.getrandbits(32)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        visit(browser, base + "/gemp-swccg/index.html", nonce, "classic", 2)
        visit(browser, base + "/GempArena/", nonce, "gemparena", 40)
        visit(browser, base + "/gemp-swccg/index.html?debug=0", nonce, "off", 2)
        browser.close()
    time.sleep(3)
    got = lines_with(base, nonce)
    for tag, client in (("classic", "classic"), ("gemparena", "gemparena")):
        marker = [l for l in got if l["text"].startswith(f"clientlog-check {nonce} {tag}") and l["client"] == client]
        uncaught = [l for l in got if f"clientlog-uncaught {nonce} {tag}" in l["text"] and l["level"] == "uncaught"]
        check(bool(marker) and bool(uncaught), f"{tag}: marker ({len(marker)}) and uncaught exception ({len(uncaught)}) "
              f"logged, level {marker[0]['level'] if marker else '-'}, page {marker[0]['page'] if marker else '-'}")
    session = {l["session"] for l in got if l["client"] == "gemparena"}
    if session:
        unity = lines_with(base, next(iter(session)))
        check(any("Successfully fetched and parsed" in l["text"] for l in unity),
              f"gemparena: Unity's own console lines arrive too ({len(unity)} lines in that session)")
    off = [l for l in got if f"{nonce} off" in l["text"]]
    check(not off, f"?debug=0: nothing sent ({len(off)} lines)")
    if "localhost" not in base:
        print("limits: rehearsal only")
    else:
        check(post(base, b"x" * (65 * 1024)) == 413, "a request over 64 KB is refused (413)")
        s = "limit-" + nonce
        for i in range(25):
            post(base, json.dumps({"client": "test", "page": "/limit", "session": s, "user": None,
                                   "lines": [[int(time.time() * 1000), "info", f"l{i}-{j}"] for j in range(1000)]}).encode())
        n = len(lines_with(base, s))
        check(n == 20000, f"a session is cut at 20,000 lines ({n} of 25,000 kept)")
        codes = [post(base, json.dumps({"client": "test", "session": "rate-" + nonce, "lines": []}).encode())
                 for _ in range(610)]
        check(429 in codes, f"over 600 requests a minute from one address: 429 ({codes.count(429)} of 610)")
    bad = [t for ok, t in results if not ok]
    print("ALL CHECKS PASSED" if not bad else f"{len(bad)} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
