"""Plan brainiac-gemparena §1.2: GempArena's Brainiac bridge must rank Brainiac decisions exactly like the Gemp web client.

    # 1. record: a game vs Brainiac on the rehearsal; this script plays the human seat over HTTP (minimal answers) and
    #    answers Brainiac with the web client's own worker in headless Chromium, saving each "BD" payload and its ranking
    .venv/bin/python deploy/beta/arena_brainiac_check.py record --n 20 --out reports/data/beta-server/arena-brainiac-payloads.jsonl.gz
    # 2. compare: GempArena's page (/GempArena/) ranks the same payloads through its bridge (BrainiacBridge.CheckRank)
    .venv/bin/python deploy/beta/arena_brainiac_check.py compare --payloads reports/data/beta-server/arena-brainiac-payloads.jsonl.gz

Both run against the local rehearsal (deploy/beta/rehearse.sh, http://localhost:18080) by default; record makes one account
(ac<7 digits>) there. Headless Chromium uses SwiftShader for GempArena's WebGL; the model runs on WebAssembly.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import http.cookiejar
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from playwright.sync_api import sync_playwright

CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

WORKER_PAGE_JS = """async ([payloads, sha256, runtime]) => {
  if (!window.__bw) {
    const w = new Worker("/gemp-swccg/js/brainiac/brainiacWorker.js");
    const pending = {};
    window.__bw = {w, pending, next: 1, ready: new Promise((res, rej) => {
      w.onmessage = ev => {
        const m = ev.data;
        if (m.type === "ready") res(m.backend);
        else if (m.type === "result") pending[m.id].res(m);
        else if (m.type === "error") (m.id && pending[m.id] ? pending[m.id].rej : rej)(new Error(m.message));
      };
    })};
    w.postMessage({type: "init", modelUrl: "/gemp-swccg-server/brainiac/model/" + sha256 + ".onnx", sha256,
                   runtimeBase: location.origin + "/gemp-swccg-server/brainiac/runtime/" + runtime + "/"});
  }
  const backend = await window.__bw.ready;
  const out = [];
  for (const t of payloads) {
    const id = window.__bw.next++;
    const r = await new Promise((res, rej) => { window.__bw.pending[id] = {res, rej}; window.__bw.w.postMessage({type: "run", id, t}); });
    out.push({ranking: r.ranking, ms: r.ms});
  }
  return {backend, out};
}"""


def params(decision: ET.Element, name: str) -> list[str]:
    return [p.get("value") for p in decision.iter("parameter") if p.get("name") == name]


def minimal_answers(decision: ET.Element) -> list[str]:
    """Answers to try, in order, for a decision of the human seat (the cheapest valid move; the game only needs to go on)."""
    kind = decision.get("decisionType")
    one = lambda name: (params(decision, name) or [None])[0]
    if kind == "CARD_ACTION_CHOICE":
        ids = params(decision, "actionId")
        return ([""] if one("noPass") != "true" else []) + ids[:3]
    if kind == "ACTION_CHOICE":
        return [str(i) for i in range(min(3, len(params(decision, "actionId")) or 1))]
    if kind == "MULTIPLE_CHOICE":
        return [str(i) for i in range(min(3, len(params(decision, "results")) or 1))]
    if kind == "INTEGER":
        lo = one("defaultValue") or one("min") or "0"
        return [lo, one("min") or "0"]
    if kind in ("CARD_SELECTION", "ARBITRARY_CARDS"):
        ids = params(decision, "cardId")
        if kind == "ARBITRARY_CARDS":
            ok = params(decision, "selectable")
            ids = [c for c, s in zip(ids, ok + ["true"] * len(ids)) if s == "true"]
        lo = int(one("min") or 0)
        return [",".join(ids[:max(lo, 0)]), ",".join(ids[:max(lo, 1)])]
    return [""]


def fetch(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


class HumanSeat:
    def __init__(self, base: str, user: str, password: str):
        self.base = base.rstrip("/") + "/gemp-swccg-server/"
        self.referer = base.rstrip("/") + "/gemp-swccg/"
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.user = user
        self.post("register", {"login": user, "password": password})
        status, _ = self.post("login", {"login": user, "password": password})
        if status != 200:
            sys.exit(f"login {user}: HTTP {status}")

    def request(self, path: str, data: dict | None = None, query: dict | None = None, timeout: float = 120) -> tuple[int, str]:
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, headers={"Referer": self.referer})
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    def post(self, path: str, data: dict, timeout: float = 120) -> tuple[int, str]:
        return self.request(path, data=data, timeout=timeout)

    def create_brainiac_table(self, deck: str, bot_deck: str) -> str:
        _, text = self.post("hall", {
            "participantId": "", "deckName": deck, "format": "open", "sampleDeck": "true", "tableDesc": "arena check",
            "isPrivate": "false", "playVsAi": "true", "aiSkill": "BRAINIAC", "aiDeckName": bot_deck, "aiDeckSample": "true"},
            timeout=60)
        if "error" in text:
            sys.exit("create table: " + text)
        for _ in range(30):
            _, hall = self.request("hall", query={"participantId": "null"})
            for t in re.findall(r"<table [^>]*>", hall):
                if self.user + " (" in t and 'gameId="' in t:
                    return re.search(r'gameId="([^"]+)"', t).group(1)
            time.sleep(1)
        sys.exit("no game started")

    def join(self, game: str) -> ET.Element:
        self.game = game
        root = ET.fromstring(self.request("game/" + game, query={"participantId": self.user})[1])
        self.cn = root.get("cn")
        return root

    def update(self, decision_id: str | None = None, value: str | None = None) -> ET.Element:
        data = {"participantId": self.user, "channelNumber": self.cn}
        if decision_id is not None:
            data.update(decisionId=decision_id, decisionValue=value)
        status, text = self.post("game/" + self.game, data)
        if status != 200:
            sys.exit(f"game update: HTTP {status} {text[:200]}")
        root = ET.fromstring(text)
        self.cn = root.get("cn", self.cn)
        return root

    def answer_brainiac(self, bot: str, decision_id: str, ranking: list[int]) -> int:
        return self.post("game/" + self.game + "/brainiacDecision", {
            "participantId": self.user, "channelNumber": self.cn, "botId": bot, "decisionId": decision_id,
            "ranking": ",".join(map(str, ranking))}, timeout=60)[0]


def record(a) -> None:
    hall = fetch(a.base.rstrip("/") + "/gemp-swccg-server/deck/libraryList")
    dark = re.findall(r"<darkDeck>([^<]*)", hall)
    light = re.findall(r"<lightDeck>([^<]*)", hall)
    deck = a.deck or next(d for d in dark if d.startswith("Classic"))
    bot_deck = a.bot_deck or next(d for d in light if d.startswith("Classic"))
    user = "ac" + str(random.randint(10 ** 6, 10 ** 7 - 1))  # Gemp names: at most 10 characters
    seat = HumanSeat(a.base, user, "p" + str(random.getrandbits(64)))
    game = seat.create_brainiac_table(deck, bot_deck)
    print(f"account {user}; game {game}: {deck} (human seat, this script) vs {bot_deck} (Brainiac)", flush=True)
    records, seen, sizes = [], set(), []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(a.base.rstrip("/") + "/gemp-swccg/")
        deadline = time.time() + a.minutes * 60

        def take(root: ET.Element) -> ET.Element | None:
            """Answer every new Brainiac decision in an update (recording it); return the human seat's last decision."""
            decision = None
            for ge in root.iter("ge"):
                if ge.get("type") == "D":
                    decision = ge
                elif ge.get("type") == "BD" and len(records) < a.n:
                    # decision ids are reused (per player), so a re-sent request is recognised by id and payload
                    payload = json.loads(ge.get("message"))
                    if (ge.get("index"), payload["t"]) in seen:
                        continue
                    seen.add((ge.get("index"), payload["t"]))
                    res = page.evaluate(WORKER_PAGE_JS, [[payload["t"]], payload["sha256"], payload["runtime"]])
                    ranking = res["out"][0]["ranking"]
                    records.append({"name": f"bd{len(records) + 1:02d}", "decisionId": ge.get("index"),
                                    "question": payload.get("question", ""), "sha256": payload["sha256"],
                                    "runtime": payload["runtime"], "t": payload["t"], "ranking": ranking,
                                    "backend": res["backend"], "ms": res["out"][0]["ms"]})
                    sizes.append(len(payload["t"]))
                    code = seat.answer_brainiac(ge.get("participantId"), ge.get("index"), ranking)
                    print(f"  {records[-1]['name']} decision {ge.get('index')}: {len(ranking)} options, "
                          f"{len(payload['t']) // 1024} KB, {res['out'][0]['ms']:.0f} ms, answer HTTP {code}", flush=True)
            return decision

        decision = take(seat.join(game))
        while len(records) < a.n and time.time() < deadline:
            if decision is None:
                decision = take(seat.update())
                continue
            # Gemp numbers decisions per player and reuses ids: a rejected answer re-sends the same id and text
            key = (decision.get("id"), decision.get("text"))
            nxt = None
            for value in minimal_answers(decision):
                nxt = take(seat.update(decision.get("id"), value))
                if nxt is None or (nxt.get("id"), nxt.get("text")) != key:
                    break
            decision = nxt
        browser.close()
    with gzip.open(a.out, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"recorded {len(records)} Brainiac decisions (payload base64 {min(sizes) if sizes else 0}–{max(sizes) if sizes else 0} "
          f"chars) -> {a.out}; account {user}, game {game} (left running)")


def keep_unity_instance(route) -> None:
    """Test only: GempArena's page template keeps its Unity instance local; expose it as window.__unity."""
    response = route.fetch()
    html = response.text()
    hooked = html.replace(".then((unityInstance) => {", ".then((unityInstance) => { window.__unity = unityInstance;", 1)
    if hooked == html:
        sys.exit("GempArena's index.html no longer has the expected createUnityInstance(...).then((unityInstance) => {")
    route.fulfill(response=response, body=hooked)


def compare(a) -> None:
    records = [json.loads(l) for l in gzip.open(a.payloads, "rt")]
    lines: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.route("**/GempArena/", keep_unity_instance)
        page.on("console", lambda m: lines.append(m.text) if m.text.startswith("BrainiacCheck") else None)
        page.goto(a.base.rstrip("/") + "/GempArena/")
        page.wait_for_function("window.__unity !== undefined", timeout=300_000)
        time.sleep(5)  # the scene's first frame; BrainiacBridge is created after the first scene loads
        for r in records:
            req = {k: r[k] for k in ("name", "sha256", "runtime", "t")}
            page.evaluate("j => window.__unity.SendMessage('BrainiacBridge', 'CheckRank', j)", json.dumps(req))
        deadline = time.time() + a.minutes * 60
        while len(lines) < len(records) and time.time() < deadline:
            page.wait_for_timeout(500)
        browser.close()
    got = {}
    for line in lines:
        parts = line.split()
        got[parts[1]] = parts[2:]
    same = 0
    print(f"GempArena bridge vs the Gemp web client's worker, {len(records)} recorded Brainiac decisions")
    for r in records:
        g = got.get(r["name"])
        if g is None or g[0] == "ERROR":
            print(f"  {r['name']}: FAIL {' '.join(g) if g else 'no answer'}")
            continue
        ranking = [int(x) for x in g[0].split(",")]
        ok = ranking == r["ranking"]
        same += ok
        print(f"  {r['name']}: {'same' if ok else 'DIFFERENT'} ({len(ranking)} options, payload {len(r['t']) // 1024} KB, "
              f"GempArena {g[1]} {g[2]}, web client ms={r['ms']:.0f} backend={r['backend']})")
    print(f"{same}/{len(records)} rankings identical")
    sys.exit(0 if same == len(records) and records else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "compare"])
    ap.add_argument("--base", default="http://localhost:18080")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--deck", help="human seat's library deck (default: the first dark 'Classic' deck)")
    ap.add_argument("--bot-deck", help="Brainiac's library deck (default: the first light 'Classic' deck)")
    ap.add_argument("--out", default="reports/data/beta-server/arena-brainiac-payloads.jsonl.gz")
    ap.add_argument("--payloads", default="reports/data/beta-server/arena-brainiac-payloads.jsonl.gz")
    a = ap.parse_args()
    record(a) if a.mode == "record" else compare(a)


if __name__ == "__main__":
    main()
