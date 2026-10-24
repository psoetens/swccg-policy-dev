"""Re-screen decks the deck-corpus screen marked provisional because of the (since fixed) GempBench null-parameter NPE.

Same criteria as the deck-corpus screen: 10 AdvancedAi vs AdvancedAi games per deck against random passed decks of the
opposite side, pass = 0 crashes/exceptions and <= 2 decision-cap/timeout games. Result: data/schedules/rescreen.json.

    uv run python -m swccg_policy.rescreen
"""
from __future__ import annotations

import collections
import csv
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECKS = ROOT / "data/decks"


def main() -> None:
    m = list(csv.DictReader(open(DECKS / "manifest.csv")))
    prov = [r for r in m if r["status"] == "provisional_exception"]
    passed = {"DARK": [r["deck_id"] for r in m if r["status"] == "passed" and r["side"] == "DARK"],
              "LIGHT": [r["deck_id"] for r in m if r["status"] == "passed" and r["side"] == "LIGHT"]}
    rnd = random.Random(2026092402)
    matchups, owner = [], []
    for r in prov:
        me = str(DECKS / "rejected" / f"{r['deck_id']}.xml")
        for _ in range(10):
            if r["side"] == "DARK":
                matchups.append(f"{me}::{DECKS / 'corpus' / (rnd.choice(passed['LIGHT']) + '.xml')}")
            else:
                matchups.append(f"{DECKS / 'corpus' / (rnd.choice(passed['DARK']) + '.xml')}::{me}")
            owner.append(r["deck_id"])
    out = ROOT / "data/schedules"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "rescreen_games.csv"
    ms = ";".join(x.lstrip("/") for x in matchups)
    subprocess.run([str(Path.home() / "src/gemp-swccg-agent/ops/bench.sh"), "run", "--", "--decks", "/", "--matchups", ms,
                    "--pairing", "advanced", "--threads", "8", "--games", str(len(matchups)), "--cap", "6000",
                    "--maxGameMs", "300000", "--out", str(csv_path)], check=True, stderr=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(csv_path)))
    res = {}
    for r in prov:
        games = [row for row in rows if owner[int(row["game"])] == r["deck_id"]]
        c = collections.Counter(g["outcome"] for g in games)
        crash = sum(v for k, v in c.items() if k not in ("finished", "cap", "timeout"))
        cap = c.get("cap", 0) + c.get("timeout", 0)
        res[r["deck_id"]] = {"side": r["side"], "era": r["era"], "games": len(games), "outcomes": dict(c),
                             "pass": crash == 0 and cap <= 2 and len(games) == 10,
                             "errors": sorted({g["error"][:120] for g in games if g["outcome"] not in ("finished", "cap", "timeout")})}
    json.dump(res, open(out / "rescreen.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'} {k} {v['outcomes']} {v['errors'] or ''}")
    print(f"{sum(v['pass'] for v in res.values())}/{len(res)} pass")


if __name__ == "__main__":
    main()
