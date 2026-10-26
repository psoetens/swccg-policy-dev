"""Plan client-logs: the beta's collector for browser console lines (deploy/beta/clientlog.js).

    POST /clientlog      {"client", "page", "session", "user", "lines": [[ms, level, text], ...]} -> 204
    GET  /clientlog.js   the browser script

Appends one JSON object per line to DATA/<UTC date>.jsonl: time (UTC), client, page, account, session, client IP
(X-Forwarded-For from Caddy), level, text. Limits: 64 KB per request (413), 20,000 lines per session, 600 requests per
minute per IP (429); files older than 14 days are deleted. Standard library only (runs in python:3.12-alpine).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path(os.environ.get("CLIENTLOG_DATA", "/data"))
SCRIPT = Path(os.environ.get("CLIENTLOG_SCRIPT", "/app/clientlog.js"))
PORT = int(os.environ.get("CLIENTLOG_PORT", "17102"))
MAX_BODY, MAX_SESSION_LINES, MAX_PER_MINUTE, KEEP_DAYS, MAX_TEXT = 64 * 1024, 20000, 600, 14, 4000

lock = threading.Lock()
session_lines: dict[str, int] = defaultdict(int)
session_seen: dict[str, float] = {}
ip_times: dict[str, deque] = defaultdict(deque)
last_cleanup = [0.0]


def text(value, limit: int = 200) -> str | None:
    return None if value is None else str(value)[:limit]


def cleanup(now: float) -> None:
    if now - last_cleanup[0] < 3600:
        return
    last_cleanup[0] = now
    for f in DATA.glob("*.jsonl"):
        if now - f.stat().st_mtime > KEEP_DAYS * 86400:
            f.unlink()
    for s in [s for s, t in session_seen.items() if now - t > 86400]:
        session_seen.pop(s, None)
        session_lines.pop(s, None)


class Handler(BaseHTTPRequestHandler):
    server_version = "clientlog"

    def log_message(self, *args):  # no access log: the lines themselves are the log
        pass

    def reply(self, code: int, body: bytes = b"", kind: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if kind.startswith("text/javascript"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/clientlog.js":
            self.reply(200, SCRIPT.read_bytes(), "text/javascript; charset=utf-8")
        else:
            self.reply(404)

    def do_POST(self):
        if self.path.split("?")[0] != "/clientlog":
            return self.reply(404)
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            return self.reply(413)
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        now = time.time()
        with lock:
            times = ip_times[ip]
            while times and now - times[0] > 60:
                times.popleft()
            if len(times) >= MAX_PER_MINUTE:
                return self.reply(429)
            times.append(now)
        try:
            msg = json.loads(self.rfile.read(size))
            lines = msg["lines"]
            assert isinstance(lines, list)
        except Exception:
            return self.reply(400)
        session = text(msg.get("session"), 64) or "?"
        base = {"client": text(msg.get("client"), 20), "page": text(msg.get("page")), "user": text(msg.get("user"), 40),
                "session": session, "ip": ip}
        out = []
        with lock:
            room = MAX_SESSION_LINES - session_lines[session]
            for line in lines[:max(room, 0)]:
                try:
                    ms, level, body = line
                    t = datetime.fromtimestamp(float(ms) / 1000, timezone.utc).isoformat(timespec="milliseconds")
                except Exception:
                    continue
                out.append(json.dumps({"t": t, **base, "level": text(level, 20), "text": text(body, MAX_TEXT + 40)},
                                      ensure_ascii=False))
            session_lines[session] += len(out)
            session_seen[session] = now
            if out:
                day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                with open(DATA / f"{day}.jsonl", "a", encoding="utf-8") as f:
                    f.write("\n".join(out) + "\n")
            cleanup(now)
        self.reply(204)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
