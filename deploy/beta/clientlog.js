// Plan client-logs: send this page's browser console to the beta server (POST /clientlog), for diagnostics and play
// testing. On by default, all levels (Peter); "?debug=0" in a page URL turns it off for this browser, "?debug=1" on again.
// Loaded first in every Gemp page (banner.py) and in GempArena's page (build_arena.sh), with data-client set.
(function () {
    "use strict";
    var KEY = "clientlogDebug", URL = "/clientlog", MAX_LINES = 20000, MAX_TEXT = 4000;
    var flag = /[?&]debug=([01])/.exec(location.search);
    try {
        if (flag) localStorage.setItem(KEY, flag[1]);
        if (localStorage.getItem(KEY) === "0") return;
    } catch (e) { /* no storage: stay on */ }

    var script = document.currentScript;
    var client = (script && script.getAttribute("data-client")) || "classic";
    var session = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
    var buffer = [], queued = 0, user = null, userAsked = 0, timer = null;
    var original = {};

    function describe(value) {
        if (typeof value === "string") return value;
        if (value instanceof Error) return value.stack || (value.name + ": " + value.message);
        try { return JSON.stringify(value); } catch (e) { return String(value); }
    }

    function add(level, text) {
        if (queued >= MAX_LINES) return;
        queued++;
        if (text.length > MAX_TEXT) text = text.slice(0, MAX_TEXT) + "…[" + text.length + " chars]";
        buffer.push([Date.now(), level, text]);
        if (buffer.length >= 200) flush(false);
        else if (!timer) timer = setTimeout(function () { flush(false); }, 2000);
    }

    // Who is logged in (Gemp's cookie is a session key, not the name); asked again at most every 30 s until known
    function askUser() {
        if (user || Date.now() - userAsked < 30000) return;
        userAsked = Date.now();
        fetch("/gemp-swccg-server/playerStats/playerInfo", {credentials: "same-origin"})
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (j) { if (j && j.name) user = j.name; })
            .catch(function () { /* not logged in */ });
    }

    function flush(leaving) {
        clearTimeout(timer);
        timer = null;
        askUser();
        while (buffer.length) {
            // batches well under the collector's 64 KB limit (JSON escaping and UTF-8 can grow the text)
            var lines = [], size = 0;
            while (buffer.length && (lines.length === 0 || size + buffer[0][2].length < 40000)) {
                size += buffer[0][2].length + 40;
                lines.push(buffer.shift());
            }
            var body = JSON.stringify({client: client, page: location.pathname, session: session, user: user, lines: lines});
            if (leaving && navigator.sendBeacon) {
                navigator.sendBeacon(URL, new Blob([body], {type: "text/plain"}));
            } else {
                fetch(URL, {method: "POST", body: body, headers: {"Content-Type": "text/plain"}})
                    .catch(function () { /* the collector is down: drop, never log about it */ });
            }
        }
    }

    ["log", "info", "warn", "error", "debug"].forEach(function (level) {
        original[level] = console[level];
        console[level] = function () {
            try { add(level, Array.prototype.map.call(arguments, describe).join(" ")); } catch (e) { /* never break the page */ }
            return original[level].apply(console, arguments);
        };
    });
    window.addEventListener("error", function (e) {
        add("uncaught", (e.error && e.error.stack) || (e.message + " at " + e.filename + ":" + e.lineno + ":" + e.colno));
    });
    window.addEventListener("unhandledrejection", function (e) {
        add("unhandledrejection", describe(e.reason));
    });
    window.addEventListener("pagehide", function () { flush(true); });
    add("info", "clientlog: " + client + " " + location.href + " | " + navigator.userAgent);
})();
