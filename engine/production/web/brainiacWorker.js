// Brainiac: runs the policy network in a Web Worker (onnxruntime-web; WebGPU if available, otherwise WebAssembly).
//
// Messages in:  {type: "init", modelUrl, sha256, runtimeBase}
//               {type: "run", id, t}   t = base64 of the server's ActorTensors browserBytes() (plan brainiac-v6:
//                                      "SWT5", tensor ABI 5, eight floats per token; nothing else is accepted)
// Messages out: {type: "progress", loaded, total}
//               {type: "ready", backend}
//               {type: "result", id, ranking, probs, ms}   ranking = option indices, best first; probs in option order
//               {type: "error", id, message}
"use strict";

var session = null;

function post(msg) {
    self.postMessage(msg);
}

function hex(buffer) {
    var bytes = new Uint8Array(buffer), out = "";
    for (var i = 0; i < bytes.length; i++)
        out += (bytes[i] < 16 ? "0" : "") + bytes[i].toString(16);
    return out;
}

// The model bytes, from the browser cache if present (the URL names the content), otherwise downloaded with progress.
async function loadModel(url, sha256) {
    var cache = null;
    try {
        cache = await caches.open("brainiac-models");
    } catch (e) {
        cache = null; // no Cache API (e.g. private window): download every time
    }
    var response = cache ? await cache.match(url) : null;
    var fromCache = !!response;
    if (!response) {
        response = await fetch(url);
        if (!response.ok)
            throw new Error("model download failed: HTTP " + response.status);
    }
    var total = parseInt(response.headers.get("Content-Length") || "0", 10);
    var reader = response.body.getReader(), chunks = [], loaded = 0;
    for (;;) {
        var r = await reader.read();
        if (r.done)
            break;
        chunks.push(r.value);
        loaded += r.value.length;
        post({type: "progress", loaded: loaded, total: total, cached: fromCache});
    }
    var bytes = new Uint8Array(loaded), offset = 0;
    for (var i = 0; i < chunks.length; i++) {
        bytes.set(chunks[i], offset);
        offset += chunks[i].length;
    }
    var digest = hex(await crypto.subtle.digest("SHA-256", bytes));
    if (digest !== sha256) {
        if (cache)
            await cache.delete(url);
        throw new Error("model checksum mismatch");
    }
    if (cache && !fromCache) {
        try {
            await cache.put(url, new Response(bytes, {headers: {"Content-Type": "application/octet-stream"}}));
        } catch (e) {
            // cache full or unavailable: the model still works, it is only downloaded again next time
        }
    }
    return bytes;
}

function decode(base64) {
    var text = atob(base64);
    var bytes = new Uint8Array(text.length);
    for (var i = 0; i < text.length; i++) {
        bytes[i] = text.charCodeAt(i);
    }
    var view = new DataView(bytes.buffer);
    var position = 0;
    function i32() {
        var value = view.getInt32(position, true);
        position += 4;
        return value;
    }
    if (i32() !== 0x35545753) {
        throw new Error("Not a tensor-ABI-5 (SWT5) policy payload; the server and this page disagree on the release");
    }
    var T = i32();
    var O = i32();
    var K = i32();
    var expected = 104 + T * 64 + O * 24 + K * 153;
    if (T < 0 || T > 1000000 || O < 1 || O > 100000 || K < 0 || K > T || expected !== bytes.length) {
        throw new Error("Invalid policy tensor payload dimensions or byte length");
    }
    var head = new Int32Array(22);
    for (var j = 0; j < 22; j++) {
        head[j] = i32();
    }
    var tok = new Int32Array(Math.max(T, 1) * 8);
    for (j = 0; j < T * 8; j++) {
        tok[j] = i32();
    }
    var tokf = new Float32Array(Math.max(T, 1) * 8);
    for (j = 0; j < T * 8; j++) {
        tokf[j] = view.getFloat32(position, true);
        position += 4;
    }
    var opt = new Int32Array(O * 6);
    for (j = 0; j < O * 6; j++) {
        opt[j] = i32();
    }
    var grid = new Uint8Array(Math.max(K, 1) * 153);
    grid.set(bytes.subarray(position));
    return {T: T, O: O, K: K, head: head, tok: tok, tokf: tokf, opt: opt, kgrid: grid};
}

// The network's inputs for one decision (a batch of one), as the training code builds them.
function feeds(d) {
    var T = Math.max(d.T, 1);
    var tmask = new Uint8Array(T);
    for (var i = 0; i < d.T; i++) {
        tmask[i] = 1;
    }
    var omask = new Uint8Array(d.O).fill(1);
    var result = {
        tok: new ort.Tensor("int32", Int32Array.from(d.tok), [1, T, 8]),
        tokf: new ort.Tensor("float32", d.tokf, [1, T, 8]),
        tmask: new ort.Tensor("bool", tmask, [1, T]),
        opt: new ort.Tensor("int32", d.opt, [1, d.O, 6]),
        omask: new ort.Tensor("bool", omask, [1, d.O]),
        dtype: new ort.Tensor("int32", d.head.slice(0, 1), [1]),
        qtmpl: new ort.Tensor("int32", d.head.slice(1, 2), [1]),
        qbp: new ort.Tensor("int32", d.head.slice(2, 3), [1]),
        side: new ort.Tensor("int32", d.head.slice(3, 4), [1]),
        glob: new ort.Tensor("int32", d.head.slice(4, 22), [1, 18]),
        kgrid: new ort.Tensor("uint8", d.kgrid, [1, Math.max(d.K, 1), 153])
    };
    var live = {};
    session.inputNames.forEach(function (name) {
        live[name] = result[name];
    });
    return live;
}

async function run(d) {
    var out = await session.run(feeds(d));
    var logits = Array.from(out.logits.data).slice(0, d.O);
    var ranking = logits.map(function (x, i) { return i; });
    ranking.sort(function (a, b) { return logits[b] - logits[a]; });
    var max = logits[ranking[0]], sum = 0;
    var probs = logits.map(function (x) {
        var value = isFinite(x) ? Math.exp(x - max) : 0;
        sum += value;
        return value;
    });
    probs = probs.map(function (e) { return sum > 0 ? e / sum : 0; });
    return {ranking: ranking, probs: probs, logits: logits};
}

async function init(msg) {
    importScripts(msg.runtimeBase + "ort.webgpu.min.js");
    ort.env.wasm.wasmPaths = msg.runtimeBase;
    ort.env.wasm.numThreads = 1; // the page is not cross-origin isolated
    var bytes = await loadModel(msg.modelUrl, msg.sha256);
    var backend = null, lastError = null;
    var providers = (msg.backends || ["webgpu", "wasm"]);
    for (var i = 0; i < providers.length && session == null; i++) {
        try {
            if (providers[i] === "webgpu" && !(self.navigator && navigator.gpu))
                continue;
            session = await ort.InferenceSession.create(bytes, {executionProviders: [providers[i]]});
            backend = providers[i];
            // Warm-up and self-check on a minimal decision: one token, two options
            var d = {T: 1, O: 2, K: 0, head: new Int32Array(22), tok: new Int32Array(8), tokf: new Float32Array(8),
                opt: new Int32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), kgrid: new Uint8Array(153)};
            d.head[0] = 4;
            var r = await run(d);
            if (r.ranking.length !== 2 || !r.logits.every(isFinite))
                throw new Error("self-check failed");
        } catch (e) {
            lastError = e;
            session = null;
            backend = null;
        }
    }
    if (session == null)
        throw new Error("no usable backend" + (lastError ? ": " + lastError.message : ""));
    post({type: "ready", backend: backend});
}

self.onmessage = async function (ev) {
    var msg = ev.data;
    try {
        if (msg.type === "init") {
            await init(msg);
        } else if (msg.type === "run") {
            var t0 = performance.now();
            var r = await run(decode(msg.t));
            post({type: "result", id: msg.id, ranking: r.ranking, probs: r.probs,
                logits: msg.withLogits ? r.logits : undefined,
                ms: performance.now() - t0});
        }
    } catch (e) {
        post({type: "error", id: msg.id, message: String(e && e.message ? e.message : e)});
    }
};
