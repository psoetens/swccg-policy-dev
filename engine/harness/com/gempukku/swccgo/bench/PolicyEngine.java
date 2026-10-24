package com.gempukku.swccgo.bench;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The RL policy server inside the JVM (plan rollout-throughput, step 2): the Java port of swccg-policy's rl.RLServer on top of
 * {@link NativePolicy} (libtorch + CUDA graphs). Same model keys and semantics as the Python server:
 * <ul>
 *   <li>"learner": RUN/ckpt/latest.ts (reloaded in place when it changes), sampled, steps recorded;</li>
 *   <li>"bc": the bc-v1 policy (policy.native.bc), sampled, not recorded;</li>
 *   <li>"snap": a random one of the last 6 RUN/snaps/*.ts per game, sampled, not recorded;</li>
 *   <li>"ck:PATH": the policy PATH with .pt replaced by .ts; any key + "@greedy" = argmax ranking, not recorded.</li>
 * </ul>
 * Game threads call {@link #query}; one batcher thread collects requests for up to policy.native.waitMs (default 1 ms, at most
 * 256), runs one forward per model slot and answers every request: greedy = options by descending logit; sampled = the sampled
 * pick sequence (PolicySampler, temperature policy.native.temperature) followed by the rest of the ranking. Sampled learner
 * steps with repeats r <= the loop-guard threshold and not marked "skip" are recorded (TrajectoryWriter, reward with
 * policy.native.lfBeta).
 *
 * Configuration (system properties): policy.native.run (RUN directory), policy.native.bundle (vocab.json, cards.json),
 * policy.native.bc, policy.native.lib, policy.native.graphs (true), policy.native.tf32 (true, as the Python server),
 * policy.native.seed (1).
 */
public final class PolicyEngine {
    private static volatile PolicyEngine instance;

    /** The process-wide engine if -Dpolicy.native.run is set (created on first use), else null. */
    public static PolicyEngine get() {
        if (System.getProperty("policy.native.run") == null) return null;
        if (instance == null) synchronized (PolicyEngine.class) {
            if (instance == null) instance = new PolicyEngine();
        }
        return instance;
    }

    static final class Slot {
        final String name;
        final long handle;
        final Path source;
        long mtime;
        String label;

        Slot(String name, long handle, Path source, long mtime, String label) {
            this.name = name;
            this.handle = handle;
            this.source = source;
            this.mtime = mtime;
            this.label = label;
        }
    }

    static final class Request {
        final Slot slot;
        final boolean greedy, record;
        final String game, side, line;
        final int k, repeats;
        final boolean skip;
        final int[] excluded;
        PolicyTensors tensors;
        List<Integer> out;
        final CountDownLatch done = new CountDownLatch(1);

        Request(Slot slot, boolean greedy, boolean record, String game, String side, String line, int k, int repeats, boolean skip,
                int[] excluded) {
            this.slot = slot;
            this.greedy = greedy;
            this.record = record;
            this.game = game;
            this.side = side;
            this.line = line;
            this.k = k;
            this.repeats = repeats;
            this.skip = skip;
            this.excluded = excluded;
        }
    }

    public final AtomicLong requests = new AtomicLong(), batches = new AtomicLong(), forwards = new AtomicLong(),
            forwardNanos = new AtomicLong(), failures = new AtomicLong();
    private final Path run;
    private final PolicyTensors.Maps maps;
    private final boolean graphs;
    private final double temperature;
    private final int guardThreshold = PolicyAi.LOOP_K, maxBatch = 256, maxSnaps = 6;
    private final long waitNanos;
    private final Slot bc;
    private volatile Slot learner;
    private final Map<Path, Slot> snaps = new LinkedHashMap<>();
    private final Map<String, Slot> evalSlots = new ConcurrentHashMap<>();
    private final Map<String, Path> gameSnap = new ConcurrentHashMap<>();
    private final Map<String, Map<String, String>> gameKeys = new ConcurrentHashMap<>();
    private final Random slotRng;
    private final BlockingQueue<Request> queue = new LinkedBlockingQueue<>();
    private final TrajectoryWriter writer;

    private PolicyEngine() {
        NativePolicy.ensureLoaded();
        NativePolicy.setTf32(!"false".equals(System.getProperty("policy.native.tf32")));
        run = Paths.get(System.getProperty("policy.native.run"));
        maps = PolicyTensors.Maps.load(Paths.get(System.getProperty("policy.native.bundle")));
        graphs = !"false".equals(System.getProperty("policy.native.graphs"));
        temperature = Double.parseDouble(System.getProperty("policy.native.temperature", "1.0"));
        waitNanos = (long) (Double.parseDouble(System.getProperty("policy.native.waitMs", "1.0")) * 1e6);
        long seed = Long.getLong("policy.native.seed", 1L);
        slotRng = new Random(seed);
        writer = new TrajectoryWriter(run, Double.parseDouble(System.getProperty("policy.native.lfBeta", "0")),
                Integer.getInteger("policy.native.gamesPerFile", 20));
        bc = load("bc", Paths.get(System.getProperty("policy.native.bc")));
        initialReload();
        Thread b = new Thread(() -> batcher(new SplittableRandom(seed + 1)), "policy-batcher");
        b.setDaemon(true);
        b.start();
        Thread r = new Thread(this::reloadLoop, "policy-reload");
        r.setDaemon(true);
        r.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                writer.close();
            } catch (InterruptedException ignored) {
            }
        }));
    }

    private Slot load(String name, Path ts) {
        long h = NativePolicy.load(ts.toString(), graphs);
        if (h == 0) throw new IllegalStateException("native load " + ts + ": " + NativePolicy.lastError());
        return new Slot(name, h, ts, mtime(ts), labelOf(name, ts));
    }

    private static long mtime(Path p) {
        try {
            return Files.getLastModifiedTime(p).toMillis();
        } catch (IOException e) {
            return -1;
        }
    }

    /** "learner:ITER" from RUN/ckpt/latest.ts.json ({"iteration": N}, written by the learner), else "learner". */
    private String labelOf(String name, Path ts) {
        if (!name.equals("learner")) return name;
        try {
            String j = Files.readString(Paths.get(ts + ".json"));
            return "learner:" + com.google.gson.JsonParser.parseString(j).getAsJsonObject().get("iteration").getAsInt();
        } catch (IOException | RuntimeException e) {
            return "learner";
        }
    }

    private synchronized void reload() {
        Path latest = run.resolve("ckpt/latest.ts");
        Path src = Files.exists(latest) ? latest : bc.source;
        long m = mtime(src);
        if (learner == null) {
            learner = load("learner", src);
            System.err.println("native learner <- " + src + " (" + learner.label + ")");
        } else if (m != learner.mtime) {
            if (NativePolicy.swap(learner.handle, src.toString()) != 0)
                throw new IllegalStateException("native swap " + src + ": " + NativePolicy.lastError());
            learner.mtime = m;
            learner.label = labelOf("learner", src);
            System.err.println("native learner weights <- " + src + " (" + learner.label + ")");
        }
        List<Path> files;
        try (Stream<Path> s = Files.list(run.resolve("snaps"))) {
            // Plan nvf-b-local: by iteration number (snap-10000 after snap-9999), not by name: a name sort kept the
            // 9xxx snapshots as "newest" once the run passed iteration 10,000.
            files = s.filter(p -> p.toString().endsWith(".ts")).sorted(java.util.Comparator.comparingLong(
                    (Path p) -> iterationOf(p)).thenComparing(Path::toString)).collect(Collectors.toList());
        } catch (IOException e) {
            files = List.of();
        }
        files = files.subList(Math.max(0, files.size() - maxSnaps), files.size());
        for (Path f : files)
            if (!snaps.containsKey(f)) {
                snaps.put(f, load(f.getFileName().toString().replace(".ts", ""), f));
                System.err.println("native snapshot loaded " + f);
            }
        List<Path> keep = files;
        snaps.keySet().removeIf(f -> !keep.contains(f)); // (their GPU memory is kept: at most a few per run)
    }

    /** Plan nvf-b-local: the learner prunes old checkpoints, so ckpt/latest.ts (or a snapshot) can name a file that is
     *  gone by the time it is opened (nvf-a-r4, 2026-10-15). The reload loop already retries; the first load does too. */
    private void initialReload() {
        for (int attempt = 1; ; attempt++) {
            try {
                reload();
                return;
            } catch (IllegalStateException e) {
                if (attempt >= 5) throw e;
                System.err.println("native initial load failed (attempt " + attempt + "), retrying: " + e.getMessage());
                try {
                    Thread.sleep(2_000);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw e;
                }
            }
        }
    }

    /** The number after the last '-' of snap-NNNN.ts (-1 when there is none). */
    public static long iterationOf(Path p) {
        String name = p.getFileName().toString();
        String digits = name.substring(name.lastIndexOf('-') + 1).replaceAll("\\D.*$", "");
        return digits.isEmpty() ? -1 : Long.parseLong(digits);
    }

    private void reloadLoop() {
        while (true) {
            try {
                Thread.sleep(15_000);
                reload();
            } catch (InterruptedException e) {
                return;
            } catch (RuntimeException e) {
                System.err.println("native reload failed: " + e);
            }
        }
    }

    private Slot slotFor(String base, String game) {
        switch (base) {
            case "learner":
                return learner;
            case "bc":
                return bc;
            case "snap":
                synchronized (this) {
                    if (snaps.isEmpty()) return bc;
                    Path f = gameSnap.get(game);
                    if (f == null || !snaps.containsKey(f)) {
                        List<Path> keys = new ArrayList<>(snaps.keySet());
                        f = keys.get(slotRng.nextInt(keys.size()));
                        gameSnap.put(game, f);
                    }
                    return snaps.get(f);
                }
            default:
                if (base.startsWith("ck:")) {
                    String pt = base.substring(3);
                    return evalSlots.computeIfAbsent(pt, p -> load(p, Paths.get(p.endsWith(".pt") ? p.substring(0, p.length() - 3) + ".ts" : p)));
                }
                throw new IllegalArgumentException("unknown model key " + base);
        }
    }

    /**
     * One decision (the Python server's envelope): returns the option indices, the answer (sampled or best) first.
     * line = the StateEncoder record; k = picks; repeats = the loop guard's occurrence count; skip / excluded = step-by-step
     * selection fields.
     */
    public List<Integer> query(String modelKey, String game, String side, String line, int k, int repeats, boolean skip,
                               List<Integer> excluded) throws InterruptedException {
        boolean greedy = modelKey.endsWith("@greedy");
        String base = greedy ? modelKey.substring(0, modelKey.length() - "@greedy".length()) : modelKey;
        Slot slot = slotFor(base, game);
        gameKeys.computeIfAbsent(game, g -> new ConcurrentHashMap<>()).put(side, modelKey);
        int[] x = excluded == null ? null : excluded.stream().mapToInt(Integer::intValue).toArray();
        Request r = new Request(slot, greedy, base.equals("learner") && !greedy, game, side, line, k, repeats, skip, x);
        r.tensors = PolicyTensors.fromRecord(line, maps); // on the game thread: spreads the conversion over the cores
        queue.add(r);
        r.done.await();
        requests.incrementAndGet();
        return r.out;
    }

    /** Game end for one side (the Python server's "end" message). */
    public void finish(String game, String side, String modelKey, String outcome, String winner, String op, int[] lf) {
        Map<String, String> keys = gameKeys.getOrDefault(game, Map.of());
        String other = side.equals("ds") ? "ls" : "ds";
        String opp = keys.containsKey(other) ? keys.get(other) : (op == null || op.equals("advanced") ? "bot" : op);
        if (modelKey.equals("learner")) writer.finish(game, side, outcome, winner, lf, opp, learner.label);
        synchronized (gameKeys) {
            Map<String, String> k = gameKeys.get(game);
            if (k != null) {
                k.put("_done_" + side, "");
                if (k.containsKey("_done_" + other) || !k.containsKey(other)) {
                    gameKeys.remove(game);
                    gameSnap.remove(game);
                }
            }
        }
    }

    private void batcher(SplittableRandom rng) {
        // per slot group of a batch window: its own input / output buffers (all groups are launched before any is collected)
        List<ByteBuffer> ins = new ArrayList<>();
        List<FloatBuffer> outs = new ArrayList<>();
        float[] logits = new float[0];
        List<Request> items = new ArrayList<>();
        while (true) {
            items.clear();
            try {
                items.add(queue.take());
                long deadline = System.nanoTime() + waitNanos;
                while (items.size() < maxBatch) {
                    Request r = queue.poll(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                    if (r == null) break;
                    items.add(r);
                }
            } catch (InterruptedException e) {
                return;
            }
            Map<Slot, List<Request>> groups = new LinkedHashMap<>();
            for (Request r : items) groups.computeIfAbsent(r.slot, s -> new ArrayList<>()).add(r);
            List<Slot> launched = new ArrayList<>();
            List<int[]> shapes = new ArrayList<>();
            List<List<Request>> lgroups = new ArrayList<>();
            long t0 = System.nanoTime();
            for (Map.Entry<Slot, List<Request>> g : groups.entrySet()) {
                List<Request> grp = g.getValue();
                try {
                    List<PolicyTensors> ts = new ArrayList<>(grp.size());
                    for (Request r : grp) ts.add(r.tensors);
                    int[] sh = PolicyTensors.shape(ts);
                    int i = launched.size();
                    int bytes = PolicyTensors.packedBytes(sh[0], sh[1], sh[2]);
                    if (ins.size() <= i) {
                        ins.add(ByteBuffer.allocateDirect(Math.max(bytes, PolicyTensors.packedBytes(16, 128, 32))).order(ByteOrder.LITTLE_ENDIAN));
                        outs.add(ByteBuffer.allocateDirect(4 * 64 * PolicyTensors.O_MAX).order(ByteOrder.nativeOrder()).asFloatBuffer());
                    }
                    if (ins.get(i).capacity() < bytes) ins.set(i, ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN));
                    if (outs.get(i).capacity() < sh[0] * sh[2])
                        outs.set(i, ByteBuffer.allocateDirect(4 * sh[0] * sh[2]).order(ByteOrder.nativeOrder()).asFloatBuffer());
                    PolicyTensors.pack(ts, sh[0], sh[1], sh[2], ins.get(i));
                    if (NativePolicy.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2]) != 0)
                        throw new IllegalStateException("native launch: " + NativePolicy.lastError());
                    launched.add(g.getKey());
                    shapes.add(sh);
                    lgroups.add(grp);
                } catch (RuntimeException e) {
                    fail(grp, e);
                }
            }
            for (int i = 0; i < launched.size(); i++) {
                List<Request> grp = lgroups.get(i);
                int[] sh = shapes.get(i);
                try {
                    FloatBuffer out = outs.get(i);
                    if (NativePolicy.collect(launched.get(i).handle, out) != 0)
                        throw new IllegalStateException("native collect: " + NativePolicy.lastError());
                    if (logits.length < sh[0] * sh[2]) logits = new float[sh[0] * sh[2]];
                    out.position(0);
                    out.get(logits, 0, sh[0] * sh[2]);
                    for (int j = 0; j < grp.size(); j++) answer(grp.get(j), logits, j * sh[2], rng);
                } catch (RuntimeException e) {
                    fail(grp, e);
                }
            }
            forwardNanos.addAndGet(System.nanoTime() - t0);
            forwards.addAndGet(launched.size());
            batches.incrementAndGet();
        }
    }

    private void fail(List<Request> grp, RuntimeException e) {
        failures.incrementAndGet();
        System.err.println("native batch failed: " + e);
        for (Request r : grp)
            if (r.out == null) {
                List<Integer> all = new ArrayList<>();
                for (int i = 0; i < r.tensors.options; i++) all.add(i);
                r.out = all;
                r.done.countDown();
            }
    }

    private void answer(Request r, float[] logits, int off, SplittableRandom rng) {
        int n = r.tensors.options;
        float[] lg = Arrays.copyOfRange(logits, off, off + n);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> -lg[i])); // stable, as numpy argsort(-lg, kind="stable")
        List<Integer> ranking = Arrays.asList(order);
        if (r.greedy) {
            r.out = new ArrayList<>(ranking);
        } else {
            PolicySampler.Sample s = PolicySampler.sample(lg, n, r.tensors.isNone, r.k, temperature, r.excluded, rng);
            List<Integer> outList = new ArrayList<>(s.seq);
            for (Integer j : ranking) if (!s.seq.contains(j)) outList.add(j);
            r.out = outList;
            if (r.record && r.repeats <= guardThreshold && !r.skip && !s.seq.isEmpty())
                writer.step(r.game, r.side, r.line, s.seq, s.logp, r.repeats, r.k,
                        r.excluded == null ? null : Arrays.stream(r.excluded).boxed().collect(Collectors.toList()));
        }
        r.done.countDown();
    }

    public String stats() {
        long b = batches.get(), f = forwards.get();
        return String.format(java.util.Locale.ROOT, "native policy: %d requests, %d batches (%.2f per batch), %d forwards (%.3f ms avg), %d failures, %.3f ms GPU per batch",
                requests.get(), b, b == 0 ? 0 : requests.get() / (double) b, f, f == 0 ? 0 : forwardNanos.get() / 1e6 / f, failures.get(),
                b == 0 ? 0 : forwardNanos.get() / 1e6 / b);
    }

    /** Flush trajectories (end of run). */
    public void close() throws InterruptedException {
        writer.close();
    }
}
