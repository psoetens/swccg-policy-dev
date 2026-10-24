package com.gempukku.swccgo.bench;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;

/**
 * Records the learner's sampled steps and writes finished game-sides as trajectory files for swccg-policy's learner: the Java
 * port of the recording in rl.RLServer (handle / finish / writer_loop), same file format.
 *
 * File: RUN/traj/part-&lt;unix seconds&gt;-&lt;n&gt;.jsonl.gz (written as .tmp, then renamed), one JSON object per game-side:
 * {"g","s","o","w","reward","won","lf","opp","learner","steps":[{"rec":&lt;StateEncoder record&gt;,"seq","logp","r","k"[,"x"]}]}.
 * Reward: +1 / -1 for a finished game with a winner, else 0; plus lfBeta * max(0, own Life Force) / 60 (weekend plan A7).
 * Game-sides that ended in an exception are dropped. Files hold up to gamesPerFile game-sides or 60 s worth.
 */
public final class TrajectoryWriter {
    private final Path dir;
    private final double lfBeta;
    private final int gamesPerFile;
    private final Map<String, List<String>> open = new HashMap<>();      // "game\0side" -> step JSON objects
    private final BlockingQueue<String> done = new LinkedBlockingQueue<>(); // finished game-side JSON lines
    private final Thread thread;
    private volatile boolean closing;
    private int nFile;

    public TrajectoryWriter(Path runDir, double lfBeta, int gamesPerFile) {
        this.dir = runDir.resolve("traj");
        this.lfBeta = lfBeta;
        this.gamesPerFile = gamesPerFile;
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        thread = new Thread(this::writerLoop, "trajectory-writer");
        thread.setDaemon(true);
        thread.start();
    }

    /** One recorded step. rec = the StateEncoder record line (JSON array) exactly as the model saw it. */
    public void step(String game, String side, String rec, List<Integer> seq, double logp, int repeats, int k, List<Integer> excluded) {
        StringBuilder sb = new StringBuilder(rec.length() + 96);
        sb.append("{\"rec\":").append(rec).append(",\"seq\":").append(ints(seq)).append(",\"logp\":").append(logp)
                .append(",\"r\":").append(repeats).append(",\"k\":").append(k);
        if (excluded != null && !excluded.isEmpty()) sb.append(",\"x\":").append(ints(excluded));
        sb.append('}');
        synchronized (open) {
            open.computeIfAbsent(game + "\0" + side, x -> new ArrayList<>()).add(sb.toString());
        }
    }

    /**
     * The game-side ended: queue its trajectory (if it recorded any step). lf = [own, opponent] Life Force or null; opp = the
     * opponent's label for the learner's statistics; learner = the learner weights' name (e.g. "learner:419").
     */
    public void finish(String game, String side, String outcome, String winner, int[] lf, String opp, String learner) {
        List<String> steps;
        synchronized (open) {
            steps = open.remove(game + "\0" + side);
        }
        if (steps == null || outcome.equals("exception")) return;
        boolean won = outcome.equals("finished") && winner.equals(side);
        double reward = outcome.equals("finished") && (winner.equals("ds") || winner.equals("ls")) ? (won ? 1.0 : -1.0) : 0.0;
        if (lfBeta != 0 && lf != null) reward += lfBeta * Math.max(0, lf[0]) / 60.0;
        StringBuilder sb = new StringBuilder(64 + steps.size() * 4096);
        sb.append("{\"g\":").append(StateEncoder.q(game)).append(",\"s\":").append(StateEncoder.q(side)).append(",\"o\":")
                .append(StateEncoder.q(outcome)).append(",\"w\":").append(StateEncoder.q(winner)).append(",\"reward\":").append(reward)
                .append(",\"won\":").append(won ? 1 : 0).append(",\"lf\":")
                .append(lf == null ? "null" : "[" + lf[0] + "," + lf[1] + "]").append(",\"opp\":").append(StateEncoder.q(opp))
                .append(",\"learner\":").append(StateEncoder.q(learner)).append(",\"steps\":[");
        for (int i = 0; i < steps.size(); i++) sb.append(i == 0 ? "" : ",").append(steps.get(i));
        sb.append("]}");
        done.add(sb.toString());
    }

    /** Write what is queued and stop the writer thread. */
    public void close() throws InterruptedException {
        closing = true;
        thread.join();
    }

    private void writerLoop() {
        List<String> buf = new ArrayList<>();
        long last = System.nanoTime();
        while (true) {
            try {
                String s = done.poll(5, TimeUnit.SECONDS);
                if (s != null) buf.add(s);
            } catch (InterruptedException e) {
                closing = true;
            }
            boolean stop = closing && done.isEmpty();
            if (!buf.isEmpty() && (buf.size() >= gamesPerFile || System.nanoTime() - last > 60_000_000_000L || stop)) {
                write(buf);
                buf = new ArrayList<>();
                last = System.nanoTime();
            }
            if (stop) return;
        }
    }

    private void write(List<String> buf) {
        nFile++;
        Path name = dir.resolve(String.format("part-%d-%05d.jsonl.gz", System.currentTimeMillis() / 1000, nFile));
        Path tmp = dir.resolve(name.getFileName() + ".tmp");
        try (OutputStream o = new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16) {{ def.setLevel(3); }}) {
            for (String s : buf) {
                o.write(s.getBytes(StandardCharsets.UTF_8));
                o.write('\n');
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        try {
            Files.move(tmp, name, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static String ints(List<Integer> xs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < xs.size(); i++) sb.append(i == 0 ? "" : ",").append(xs.get(i));
        return sb.append(']').toString();
    }
}
