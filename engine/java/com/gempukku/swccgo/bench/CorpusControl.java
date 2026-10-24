package com.gempukku.swccgo.bench;

import java.nio.file.Files;
import java.nio.file.Path;

/** Stop corpus collection on its first error, retaining partial evidence for review. */
public final class CorpusControl {
    private static volatile boolean stopped;
    private static volatile long checked;

    public static void stop() {
        stopped = true;
    }

    public static boolean stopped() {
        long now = System.nanoTime();
        if (!stopped && now - checked > 250_000_000L) {
            String run = System.getProperty("policy.native.run");
            stopped = run != null && Files.exists(Path.of(run, "STOP"));
            checked = now;
        }
        return stopped;
    }
}
