package com.gempukku.swccgo.bench;

import java.nio.file.*;
import java.io.IOException;

/** Bound completed trajectories between games while keeping the JVM warm. */
public final class RolloutBackpressure {
    private static final int limit=Integer.getInteger("policy.native.maxQueueSteps",0);
    private static final long byteLimit=Long.getLong("policy.native.maxQueueBytes",0L);
    private static long checked,steps,bytes;
    private static synchronized boolean ready(Path run) throws IOException {
        long now=System.nanoTime();
        if(now-checked>250_000_000L) {
            steps=0;bytes=0;
            // The first worker can reach this before the lazy native writer
            // creates its queue. An absent queue is empty, not a failed rollout.
            if(!Files.exists(run.resolve("traj"))) { checked=now;return true; }
            try(var files=Files.newDirectoryStream(run.resolve("traj"),"part-*")) {
                for(Path p:files) try {
                    if(!p.toString().endsWith(".jsonl.gz") && !p.toString().endsWith(".traj"))continue;
                    var meta=com.google.gson.JsonParser.parseString(Files.readString(Path.of(p+".meta.json"))).getAsJsonObject();
                    steps+=meta.get("steps").getAsLong();bytes+=Files.size(p);
                } catch(NoSuchFileException ignored) { /* learner consumed it */ }
            }
            checked=now;
        }
        return (limit<=0 || steps<limit) && (byteLimit<=0 || bytes<byteLimit);
    }
    public static boolean await() throws IOException,InterruptedException {
        if(limit<=0 && byteLimit<=0) return true;
        Path run=Path.of(System.getProperty("policy.native.run"));
        while(!Files.exists(run.resolve("STOP"))) {
            if(ready(run)) return true;
            Thread.sleep(250);
        }
        return false;
    }
}
