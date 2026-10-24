package com.gempukku.swccgo.game.knowledge;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.nio.file.*;

/** Opt-in wall-time counters. Concurrent thread totals can exceed elapsed time. */
public final class KnowledgeTiming {
    private static final String path=System.getProperty("policy.timing");
    private static final ConcurrentHashMap<String,LongAdder[]> counters=new ConcurrentHashMap<>();
    static {
        if(path!=null) Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            StringBuilder out=new StringBuilder("{\n");
            for(var e:new java.util.TreeMap<>(counters).entrySet()) {
                if(out.length()>2) out.append(",\n");
                out.append('"').append(e.getKey()).append("\":{\"calls\":")
                    .append(e.getValue()[0].sum()).append(",\"seconds\":")
                    .append(e.getValue()[1].sum()/1e9).append('}');
            }
            try {Files.writeString(Path.of(path),out.append("\n}\n").toString());}
            catch(Exception e) {e.printStackTrace();}
        },"knowledge-timing"));
    }
    public static long start() {return path==null?0:System.nanoTime();}
    public static void end(String key,long start) {
        if(start==0) return;
        long elapsed=System.nanoTime()-start;
        var v=counters.computeIfAbsent(key,k -> new LongAdder[]{new LongAdder(),new LongAdder()});
        v[0].increment();v[1].add(elapsed);
    }
}
