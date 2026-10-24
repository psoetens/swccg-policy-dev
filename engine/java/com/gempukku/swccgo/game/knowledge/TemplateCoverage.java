package com.gempukku.swccgo.game.knowledge;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;
import java.nio.file.*;
/** Frozen vocabulary lookup with bounded examples and complete unknown counters. */
public final class TemplateCoverage {
    private static final Map<String,LongAdder> total=new ConcurrentHashMap<>(),unknown=new ConcurrentHashMap<>();
    private static final Map<String,Set<String>> examples=new HashMap<>();
    static {
        Runtime.getRuntime().addShutdownHook(new Thread(()-> {
            String output=System.getProperty("policy.vocabularyCoverage");if(output==null)return;
            StringJoiner report=new StringJoiner(",","{","}\n");
            for(String family:List.of("q","o","context")) {
                StringJoiner sample=new StringJoiner(",","[","]");
                synchronized(examples){for(String s:examples.getOrDefault(family,Set.of()))sample.add(quote(s));}
                report.add(quote(family)+":{\"total\":"+total.getOrDefault(family,new LongAdder()).sum()+",\"unknown\":"+
                    unknown.getOrDefault(family,new LongAdder()).sum()+",\"examples\":"+sample+"}");
            }
            try{Files.writeString(Path.of(output),report.toString());}catch(Exception e){System.err.println("vocabulary coverage: "+e);}
        },"vocabulary-coverage"));
    }
    public static int lookup(Map<String,Integer> vocabulary,String family,String text) {
        total.computeIfAbsent(family,k->new LongAdder()).increment();Integer id=vocabulary.get(text);
        if(id!=null)return id;
        unknown.computeIfAbsent(family,k->new LongAdder()).increment();
        synchronized(examples) {
            Set<String> set=examples.computeIfAbsent(family,k->new TreeSet<>());
            if(set.size()<100)set.add(text.length()>500?text.substring(0,500):text);
        }
        return 0;
    }
    public static String quote(String s) {
        StringBuilder b=new StringBuilder("\"");
        for(char c:s.toCharArray()) {
            if(c=='\\' || c=='\"')b.append('\\').append(c);
            else if(c<32)b.append(String.format("\\u%04x",(int)c));else b.append(c);
        }
        return b.append('"').toString();
    }
}
