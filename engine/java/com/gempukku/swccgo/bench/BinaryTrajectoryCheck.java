package com.gempukku.swccgo.bench;

import com.google.gson.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/** CPU-only replay of saved trajectories through the actual Java binary writer. */
public final class BinaryTrajectoryCheck {
    private record Step(PolicyTensors actor,PolicyTensors critic,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,String policy) {}
    private record Game(JsonObject meta,List<Step> steps) {}
    static List<Integer> ints(JsonArray a){List<Integer> out=new ArrayList<>();for(JsonElement e:a)out.add(e.getAsInt());return out;}
    public static void check(StateEncoder.Encoded e,PolicyTensors.Maps maps,PolicyTensors actor) {
        same(actor,PolicyTensors.fromRecord(e.line("parity",0,List.of()),maps));
        if(e.binary.privileged!=null) {
            same(e.binary.critic(e,maps),PolicyTensorsV4.fromRecord(
                JsonParser.parseString(e.trainingLine("parity",0,List.of())).getAsJsonArray(),maps,true));
        }
    }
    static void same(PolicyTensors a,PolicyTensors b) {
        if(!Arrays.equals(a.tok,b.tok)||!Arrays.equals(a.tok32,b.tok32)||!Arrays.equals(a.kgrid,b.kgrid)
                ||a.gridRows!=b.gridRows||!Arrays.equals(a.tokf,b.tokf)||!Arrays.equals(a.opt,b.opt)||!Arrays.equals(a.head,b.head))
            throw new IllegalStateException("direct binary vs JSON tensor mismatch");
    }
    public static void main(String[] args)throws Exception {
        var maps=PolicyTensors.Maps.load(Path.of(args[1]));List<Game> games=new ArrayList<>();int count=0;
        try(var in=new com.google.gson.stream.JsonReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(Path.of(args[0])))))) {
            in.setLenient(true);
            while(in.peek()!=com.google.gson.stream.JsonToken.END_DOCUMENT){
                var g=new JsonObject();List<Step> steps=new ArrayList<>();in.beginObject();
                while(in.hasNext()) {
                    String name=in.nextName();
                    if(!name.equals("steps")){g.add(name,JsonParser.parseReader(in));continue;}
                    in.beginArray();while(in.hasNext()) {
                        var s=JsonParser.parseReader(in).getAsJsonObject();var r=s.getAsJsonArray("rec");var actor=PolicyTensors.fromRecord(r.toString(),maps);
                        long knowledge=r.get(12).getAsLong();var cr=r.deepCopy();cr.set(8,cr.get(11));cr.set(9,new JsonArray());cr.set(10,new JsonArray());
                        int[] excluded=s.has("x")?ints(s.getAsJsonArray("x")).stream().mapToInt(Integer::intValue).toArray():null;
                        steps.add(new Step(actor,PolicyTensors.fromRecord(cr.toString(),maps),knowledge,ints(s.getAsJsonArray("seq")),s.get("logp").getAsDouble(),s.get("r").getAsInt(),s.get("k").getAsInt(),excluded,s.get("policy").getAsString()));count++;
                    }in.endArray();
                }in.endObject();games.add(new Game(g,steps));
            }
        }
        // Repeat codec order to reduce JVM warmup / changing concurrent-load bias.
        for(int rep=0;rep<4;rep++)for(int level:new int[]{0,1,3,6}){
            System.setProperty("policy.trajectoryCompression",String.valueOf(level));Path dir=Path.of(args[2],"rep"+rep+"-gzip"+level);Files.createDirectories(dir);
            long start=System.nanoTime();var writer=new BinaryTrajectoryWriter(dir,4);
            for(Game game:games){var g=game.meta;String id=g.get("g").getAsString(),side=g.get("s").getAsString();
                for(Step s:game.steps)writer.step(id,side,s.actor,s.critic,s.knowledge,s.seq,s.logp,s.repeats,s.k,s.excluded,s.policy,maps);
                int[] lf=g.get("lf").isJsonNull()?null:ints(g.getAsJsonArray("lf")).stream().mapToInt(Integer::intValue).toArray();
                writer.finish(id,side,g.get("o").getAsString(),g.get("w").getAsString(),lf,g.get("opp").getAsString(),g.get("learner").getAsString());
            }
            double producer=(System.nanoTime()-start)/1e9;writer.close();double elapsed=(System.nanoTime()-start)/1e9;long bytes=0;try(var paths=Files.newDirectoryStream(dir.resolve("traj"),"*.traj")){for(Path p:paths)bytes+=Files.size(p);}
            System.out.printf(Locale.ROOT,"{\"repeat\":%d,\"level\":%d,\"steps\":%d,\"bytes\":%d,\"producer_seconds\":%.6f,\"seconds\":%.6f}%n",rep,level,count,bytes,producer,elapsed);
        }
    }
}
