package com.gempukku.swccgo.bench;

import java.io.*;
import com.gempukku.swccgo.game.knowledge.KnowledgeTiming;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPOutputStream;

/** Bounded compressed trajectories. Concatenated gzip members form ordinary
 * JSONL after decompression. No uncompressed game-length observation buffer. */
public final class TrajectoryWriter {
    private static final class Open {
        final ByteArrayOutputStream bytes=new ByteArrayOutputStream();
        final GZIPOutputStream zip;
        int steps;
        Open() throws IOException {zip=gzip(bytes);}
        void add(String json) throws IOException {
            if(steps++>0) zip.write(',');
            zip.write(json.getBytes(StandardCharsets.UTF_8));
        }
    }
    private static final class Ready {
        final byte[] bytes;final int steps;
        Ready(byte[] bytes,int steps) {this.bytes=bytes;this.steps=steps;}
    }
    private final Path dir;
    private final BinaryTrajectoryWriter binary;
    // Plan critic-noise stage 0: the legacy actor view of the same steps (-Dpolicy.dualLegacy=DIR, files in DIR/traj).
    private final BinaryTrajectoryWriter legacy;
    private final int gamesPerFile;
    private final Map<String,Open> open=new HashMap<>();
    private final BlockingQueue<Ready> done=new ArrayBlockingQueue<>(32);
    private final Thread thread;
    private volatile boolean closing;
    private volatile RuntimeException failure;
    private int nFile;
    private static GZIPOutputStream gzip(OutputStream out) throws IOException {
        return new GZIPOutputStream(out,1<<16) {{def.setLevel(3);}};
    }
    private static void member(OutputStream out,String text) throws IOException {
        GZIPOutputStream zip=gzip(out);zip.write(text.getBytes(StandardCharsets.UTF_8));zip.finish();
    }
    public TrajectoryWriter(Path runDir,int gamesPerFile) {
        binary=Boolean.getBoolean("policy.binaryTrajectory")?new BinaryTrajectoryWriter(runDir,gamesPerFile):null;
        String dual=System.getProperty("policy.dualLegacy");
        legacy=binary!=null && dual!=null?new BinaryTrajectoryWriter(Path.of(dual),gamesPerFile):null;
        dir=runDir.resolve("traj");this.gamesPerFile=gamesPerFile;
        try {Files.createDirectories(dir);} catch(IOException e) {throw new UncheckedIOException(e);}
        thread=new Thread(this::writerLoop,"trajectory-writer");thread.setDaemon(true);thread.start();
    }
    public void step(String game,String side,String rec,List<Integer> seq,double logp,int repeats,int k,List<Integer> excluded,String policy) {
        long timing=KnowledgeTiming.start();
        if(failure!=null) throw failure;
        StringBuilder sb=new StringBuilder(rec.length()+128);
        sb.append("{\"rec\":").append(rec).append(",\"seq\":").append(seq).append(",\"logp\":").append(logp)
            .append(",\"r\":").append(repeats).append(",\"k\":").append(k).append(",\"policy\":").append(StateEncoder.q(policy));
        if(excluded!=null && !excluded.isEmpty()) sb.append(",\"x\":").append(excluded);
        sb.append('}');
        try {
            Open o;
            synchronized(open) {
                String key=game+"\0"+side;o=open.get(key);
                if(o==null) {o=new Open();open.put(key,o);}
            }
            // A game has one requesting thread; finish occurs after query returns.
            // Lock only this stream, never all games during compression.
            synchronized(o) {o.add(sb.toString());}
        } catch(IOException e) {throw new UncheckedIOException(e);}
        finally {KnowledgeTiming.end("trajectory_step",timing);}
    }
    public void stepBinary(String game,String side,PolicyTensors actor,PolicyTensors critic,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,String policy,PolicyTensors.Maps maps) {
        long timing=KnowledgeTiming.start();
        try {if(binary==null)throw new IllegalStateException("binary writer disabled");binary.step(game,side,actor,critic,knowledge,seq,logp,repeats,k,excluded,policy,maps);}
        finally {KnowledgeTiming.end("trajectory_step",timing);}
    }
    public void stepLegacy(String game,String side,PolicyTensors actor,PolicyTensors critic,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,String policy,PolicyTensors.Maps maps) {
        if(legacy==null) throw new IllegalStateException("legacy writer disabled");
        legacy.step(game,side,actor,critic,knowledge,seq,logp,repeats,k,excluded,policy,maps);
    }
    public void finish(String game,String side,String outcome,String winner,int[] lf,String opp,String learner) {
        if(legacy!=null) legacy.finish(game,side,outcome,winner,lf,opp,learner);
        if(binary!=null){binary.finish(game,side,outcome,winner,lf,opp,learner);return;}
        Open o;synchronized(open) {o=open.remove(game+"\0"+side);}
        if(o==null) return;
        try {
            o.zip.close();
            if(outcome.equals("exception")) return;
            boolean won=outcome.equals("finished") && winner.equals(side);
            double reward=outcome.equals("finished") && (winner.equals("ds")||winner.equals("ls"))?(won?1.0:-1.0):0;
            String header="{\"g\":"+StateEncoder.q(game)+",\"s\":"+StateEncoder.q(side)+",\"o\":"+StateEncoder.q(outcome)
                +",\"w\":"+StateEncoder.q(winner)+",\"reward\":"+reward+",\"won\":"+(won?1:0)
                +",\"lf\":"+(lf==null?"null":Arrays.toString(lf))+",\"opp\":"+StateEncoder.q(opp)
                +",\"learner\":"+StateEncoder.q(learner)+",\"steps\":[";
            ByteArrayOutputStream bytes=new ByteArrayOutputStream(o.bytes.size()+512);
            member(bytes,header);o.bytes.writeTo(bytes);member(bytes,"]}\n");
            Ready ready=new Ready(bytes.toByteArray(),o.steps);
            while(!done.offer(ready,1,TimeUnit.SECONDS)) if(failure!=null) throw failure;
        } catch(IOException e) {throw new UncheckedIOException(e);}
        catch(InterruptedException e) {Thread.currentThread().interrupt();throw new IllegalStateException(e);}
    }
    public void close() throws InterruptedException {if(legacy!=null)legacy.close();if(binary!=null)binary.close();closing=true;thread.join();if(failure!=null) throw failure;}
    private void writerLoop() {
        List<Ready> buf=new ArrayList<>();long last=System.nanoTime();
        try {
            while(true) {
                Ready item=done.poll(1,TimeUnit.SECONDS);if(item!=null) buf.add(item);
                boolean stop=closing && done.isEmpty();
                if(!buf.isEmpty() && (buf.size()>=gamesPerFile || System.nanoTime()-last>15_000_000_000L || stop)) {
                    write(buf);buf.clear();last=System.nanoTime();
                }
                if(stop) return;
            }
        } catch(IOException e) {failure=new UncheckedIOException(e);throw failure;}
        catch(InterruptedException e) {Thread.currentThread().interrupt();failure=new IllegalStateException(e);}
    }
    private void write(List<Ready> buf) throws IOException {
        String base=String.format("part-%d-%d-%05d.jsonl.gz",System.currentTimeMillis(),ProcessHandle.current().pid(),++nFile);
        Path target=dir.resolve(base),tmp=dir.resolve(base+".tmp");int steps=0;
        try(OutputStream out=Files.newOutputStream(tmp)) {
            for(Ready item:buf) {out.write(item.bytes);steps+=item.steps;}
        }
        Files.move(tmp,target,StandardCopyOption.ATOMIC_MOVE);
    }
}
