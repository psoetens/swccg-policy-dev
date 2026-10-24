package com.gempukku.swccgo.bench;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.*;

/** SWTRJ003 (schema 4): little-endian tensor blocks, big-endian framing; independently compressed game sides. Plan
 * remove-old-generations removed the SWTRJ001 / SWTRJ002 (schema 2 / 3) variants. */
public final class BinaryTrajectoryWriter {
    private final Path dir;
    private static volatile BinaryTrajectoryWriter active;
    private final int gamesPerFile;
    private final int level=Integer.getInteger("policy.trajectoryCompression",1); // 0=raw, 1..9=gzip
    private final Map<String,Open> open=new HashMap<>();
    private final BlockingQueue<Ready> done=new ArrayBlockingQueue<>(32);
    private final Thread thread;
    private volatile boolean closing;
    private volatile RuntimeException failure;
    private byte[] fingerprint;
    private String tableSha;
    private int schema=4;
    private int serial;
    private final class Open {
        final ByteArrayOutputStream bytes=new ByteArrayOutputStream();
        final CRC32 crc=new CRC32();
        final OutputStream compressor;
        final CheckedOutputStream out;
        long rawBytes;int steps;int width=4;final StringBuilder dojo=new StringBuilder();
        ByteBuffer scratch=ByteBuffer.allocate(65536);
        Open() throws IOException {
            compressor=level==0?bytes:new GZIPOutputStream(bytes,65536) {{def.setLevel(level);}};
            out=new CheckedOutputStream(compressor,crc);
        }
        void add(PolicyTensors a,PolicyTensors c,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,int version) throws IOException {
            if (schema == 4) {
                addV4(a,c,knowledge,seq,logp,repeats,k,excluded,version);
                return;
            }
            int nx=excluded==null?-1:excluded.length;
            int size=Math.addExact(48+88+4*(seq.size()+Math.max(0,nx)),Math.addExact(32*(a.tokens+c.tokens),24*a.options));
            if(scratch.capacity()<size) scratch=ByteBuffer.allocate(size);
            ByteBuffer b=scratch;b.clear();b.order(ByteOrder.BIG_ENDIAN);
            b.putInt(a.tokens).putInt(a.options).putInt(c.tokens).putInt(seq.size()).putInt(nx).putInt(version).putInt(repeats).putInt(k).putDouble(logp).putLong(knowledge);
            b.order(ByteOrder.LITTLE_ENDIAN);
            for(int h:a.head)b.putInt(h);
            tensors(b,a);for(int v:a.opt)b.putInt(v);tensors(b,c);
            b.order(ByteOrder.BIG_ENDIAN);for(int v:seq)b.putInt(v);if(excluded!=null)for(int v:excluded)b.putInt(v);
            if(b.position()!=size)throw new IllegalStateException("binary size mismatch");
            out.write(b.array(),0,size);rawBytes+=size;steps++;
        }
        void addV4(PolicyTensors a,PolicyTensors c,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,int version) throws IOException {
            if(a.floatWidth!=c.floatWidth||steps>0&&a.floatWidth!=width)throw new IllegalArgumentException("mixed float widths");width=a.floatWidth;
            if (a.privileged || !c.privileged || a.tok32 == null || c.tok32 == null
                    || !tableSha.equals(a.tableSha) || !tableSha.equals(c.tableSha)
                    || !Arrays.equals(a.head,c.head) || !Double.isFinite(logp)) {
                throw new IllegalArgumentException("Invalid ABI-4 trajectory views/table/head");
            }
            int nx=excluded==null?-1:excluded.length;
            for(int selected:seq)if(selected<0||selected>=a.options)throw new IllegalArgumentException("Invalid choice");
            if(excluded!=null)for(int selected:excluded)if(selected<0||selected>=a.options)throw new IllegalArgumentException("Invalid exclusion");
            int size=Math.addExact(56+88+4*(seq.size()+Math.max(0,nx)),
                Math.addExact((32+4*a.floatWidth)*a.tokens+(32+4*c.floatWidth)*c.tokens+24*a.options,153*(a.gridRows+c.gridRows)));
            if(scratch.capacity()<size)scratch=ByteBuffer.allocate(size);
            ByteBuffer b=scratch;b.clear();b.order(ByteOrder.BIG_ENDIAN);
            b.putInt(a.tokens).putInt(a.options).putInt(c.tokens).putInt(a.gridRows).putInt(c.gridRows)
                .putInt(seq.size()).putInt(nx).putInt(version).putInt(repeats).putInt(k).putDouble(logp).putLong(knowledge);
            b.order(ByteOrder.LITTLE_ENDIAN);
            for(int h:a.head)b.putInt(h);
            tensorsV4(b,a);for(int v:a.opt)b.putInt(v);b.put(a.kgrid);
            tensorsV4(b,c);b.put(c.kgrid);
            b.order(ByteOrder.BIG_ENDIAN);for(int v:seq)b.putInt(v);if(excluded!=null)for(int v:excluded)b.putInt(v);
            if(b.position()!=size)throw new IllegalStateException("ABI-4 binary size mismatch");
            out.write(b.array(),0,size);rawBytes+=size;steps++;
        }
    }
    private static void tensors(ByteBuffer b,PolicyTensors t) {
        for(short v:t.tok)b.putShort(v);for(float v:t.tokf)b.putFloat(v);
    }
    private static void tensorsV4(ByteBuffer b,PolicyTensors t) {
        for(int v:t.tok32)b.putInt(v);for(float v:t.tokf)b.putFloat(v);
    }
    private record Ready(byte[] data,byte[] metadata,long rawBytes,int checksum,int steps) {}
    public BinaryTrajectoryWriter(Path run,int gamesPerFile) {
        if(level<0 || level>9)throw new IllegalArgumentException("compression must be 0..9");
        this.dir=run.resolve("traj");this.gamesPerFile=gamesPerFile;
        try {Files.createDirectories(dir);}catch(IOException e){throw new UncheckedIOException(e);}
        active=this;
        thread=new Thread(this::loop,"binary-trajectory-writer");thread.setDaemon(true);thread.start();
    }
    static byte[] fingerprint(PolicyTensors.Maps maps) {
        try {
            MessageDigest md=MessageDigest.getInstance("SHA-256");
            new TreeMap<>(maps.cards).forEach((k,v)->md.update((k+"\t"+v+"\n").getBytes(StandardCharsets.UTF_8)));
            return md.digest();
        }catch(NoSuchAlgorithmException e){throw new IllegalStateException(e);}
    }
    public void step(String game,String side,PolicyTensors actor,PolicyTensors critic,long knowledge,List<Integer> seq,double logp,int repeats,int k,int[] excluded,String policy,PolicyTensors.Maps maps) {
        if(failure!=null)throw failure;
        if(!policy.startsWith("learner:"))throw new IllegalArgumentException("missing policy version");
        try {
            Open o;synchronized(open) {
                if(maps.schemaVersion!=4)throw new IllegalArgumentException("binary trajectories are schema 4 only");
                if(fingerprint==null) {fingerprint=fingerprint(maps);schema=maps.schemaVersion;tableSha=maps.tableSha;}
                if(schema!=maps.schemaVersion)throw new IllegalArgumentException("mixed trajectory schemas");
                if(schema==4 && !tableSha.equals(maps.tableSha))throw new IllegalArgumentException("mixed acting tables");
                String key=game+"\0"+side;o=open.get(key);if(o==null){o=new Open();open.put(key,o);StringBuilder early=PENDING_DOJO.remove(key);if(early!=null)o.dojo.append(early);}
            }
            synchronized(o){o.add(actor,critic,knowledge,seq,logp,repeats,k,excluded,Integer.parseInt(policy.substring(8)));}
        }catch(IOException e){throw new UncheckedIOException(e);}
    }
    /** Plan objective-dojo: a signed Dojo reward on the side's last decision before the event ([step, value, item, milestone]). */
    public static void markDojo(String game,String side,double value,String item,String milestone) {
        BinaryTrajectoryWriter w=active;if(w==null)return;
        String key=game+"\0"+side;String entry=","+StateEncoder.q(item)+","+StateEncoder.q(milestone)+"]";
        Open o;synchronized(w.open){
            o=w.open.get(key);
            // Before the side's first recorded decision (e.g. during setup): book it on step 0 once the trajectory opens.
            if(o==null){PENDING_DOJO.computeIfAbsent(key,k->new StringBuilder()).append(PENDING_DOJO.get(key).length()==0?"":",").append("[0,").append(value).append(entry);return;}
        }
        synchronized(o){o.dojo.append(o.dojo.length()==0?"":",").append("[").append(Math.max(0,o.steps-1)).append(",").append(value).append(entry);}
    }
    /** Dojo rewards of game sides whose trajectory has not opened yet; non-learner sides are dropped when the game ends. */
    private static final Map<String,StringBuilder> PENDING_DOJO=new java.util.concurrent.ConcurrentHashMap<>();
    /** Plan objective-dojo: forget the pending rewards of a finished game (sides that never recorded a decision). */
    public static void releaseDojo(String game) {
        PENDING_DOJO.keySet().removeIf(key->key.startsWith(game+"\0"));
    }
    public void finish(String game,String side,String outcome,String winner,int[] lf,String opp,String learner) {
        Open o;synchronized(open){o=open.remove(game+"\0"+side);}if(o==null)return;
        try {
            o.out.close();if(outcome.equals("exception"))return;
            boolean won=outcome.equals("finished")&&winner.equals(side);
            double reward=outcome.equals("finished")&&(winner.equals("ds")||winner.equals("ls"))?(won?1:-1):0;
            // Standard Gameplay reward (plan objective-dojo); the Dojo's item rewards travel separately as dojo_rewards.
            String meta="{\"g\":"+StateEncoder.q(game)+",\"s\":"+StateEncoder.q(side)+",\"o\":"+StateEncoder.q(outcome)+",\"w\":"+StateEncoder.q(winner)
                +",\"reward\":"+reward+",\"won\":"+(won?1:0)+",\"lf\":"+(lf==null?"null":Arrays.toString(lf))+",\"opp\":"+StateEncoder.q(opp)
                +",\"learner\":"+StateEncoder.q(learner)+(o.dojo.length()==0?"":",\"dojo_rewards\":["+o.dojo+"]")+",\"n_steps\":"+o.steps
                +(schema==4?",\"schema_version\":4,\"table_sha\":"+StateEncoder.q(tableSha):"")+(o.width==8?",\"float_width\":8":"")+"}";
            Ready r=new Ready(o.bytes.toByteArray(),meta.getBytes(StandardCharsets.UTF_8),o.rawBytes,(int)o.crc.getValue(),o.steps);
            while(!done.offer(r,1,TimeUnit.SECONDS))if(failure!=null)throw failure;
        }catch(IOException e){throw new UncheckedIOException(e);}catch(InterruptedException e){Thread.currentThread().interrupt();throw new IllegalStateException(e);}
    }
    public void close() throws InterruptedException {closing=true;thread.join();if(failure!=null)throw failure;}
    private void loop() {
        List<Ready> buf=new ArrayList<>();long last=System.nanoTime();
        try {
            while(true){Ready r=done.poll(1,TimeUnit.SECONDS);if(r!=null)buf.add(r);boolean stop=closing&&done.isEmpty();
                if(!buf.isEmpty()&&(buf.size()>=gamesPerFile||System.nanoTime()-last>15_000_000_000L||stop)){write(buf);buf.clear();last=System.nanoTime();}
                if(stop)return;
            }
        }catch(IOException e){failure=new UncheckedIOException(e);}catch(InterruptedException e){Thread.currentThread().interrupt();failure=new IllegalStateException(e);}
    }
    private void write(List<Ready> games)throws IOException {
        String base=String.format("part-%d-%d-%05d.traj",System.currentTimeMillis(),ProcessHandle.current().pid(),++serial);
        Path target=dir.resolve(base),tmp=dir.resolve(base+".tmp");int steps=0;
        try(DataOutputStream out=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp),65536))){
            out.write(magic().getBytes(StandardCharsets.US_ASCII));out.writeByte(level==0?0:1);
            out.write(HexFormat.of().parseHex(tableSha));
            for(Ready r:games){out.writeInt(r.metadata.length);out.writeLong(r.rawBytes);out.writeLong(r.data.length);out.writeInt(r.checksum);out.write(r.metadata);out.write(r.data);steps+=r.steps;}
        }
        Files.writeString(dir.resolve(base+".meta.json"),"{\"schema_version\":"+schema+",\"format\":\""+magic()+"\",\"compression_level\":"+level+",\"steps\":"+steps+",\"game_sides\":"+games.size()+",\"bytes\":"+Files.size(tmp)+"}");
        Files.move(tmp,target,StandardCopyOption.ATOMIC_MOVE);
    }
    private String magic(){return "SWTRJ003";}
}
