package com.gempukku.swccgo.bench;

import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Equality check of the in-JVM policy path (driven by swccg-policy scripts/native_check.py): record lines -> PolicyTensors ->
 * packed bucket batches of size B -> NativePolicy.forward; writes each record's logits (first `options` entries) as one JSON
 * array per line.
 *   NativeCheck MODEL.ts BUNDLE_DIR records.txt B out.jsonl [tf32 true|false]
 */
public final class NativeCheck {
    public static void main(String[] args) throws Exception {
        NativePolicy.ensureLoaded();
        NativePolicy.setTf32(args.length > 5 && Boolean.parseBoolean(args[5]));
        long h = NativePolicy.load(args[0], true);
        if (h == 0) throw new IllegalStateException(NativePolicy.lastError());
        PolicyTensors.Maps maps = PolicyTensors.Maps.load(Paths.get(args[1]));
        List<String> lines = Files.readAllLines(Paths.get(args[2]));
        int B = Integer.parseInt(args[3]);
        ByteBuffer in = ByteBuffer.allocateDirect(PolicyTensors.packedBytes(256, 400, 512)).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer out = ByteBuffer.allocateDirect(4 * 256 * 512).order(ByteOrder.nativeOrder()).asFloatBuffer();
        try (PrintWriter w = new PrintWriter(args[4])) {
            for (int s = 0; s < lines.size(); s += B) {
                List<PolicyTensors> ts = new ArrayList<>();
                for (int i = s; i < Math.min(lines.size(), s + B); i++) ts.add(PolicyTensors.fromRecord(lines.get(i), maps));
                int[] sh = PolicyTensors.shape(ts);
                PolicyTensors.pack(ts, sh[0], sh[1], sh[2], in);
                if (NativePolicy.forward(h, in, sh[0], sh[1], sh[2], out) != 0) throw new IllegalStateException(NativePolicy.lastError());
                float[] lg = new float[sh[0] * sh[2]];
                out.position(0);
                out.get(lg);
                for (int i = 0; i < ts.size(); i++) {
                    StringBuilder sb = new StringBuilder("[");
                    for (int j = 0; j < ts.get(i).options; j++) sb.append(j == 0 ? "" : ",").append(lg[i * sh[2] + j]);
                    w.println(sb.append(']'));
                }
            }
        }
        NativePolicy.free(h);
    }
}
