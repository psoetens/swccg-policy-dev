package com.gempukku.swccgo.bench;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Check of {@link PolicySampler} against swccg-policy's rl.sample_sequence (driven by scripts/sampler_check.py).
 * Input JSON: {"cases": [{"logits": [...], "none": [0/1...], "k": int, "t": float, "x": [...], "seq": [...]}...],
 *              "dist": [{"logits": ..., "none": ..., "k": 1, "t": float, "x": [...]}...], "draws": N}
 * Output JSON: {"logp": [per case: Java log-probability of the Python-sampled seq],
 *               "counts": [per dist case: counts of the first pick over N Java samples]}
 *   SamplerCheck in.json out.json
 */
public final class SamplerCheck {
    public static void main(String[] args) throws Exception {
        JsonObject in = JsonParser.parseReader(new FileReader(args[0])).getAsJsonObject();
        JsonObject out = new JsonObject();
        JsonArray lps = new JsonArray();
        for (JsonElement c : in.getAsJsonArray("cases")) {
            JsonObject o = c.getAsJsonObject();
            float[] lg = floats(o.getAsJsonArray("logits"));
            List<Integer> seq = new ArrayList<>();
            for (JsonElement e : o.getAsJsonArray("seq")) seq.add(e.getAsInt());
            lps.add(PolicySampler.logProb(lg, lg.length, bools(o.getAsJsonArray("none")), o.get("k").getAsInt(), o.get("t").getAsDouble(),
                    ints(o.getAsJsonArray("x")), seq));
        }
        out.add("logp", lps);
        int draws = in.get("draws").getAsInt();
        SplittableRandom rng = new SplittableRandom(7);
        JsonArray counts = new JsonArray();
        for (JsonElement c : in.getAsJsonArray("dist")) {
            JsonObject o = c.getAsJsonObject();
            float[] lg = floats(o.getAsJsonArray("logits"));
            boolean[] none = bools(o.getAsJsonArray("none"));
            int[] x = ints(o.getAsJsonArray("x"));
            long[] n = new long[lg.length];
            for (int d = 0; d < draws; d++)
                n[PolicySampler.sample(lg, lg.length, none, o.get("k").getAsInt(), o.get("t").getAsDouble(), x, rng).seq.get(0)]++;
            JsonArray a = new JsonArray();
            for (long v : n) a.add(v);
            counts.add(a);
        }
        out.add("counts", counts);
        try (PrintWriter w = new PrintWriter(new FileWriter(args[1]))) {
            w.print(out);
        }
    }

    private static float[] floats(JsonArray a) {
        float[] f = new float[a.size()];
        for (int i = 0; i < f.length; i++) f[i] = a.get(i).getAsFloat();
        return f;
    }

    private static boolean[] bools(JsonArray a) {
        boolean[] b = new boolean[a.size()];
        for (int i = 0; i < b.length; i++) b[i] = a.get(i).getAsInt() != 0;
        return b;
    }

    private static int[] ints(JsonArray a) {
        int[] v = new int[a.size()];
        for (int i = 0; i < v.length; i++) v[i] = a.get(i).getAsInt();
        return v;
    }
}
