package com.gempukku.swccgo.bench;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The policy network's input for one decision, built from a StateEncoder record line: the same conversion as swccg-policy's
 * data.parse_record + rl.compact (adapted from Brainiac's BrainiacTensors), and the batch packing of rl.GraphForward
 * (bucket shapes and padding) for {@link NativePolicy#forward}.
 * <ul>
 *   <li>tok [T, 8] int16: kind, card index, zone, mine, location slot + 1, attached-to token + 1, count (max 127), flags</li>
 *   <li>tokf [T, 4] float32 at float16 precision (NaN = not applicable): power, ability, forfeit, deploy cost</li>
 *   <li>opt [O, 6] int32: option template, first text card, source token + 1 (0 beyond the kept tokens), card, value + 1 (max 32),
 *       none flag</li>
 *   <li>head [22] int32: decision type, question template, question card, side, 18 global features</li>
 * </ul>
 */
public final class PolicyTensors {
    public static final int MAX_TOKENS = 320;
    static final int[] B_BUCKETS = {1, 2, 4, 8, 16, 32, 64};
    static final int T_STEP = Boolean.getBoolean("policy.knowledge")?128:16, O_STEP = 8, O_MAX = 64;
    private static final String[] DTYPES = {"EMPTY", "INTEGER", "MULTIPLE_CHOICE", "ARBITRARY_CARDS", "CARD_ACTION_CHOICE",
            "ACTION_CHOICE", "CARD_SELECTION"};

    /** Card index and template vocabularies (swccg_policy.native bundle: cards.json, vocab.json). */
    public static final class Maps {
        final Map<String, Integer> cards, q, o;
        int schemaVersion=1;
        /** Plan objective-dojo Amendment 2: ABI 5 = schema-4 tensors with eight floats per token (bundle.json tensor_abi). */
        int tensorAbi=4;
        Map<String,Integer> context=Map.of();
        /** Plan context-overhaul: the context version of this vocabulary (vocab.json context_version, default 4). */
        int contextVersion=4;
        String tableSha;
        /** Plan brainiac-v6: the schema-4 view of these maps for the shared ActorTensors conversion. */
        com.gempukku.swccgo.game.knowledge.ActorTensors.Tables actorTables;

        Maps(Map<String, Integer> cards, Map<String, Integer> q, Map<String, Integer> o) {
            this.cards = cards;
            this.q = q;
            this.o = o;
        }

        public static Maps load(Path bundle) {
            try {
                JsonObject v = JsonParser.parseString(Files.readString(bundle.resolve("vocab.json"))).getAsJsonObject();
                JsonObject c = JsonParser.parseString(Files.readString(bundle.resolve("cards.json"))).getAsJsonObject();
                Maps maps=new Maps(map(c), map(v.getAsJsonObject("q")), map(v.getAsJsonObject("o")));
                maps.schemaVersion=v.has("schema_version")?v.get("schema_version").getAsInt():1;
                maps.context=v.has("context")?map(v.getAsJsonObject("context")):Map.of();
                maps.contextVersion=v.has("context_version")?v.get("context_version").getAsInt():4;
                if (maps.schemaVersion == 4) {
                    PolicyTensorsV4.validateMaps(maps, v, bundle);
                }
                return maps;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        private static Map<String, Integer> map(JsonObject o) {
            Map<String, Integer> m = new HashMap<>(o.size() * 2);
            for (Map.Entry<String, JsonElement> e : o.entrySet()) m.put(e.getKey(), e.getValue().getAsInt());
            return m;
        }
    }

    public final int tokens, options;
    public final short[] tok;
    public final float[] tokf;
    /** Floats per token: 4, or 8 under tensor ABI 5 (plan objective-dojo Amendment 2). */
    public final int floatWidth;
    public final int[] opt;
    public final int[] head = new int[22];
    public final boolean[] isNone;
    public int[] tok32;
    public byte[] kgrid;
    public int gridRows;
    public String tableSha;
    public boolean privileged;

    PolicyTensors(int tokens, int options) {
        this(tokens, options, 4);
    }

    PolicyTensors(int tokens, int options, int floatWidth) {
        this.tokens = tokens;
        this.options = options;
        this.floatWidth = floatWidth;
        tok = new short[tokens * 8];
        tokf = new float[tokens * floatWidth];
        opt = new int[options * 6];
        isNone = new boolean[options];
    }

    /** Plan brainiac-v6: adopts the shared ActorTensors conversion's arrays (no copy). */
    PolicyTensors(com.gempukku.swccgo.game.knowledge.ActorTensors.Tensors tensors) {
        this.tokens = tensors.tokens;
        this.options = tensors.options;
        this.floatWidth = tensors.floatWidth;
        tok = new short[tokens * 8];
        tokf = tensors.tokf;
        opt = tensors.opt;
        isNone = tensors.isNone;
        System.arraycopy(tensors.head, 0, head, 0, 22);
        tok32 = tensors.tok32;
        kgrid = tensors.kgrid;
        gridRows = tensors.gridRows;
        tableSha = tensors.tableSha;
        privileged = tensors.privileged;
    }

    /** Converts one record line (StateEncoder layout, version 1). */
    public static PolicyTensors fromRecord(String line, Maps m) {
        JsonArray r = JsonParser.parseString(line).getAsJsonArray();
        int version=r.get(0).getAsInt();
        if (version == 4) {
            return PolicyTensorsV4.fromRecord(r, m, false);
        }
        // Plan remove-old-generations: version 1 (V1) here, version 4 (V3.1) above; versions 2 / 3 were removed.
        if(version!=m.schemaVersion || version!=1) throw new IllegalArgumentException("observation/bundle schema mismatch");
        JsonArray toks = r.get(8).getAsJsonArray();
        JsonArray opts = r.get(9).getAsJsonArray();
        int nt = Math.min(toks.size(), MAX_TOKENS);
        PolicyTensors t = new PolicyTensors(nt, opts.size());
        for (int i = 0; i < nt; i++) {
            JsonArray k = toks.get(i).getAsJsonArray();
            int b = i * 8;
            t.tok[b] = (short) k.get(0).getAsInt();
            t.tok[b + 1] = (short) card(k.get(1), m.cards);
            t.tok[b + 2] = (short) k.get(2).getAsInt();
            t.tok[b + 3] = (short) k.get(3).getAsInt();
            t.tok[b + 4] = (short) (k.get(4).getAsInt() + 1);
            t.tok[b + 5] = (short) (k.get(5).getAsInt() + 1); // clamped to the batch's real token count in pack(), as rl.stack
            t.tok[b + 6] = (short) Math.min(k.get(6).getAsInt(), 127);
            t.tok[b + 7] = (short) k.get(7).getAsInt();
            for (int j = 0; j < 4; j++) t.tokf[i * 4 + j] = half(k.get(8 + j));
        }
        for (int i = 0; i < opts.size(); i++) {
            JsonArray o = opts.get(i).getAsJsonArray();
            int b = i * 6;
            String template = o.get(0).getAsString();
            JsonArray textCards = o.get(1).getAsJsonArray();
            int src = o.get(2).getAsInt() + 1;
            t.opt[b] = com.gempukku.swccgo.game.knowledge.TemplateCoverage.lookup(m.o,"o",template);
            t.opt[b + 1] = textCards.isEmpty() ? 0 : card(textCards.get(0), m.cards);
            t.opt[b + 2] = src > nt ? 0 : src;
            t.opt[b + 3] = isEmpty(o.get(3)) ? 0 : card(o.get(3), m.cards);
            t.opt[b + 4] = o.get(5).isJsonNull() ? 0 : Math.min(o.get(5).getAsInt() + 1, 32);
            t.isNone[i] = template.equals("<pass>") || template.equals("<none>");
            t.opt[b + 5] = t.isNone[i] ? 1 : 0;
        }
        String dtype = r.get(4).getAsString();
        JsonArray qbps = r.get(6).getAsJsonArray();
        JsonArray glob = r.get(7).getAsJsonArray();
        for (int i = 0; i < DTYPES.length; i++)
            if (DTYPES[i].equals(dtype)) t.head[0] = i;
        t.head[1] = com.gempukku.swccgo.game.knowledge.TemplateCoverage.lookup(m.q,"q",r.get(5).getAsString());
        t.head[2] = qbps.isEmpty() ? 0 : card(qbps.get(0), m.cards);
        t.head[3] = "DS".equals(r.get(3).getAsString()) ? 0 : 1;
        for (int i = 0; i < 18; i++) t.head[4 + i] = glob.get(i).getAsInt();
        return t;
    }

    /** Card index of a blueprint (1 = unknown card), as data.parse_record's bp_idx.get(bp, 1). */
    private static int card(JsonElement bp, Map<String, Integer> cards) {
        return bp.isJsonNull() ? 1 : cards.getOrDefault(bp.getAsString(), 1);
    }

    private static boolean isEmpty(JsonElement e) {
        return e.isJsonNull() || e.getAsString().isEmpty();
    }

    /** A token stat rounded to float16, as rl.compact stores it; null becomes NaN. */
    private static float half(JsonElement e) {
        if (e.isJsonNull()) return Float.NaN;
        return Float.float16ToFloat(Float.floatToFloat16(e.getAsFloat()));
    }

    /**
     * Shape the batch runs with, as rl.GraphForward.bucket (graph buckets) or rl.stack (larger batches: unpadded maxima):
     * {B, T, O}.
     */
    public static int[] shape(List<PolicyTensors> items) {
        if (items.get(0).tok32 != null) {
            return PolicyTensorsV4.shape(items);
        }
        int n = items.size(), T = 1, O = 1;
        for (PolicyTensors it : items) {
            T = Math.max(T, it.tokens);
            O = Math.max(O, it.options);
        }
        int Bb = -1;
        for (int b : B_BUCKETS)
            if (b >= n) {
                Bb = b;
                break;
            }
        int Tb = (T + T_STEP - 1) / T_STEP * T_STEP, Ob = (O + O_STEP - 1) / O_STEP * O_STEP;
        if (Bb < 0 || Tb > Integer.getInteger("policy.native.maxGraphTokens",MAX_TOKENS) || Ob > O_MAX) return new int[]{n, T, O};
        return new int[]{Bb, Tb, Ob};
    }

    /** Bytes of the packed layout for shape (B, T, O): six fields, each starting at a 16-byte boundary. */
    public static int packedBytes(int B, int T, int O) {
        int[] n = sizes(B, T, O);
        int total = 0;
        for (int x : n) total += (x + 15) / 16 * 16;
        return total;
    }

    public static int packedBytes(int[] shape) {
        return shape.length == 5 ? PolicyTensorsV4.packedBytes(shape) : packedBytes(shape[0], shape[1], shape[2]);
    }

    public static void pack(List<PolicyTensors> items, int[] shape, ByteBuffer buffer) {
        if (shape.length == 5) {
            PolicyTensorsV4.pack(items, shape, buffer);
        } else {
            pack(items, shape[0], shape[1], shape[2], buffer);
        }
    }

    private static int[] sizes(int B, int T, int O) {
        return new int[]{B * T * 8 * 2, B * T * 4 * 4, B * T, B * O * 6 * 4, B * O, B * 22 * 4};
    }

    /**
     * Writes the batch into buf (little endian) in the layout of rl.GraphForward._fill: tok, tokf (NaN padding), tmask, opt,
     * omask, head; parent-token indices beyond the batch's real token count become 0 (rl.stack).
     */
    public static void pack(List<PolicyTensors> items, int B, int T, int O, ByteBuffer buf) {
        buf.clear();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        int[] n = sizes(B, T, O);
        int[] off = new int[6];
        for (int i = 1; i < 6; i++) off[i] = off[i - 1] + (n[i - 1] + 15) / 16 * 16;
        int realT = 1;
        for (PolicyTensors it : items) realT = Math.max(realT, it.tokens);
        int total = off[5] + (n[5] + 15) / 16 * 16;
        for (int i = 0; i < total; i++) buf.put(i, (byte) 0);
        for (int i = 0; i < B * T * 4; i++) buf.putFloat(off[1] + 4 * i, Float.NaN);
        for (int r = 0; r < items.size(); r++) {
            PolicyTensors it = items.get(r);
            for (int t = 0; t < it.tokens; t++) {
                int base = (r * T + t) * 8;
                for (int j = 0; j < 8; j++) {
                    short v = it.tok[t * 8 + j];
                    if (j == 5 && v > realT) v = 0;
                    buf.putShort(off[0] + 2 * (base + j), v);
                }
                for (int j = 0; j < 4; j++) buf.putFloat(off[1] + 4 * ((r * T + t) * 4 + j), it.tokf[t * 4 + j]);
                buf.put(off[2] + r * T + t, (byte) 1);
            }
            for (int o = 0; o < it.options; o++) {
                for (int j = 0; j < 6; j++) buf.putInt(off[3] + 4 * ((r * O + o) * 6 + j), it.opt[o * 6 + j]);
                buf.put(off[4] + r * O + o, (byte) 1);
            }
            for (int j = 0; j < 22; j++) buf.putInt(off[5] + 4 * (r * 22 + j), it.head[j]);
        }
        buf.limit(total);
    }
}
