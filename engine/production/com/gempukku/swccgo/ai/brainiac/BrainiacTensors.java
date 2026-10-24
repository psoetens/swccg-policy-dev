package com.gempukku.swccgo.ai.brainiac;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

/**
 * The policy network's input for one decision, built from a {@link StateEncoder} record. This is the same conversion the
 * training code uses (swccg-policy: data.parse_record + rl.compact), so the browser only has to run the network:
 * <ul>
 *   <li>tok  [T, 8] int16: kind, card index, zone, mine, location slot + 1, attached-to token + 1, count, flags</li>
 *   <li>tokf [T, 4] float32 (float16 precision, NaN = not applicable): power, ability, forfeit, deploy cost</li>
 *   <li>opt  [O, 6] int32: option template, first text card index, source token + 1, card index, value + 1, is none</li>
 *   <li>head [22] int32: decision type, question template, question card index, side, then 18 global features</li>
 * </ul>
 */
public final class BrainiacTensors {
    public static final int MAX_TOKENS = 320;
    public static final int N_GLOBAL = 18;
    public static final int HEAD = 4 + N_GLOBAL;
    private static final String[] DTYPES = {"EMPTY", "INTEGER", "MULTIPLE_CHOICE", "ARBITRARY_CARDS", "CARD_ACTION_CHOICE",
            "ACTION_CHOICE", "CARD_SELECTION"};

    public final int tokens;
    public final int options;
    public final short[] tok;
    public final float[] tokf;
    public final int[] opt;
    public final int[] head = new int[HEAD];

    public int[] tok32;
    public byte[] kgrid;
    public int gridRows;
    public String tableSha;
    public boolean privileged;
    public final boolean[] isNone;

    static final class Maps {
        final Map<String, Integer> cards, q, o, context;
        final int schemaVersion = 4;
        String tableSha;

        Maps(Map<String, Integer> cards, Map<String, Integer> q, Map<String, Integer> o,
                Map<String, Integer> context) {
            this.cards = cards;
            this.q = q;
            this.o = o;
            this.context = context;
        }
    }

    BrainiacTensors(int tokens, int options) {
        this.tokens = tokens;
        this.options = options;
        isNone = new boolean[options];
        tok = new short[tokens * 8];
        tokf = new float[tokens * 4];
        opt = new int[options * 6];
    }

    /**
     * Little-endian binary form for the browser: int32 tokens, int32 options, int32[22] head, int16[T*8] tok,
     * float32[T*4] tokf, int32[O*6] opt. Every section starts 4-byte aligned (T*8 int16 values are a multiple of 4 bytes).
     */
    public byte[] toBytes() {
        if (tok32 != null) {
            return bytesV4();
        }
        ByteBuffer b = ByteBuffer.allocate(4 * (2 + HEAD) + 2 * tok.length + 4 * tokf.length + 4 * opt.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(tokens).putInt(options);
        for (int v : head) b.putInt(v);
        for (short v : tok) b.putShort(v);
        for (float v : tokf) b.putFloat(v);
        for (int v : opt) b.putInt(v);
        return b.array();
    }

    private byte[] bytesV4() {
        if (privileged) {
            throw new IllegalArgumentException("Privileged tensors rejected by browser actor path");
        }
        ByteBuffer buffer = ByteBuffer.allocate(4 * (4 + HEAD) + 4 * tok32.length
            + 4 * tokf.length + 4 * opt.length + kgrid.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(0x34545753).putInt(tokens).putInt(options).putInt(gridRows);
        for (int value : head) {
            buffer.putInt(value);
        }
        for (int value : tok32) {
            buffer.putInt(value);
        }
        for (float value : tokf) {
            buffer.putFloat(value);
        }
        for (int value : opt) {
            buffer.putInt(value);
        }
        buffer.put(kgrid);
        return buffer.array();
    }
}
