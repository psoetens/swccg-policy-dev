package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.knowledge.ActorTensors;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * ABI 4 has int32 tokens and a seventh uint8 grid input. Old tensor paths stay independent. Plan brainiac-v6: the
 * conversion is {@link ActorTensors} (shared with production Brainiac); this class adds the native batch packing.
 */
public final class PolicyTensorsV4 {
    static void validateMaps(PolicyTensors.Maps maps, JsonObject vocabulary, Path directory) throws IOException {
        maps.actorTables = new ActorTensors.Tables(maps.cards, maps.q, maps.o, maps.context, maps.contextVersion);
        ActorTensors.validate(maps.actorTables, vocabulary, directory);
        maps.tensorAbi = maps.actorTables.tensorAbi;
        maps.tableSha = maps.actorTables.tableSha;
    }

    static int card(String blueprint, PolicyTensors.Maps maps) {
        return ActorTensors.card(blueprint, tables(maps));
    }

    public static PolicyTensors convert(List<ObservationV2.Token> rows, List<int[]> grid, List<Object[]> options,
            String type, String question, int[] globals, int side, PolicyTensors.Maps maps, boolean privileged) {
        return new PolicyTensors(ActorTensors.convert(rows, grid, options, type, question, globals, side,
            tables(maps), privileged));
    }

    /** Plan context-overhaul: see {@link ActorTensors#contextRows}. */
    public static List<ObservationV2.Token> contextRows(List<ObservationV2.Token> rows, int version) {
        return ActorTensors.contextRows(rows, version);
    }

    public static PolicyTensors fromRecord(JsonArray record, PolicyTensors.Maps maps, boolean privileged) {
        return new PolicyTensors(ActorTensors.fromRecord(record, tables(maps), privileged));
    }

    public static void validate(PolicyTensors tensors, PolicyTensors.Maps maps, boolean actor) {
        ActorTensors.validate(new ActorTensors.Tensors(tensors.tokens, tensors.options, tensors.floatWidth,
            tensors.tok32, tensors.tokf, tensors.opt, tensors.head, tensors.isNone, tensors.kgrid, tensors.gridRows,
            tensors.tableSha, tensors.privileged), tables(maps), actor);
    }

    /** Plan brainiac-v6: the conversion itself lives in the shared {@link ActorTensors} (production Brainiac uses it). */
    private static ActorTensors.Tables tables(PolicyTensors.Maps maps) {
        if (maps.schemaVersion != 4 || maps.tableSha == null || maps.actorTables == null) {
            throw new IllegalArgumentException("ABI-4 own-table bundle required");
        }
        return maps.actorTables;
    }

    static int[] shape(List<PolicyTensors> items) {
        int tokens = 1;
        int options = 1;
        int grids = 1;
        int width = items.get(0).floatWidth;
        for (PolicyTensors item : items) {
            if (item.tok32 == null || item.privileged || item.floatWidth != width) {
                throw new IllegalArgumentException("ABI-4/5 actor batches require actor tensors of one float width");
            }
            tokens = Math.max(tokens, item.tokens);
            options = Math.max(options, item.options);
            grids = Math.max(grids, item.gridRows);
        }
        int batch = 1;
        while (batch < items.size()) {
            batch *= 2;
        }
        int paddedTokens = (tokens + 127) / 128 * 128;
        int paddedOptions = (options + 7) / 8 * 8;
        if (batch > 64 || paddedTokens > Integer.getInteger("policy.native.maxGraphTokens", 320)
                || paddedOptions > 64) {
            return new int[]{items.size(), tokens, options, grids, width};
        }
        // Fifth element: floats per token (4, or 8 under tensor ABI 5).
        return new int[]{batch, paddedTokens, paddedOptions, (grids + 15) / 16 * 16, width};
    }

    private static int[] sizes(int[] shape) {
        int batch = shape[0];
        int tokens = shape[1];
        int options = shape[2];
        return new int[]{batch * tokens * 32, batch * tokens * 4 * shape[4], batch * tokens,
            batch * options * 24, batch * options, batch * 88, batch * shape[3] * 153};
    }

    static int packedBytes(int[] shape) {
        return Arrays.stream(sizes(shape)).map(size -> (size + 15) / 16 * 16).sum();
    }

    static void pack(List<PolicyTensors> items, int[] shape, ByteBuffer buffer) {
        int[] sizes = sizes(shape);
        int[] offsets = new int[7];
        for (int index = 1; index < 7; index++) {
            offsets[index] = offsets[index - 1] + (sizes[index - 1] + 15) / 16 * 16;
        }
        int total = packedBytes(shape);
        buffer.clear().order(ByteOrder.LITTLE_ENDIAN);
        for (int index = 0; index < total; index++) {
            buffer.put(index, (byte) 0);
        }
        for (int row = 0; row < items.size(); row++) {
            PolicyTensors item = items.get(row);
            if (item.privileged || item.tok32 == null || item.tokens > shape[1] || item.options > shape[2]
                    || item.gridRows > shape[3] || item.floatWidth != shape[4]) {
                throw new IllegalArgumentException("Invalid ABI-4 actor pack shape/view");
            }
            for (int index = 0; index < item.tok32.length; index++) {
                buffer.putInt(offsets[0] + 4 * (row * shape[1] * 8 + index), item.tok32[index]);
            }
            for (int index = 0; index < item.tokf.length; index++) {
                buffer.putFloat(offsets[1] + 4 * (row * shape[1] * shape[4] + index), item.tokf[index]);
            }
            for (int index = 0; index < item.tokens; index++) {
                buffer.put(offsets[2] + row * shape[1] + index, (byte) 1);
            }
            for (int index = 0; index < item.opt.length; index++) {
                buffer.putInt(offsets[3] + 4 * (row * shape[2] * 6 + index), item.opt[index]);
            }
            for (int index = 0; index < item.options; index++) {
                buffer.put(offsets[4] + row * shape[2] + index, (byte) 1);
            }
            for (int index = 0; index < 22; index++) {
                buffer.putInt(offsets[5] + 4 * (row * 22 + index), item.head[index]);
            }
            for (int index = 0; index < item.kgrid.length; index++) {
                buffer.put(offsets[6] + row * shape[3] * 153 + index, item.kgrid[index]);
            }
        }
        buffer.limit(total);
    }
}
