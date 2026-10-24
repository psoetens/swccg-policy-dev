package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.game.knowledge.KnowledgeGrid;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import com.gempukku.swccgo.game.knowledge.VocabularyV4;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/** ABI 4 has int32 tokens and a seventh uint8 grid input. Old tensor paths stay independent. */
public final class BrainiacTensorsV4 {
    private static final List<String> TYPES = List.of("EMPTY", "INTEGER", "MULTIPLE_CHOICE", "ARBITRARY_CARDS",
        "CARD_ACTION_CHOICE", "ACTION_CHOICE", "CARD_SELECTION");

    static void validateMaps(BrainiacTensors.Maps maps, JsonObject vocabulary, Path directory) throws IOException {
        String manifest = Files.readString(directory.resolve("bundle.json"));
        JsonObject bundle = JsonParser.parseString(manifest).getAsJsonObject();
        String sha = HexFormat.of().formatHex(VocabularyV4.fingerprint(vocabulary));
        if (vocabulary.get("tensor_abi").getAsInt() != 4 || vocabulary.get("table_format").getAsInt() != 2
                || vocabulary.get("oov_buckets").getAsInt() != 512
                || !vocabulary.get("normalizer_version").getAsString().equals(TitleMatcher.NORMALIZER_VERSION)
                || !vocabulary.get("oov_hash").getAsString().equals(VocabularyV4.HASH)
                || !sha.equals(vocabulary.get("sha").getAsString())
                || !sha.equals(bundle.get("vocabulary_sha").getAsString())) {
            throw new IllegalArgumentException("ABI-4 vocabulary contract/hash mismatch");
        }
        for (Map<String, Integer> family : List.of(maps.q, maps.o, maps.context)) {
            Set<Integer> ids = new HashSet<>(family.values());
            if (ids.size() != family.size()) {
                throw new IllegalArgumentException("Duplicate template IDs");
            }
            for (int index = 0; index < family.size(); index++) {
                if (!ids.contains(index) || index < 512 && !Integer.valueOf(index).equals(
                        family.get(String.format("<oov:%03d>", index)))) {
                    throw new IllegalArgumentException("Noncontiguous or changed OOV IDs");
                }
            }
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            new TreeMap<>(maps.cards).forEach((key, value) -> digest.update((key + "\t" + value + "\n")
                .getBytes(java.nio.charset.StandardCharsets.UTF_8)));
            if (!HexFormat.of().formatHex(digest.digest()).equals(bundle.get("card_map_sha").getAsString())) {
                throw new IllegalArgumentException("ABI-4 card-map hash mismatch");
            }
        } catch (java.security.NoSuchAlgorithmException exception) {
            throw new IllegalStateException(exception);
        }
        maps.tableSha = bundle.get("table_sha").getAsString();
        if (maps.tableSha.length() != 64 || HexFormat.of().parseHex(maps.tableSha).length != 32) {
            throw new IllegalArgumentException("Invalid table SHA");
        }
    }

    static int card(String blueprint, BrainiacTensors.Maps maps) {
        String stripped = ObservationV2.strip(blueprint);
        Integer index = maps.cards.get(stripped);
        if (index == null) {
            if (stripped.matches("[0-9]+_.*") && Integer.parseInt(stripped.split("_", 2)[0]) >= 200) {
                throw new IllegalArgumentException("Virtual blueprint requires explicit table growth: " + blueprint);
            }
            throw new IllegalArgumentException("Blueprint absent from this card table: " + blueprint);
        }
        return index;
    }

    public static BrainiacTensors convert(List<ObservationV2.Token> rows, List<int[]> grid, List<Object[]> options,
            String type, String question, int[] globals, int side, BrainiacTensors.Maps maps, boolean privileged) {
        if (maps.schemaVersion != 4 || maps.tableSha == null) {
            throw new IllegalArgumentException("ABI-4 own-table bundle required");
        }
        BrainiacTensors result = new BrainiacTensors(rows.size(), options.size());
        result.tok32 = new int[rows.size() * 8];
        result.kgrid = new byte[grid.size() * KnowledgeGrid.WIDTH];
        result.gridRows = grid.size();
        result.tableSha = maps.tableSha;
        result.privileged = privileged;
        for (int index = 0; index < rows.size(); index++) {
            ObservationV2.Token row = rows.get(index);
            int slot = row.kind == 7 ? VocabularyV4.lookup(maps.context, "context", row.contextKey) : row.slot + 1;
            int[] values = {row.kind, card(row.bp, maps), row.zone, row.mine, slot,
                row.parent + 1, row.count, row.flags};
            System.arraycopy(values, 0, result.tok32, index * 8, 8);
            float[] floats = {row.a, row.b, row.c, row.d};
            System.arraycopy(floats, 0, result.tokf, index * 4, 4);
            if (!privileged && row.kind == 7 && row.contextKey.startsWith("rules.")) {
                throw new IllegalArgumentException("Privileged context rejected by actor path");
            }
        }
        for (int index = 0; index < grid.size(); index++) {
            int[] row = grid.get(index);
            if (row.length != KnowledgeGrid.WIDTH) {
                throw new IllegalArgumentException("Invalid grid width");
            }
            for (int column = 0; column < row.length; column++) {
                if (row[column] < 0 || row[column] > 255) {
                    throw new IllegalArgumentException("Grid uint8 overflow");
                }
                result.kgrid[index * KnowledgeGrid.WIDTH + column] = (byte) row[column];
            }
        }
        for (int index = 0; index < options.size(); index++) {
            Object[] option = options.get(index);
            String template = (String) option[0];
            String blueprint = (String) option[3];
            int blueprintId = blueprint == null || blueprint.isEmpty() ? 0
                : blueprint.equals("inPlay") || blueprint.equals("rules") ? 1 : card(blueprint, maps);
            result.isNone[index] = template.equals("<none>") || template.equals("<pass>");
            int[] values = {VocabularyV4.lookup(maps.o, "o", template), 0, (Integer) option[2] + 1,
                blueprintId, option[4] == null ? 0 : Math.addExact((Integer) option[4], 1),
                result.isNone[index] ? 1 : 0};
            System.arraycopy(values, 0, result.opt, index * 6, 6);
        }
        result.head[0] = TYPES.indexOf(type);
        result.head[1] = VocabularyV4.lookup(maps.q, "q", question);
        result.head[3] = side;
        if (globals.length != 18) {
            throw new IllegalArgumentException("Invalid global width");
        }
        System.arraycopy(globals, 0, result.head, 4, 18);
        validate(result, maps, !privileged);
        return result;
    }

    public static BrainiacTensors fromRecord(JsonArray record, BrainiacTensors.Maps maps, boolean privileged) {
        if (record.get(0).getAsInt() != 4 || record.size() != 12 || privileged) {
            throw new IllegalArgumentException("Final schema-4 actor/critic record required");
        }
        List<ObservationV2.Token> rows = new ArrayList<>();
        for (JsonElement element : record.get(privileged ? 12 : 8).getAsJsonArray()) {
            JsonArray values = element.getAsJsonArray();
            ObservationV2.Token row = new ObservationV2.Token();
            row.kind = values.get(0).getAsInt();
            row.bp = values.get(1).getAsString();
            row.zone = values.get(2).getAsInt();
            row.mine = values.get(3).getAsInt();
            if (row.kind == 7) {
                row.contextKey = values.get(4).getAsString();
            } else {
                row.slot = values.get(4).getAsInt();
            }
            row.parent = values.get(5).getAsInt();
            row.count = values.get(6).getAsInt();
            row.flags = values.get(7).getAsInt();
            row.a = values.get(8).getAsFloat();
            row.b = values.get(9).getAsFloat();
            row.c = values.get(10).getAsFloat();
            row.d = values.get(11).getAsFloat();
            rows.add(row);
        }
        List<int[]> grid = new ArrayList<>();
        for (JsonElement element : record.get(privileged ? 14 : 11).getAsJsonArray()) {
            grid.add(ints(element.getAsJsonArray()));
        }
        List<Object[]> options = new ArrayList<>();
        if (!privileged) {
            for (JsonElement element : record.get(9).getAsJsonArray()) {
                JsonArray option = element.getAsJsonArray();
                options.add(new Object[]{option.get(0).getAsString(), List.of(), option.get(2).getAsInt(),
                    option.get(3).isJsonNull() ? null : option.get(3).getAsString(),
                    option.get(5).isJsonNull() ? null : option.get(5).getAsInt()});
            }
        }
        return convert(rows, grid, options, record.get(4).getAsString(), record.get(5).getAsString(),
            ints(record.get(7).getAsJsonArray()), record.get(3).getAsString().equals("DS") ? 0 : 1, maps, privileged);
    }

    private static int[] ints(JsonArray values) {
        int[] result = new int[values.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = values.get(index).getAsInt();
        }
        return result;
    }

    public static void validate(BrainiacTensors tensors, BrainiacTensors.Maps maps, boolean actor) {
        if (actor && tensors.privileged || !maps.tableSha.equals(tensors.tableSha)) {
            throw new IllegalArgumentException("Privileged payload or table mismatch in actor path");
        }
        if (tensors.tok32 == null || tensors.tok32.length != 8 * tensors.tokens
                || tensors.tokf.length != 4 * tensors.tokens || tensors.opt.length != 6 * tensors.options
                || tensors.kgrid.length != 153 * tensors.gridRows || tensors.head[0] < 0 || tensors.head[0] >= 7
                || tensors.head[1] < 0 || tensors.head[1] >= maps.q.size() || tensors.head[2] != 0
                || tensors.head[3] < 0 || tensors.head[3] > 1) {
            throw new IllegalArgumentException("Invalid ABI-4 tensor shapes/head");
        }
        Set<String> identities = new HashSet<>();
        boolean[] used = new boolean[tensors.gridRows];
        for (int index = 0; index < tensors.tokens; index++) {
            int start = index * 8;
            int kind = tensors.tok32[start];
            int slot = tensors.tok32[start + 4];
            int parent = tensors.tok32[start + 5];
            int flags = tensors.tok32[start + 7];
            float endpoint = tensors.tokf[index * 4];
            if (kind < 0 || kind > 10 || kind == 4 || kind == 6 || tensors.tok32[start + 2] < 0
                    || tensors.tok32[start + 2] >= 64 || tensors.tok32[start + 3] < 0
                    || tensors.tok32[start + 3] > 1 || tensors.tok32[start + 1] < 0
                    || tensors.tok32[start + 1] >= maps.cards.size()) {
                throw new IllegalArgumentException("Invalid ABI-4 token");
            }
            for (int column = 0; column < 4; column++) {
                if (!Float.isFinite(tensors.tokf[index * 4 + column])) {
                    throw new IllegalArgumentException("Nonfinite ABI-4 float");
                }
            }
            if (kind == 8 && (parent < 1 || parent > tensors.tokens || endpoint < 1
                    || endpoint > tensors.tokens || endpoint != (int) endpoint || slot < 1 || slot > 14)) {
                throw new IllegalArgumentException("Invalid physical edge");
            }
            if (kind == 7 && (slot < 0 || slot >= maps.context.size() || parent < 0 || parent > tensors.tokens
                    || flags < 0 || flags > 127)) {
                throw new IllegalArgumentException("Invalid inline context edge/key");
            }
            if (kind == 7) {
                float[] indices={endpoint,tensors.tokf[index*4+2],tensors.tokf[index*4+3],tensors.tok32[start+6]};
                boolean absent=false;
                for(float value:indices) {
                    if(value < -1 || value != (int)value || absent && value != -1)
                        throw new IllegalArgumentException("Invalid ordered context index tuple");
                    absent |= value == -1;
                }
            }
            if ((kind == 9 || kind == 10) && (slot < 1 || endpoint != (int) endpoint
                    || kind == 9 && endpoint != 0 || kind == 10 && (endpoint < 1 || endpoint > tensors.options))) {
                throw new IllegalArgumentException("Invalid text reference route");
            }
            if (actor && (kind == 3 || kind == 5) && (flags & 8) != 0) {
                throw new IllegalArgumentException("Privileged actor fact");
            }
            if ((kind == 3 || kind == 5) && (flags < 2 || flags > 15)) {
                throw new IllegalArgumentException("Invalid retained fact scope");
            }
            if (kind == 2) {
                if (tensors.tok32[start+2]!=0 || parent!=0 || tensors.tok32[start+6]!=0 || flags!=0
                        || endpoint!=0 || tensors.tokf[index*4+1]!=0 || tensors.tokf[index*4+2]!=0
                        || tensors.tokf[index*4+3]!=0) {
                    throw new IllegalArgumentException("Reserved grid fields must be zero");
                }
                if (slot < 1 || slot > used.length || used[slot - 1]
                        || !identities.add(tensors.tok32[start + 1] + ":" + tensors.tok32[start + 3])) {
                    throw new IllegalArgumentException("Duplicate or invalid grid binding");
                }
                used[slot - 1] = true;
            }
        }
        for (boolean present : used) {
            if (!present) {
                throw new IllegalArgumentException("Unbound grid row");
            }
        }
        for (int offset = 0; offset < tensors.kgrid.length; offset += 3) {
            int lower = tensors.kgrid[offset] & 255;
            int upper = tensors.kgrid[offset + 1] & 255;
            int asserted = tensors.kgrid[offset + 2] & 255;
            int cell = offset % KnowledgeGrid.WIDTH / 3;
            if (lower > upper || asserted > 1 || (cell / 3 == 6 || actor && cell % 3 == 2)
                    && (lower != 0 || upper != 0 || asserted != 0)) {
                throw new IllegalArgumentException("Invalid or privileged grid cell");
            }
        }
        for (int index = 0; index < tensors.options; index++) {
            int source = tensors.opt[index * 6 + 2];
            if (source < 0 || source > tensors.tokens || tensors.opt[index*6] < 0
                    || tensors.opt[index*6] >= maps.o.size() || tensors.opt[index*6+1] != 0
                    || tensors.opt[index*6+3] < 0 || tensors.opt[index*6+3] >= maps.cards.size()) {
                throw new IllegalArgumentException("Invalid option source");
            }
        }
    }

}
