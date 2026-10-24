package com.gempukku.swccgo.game.knowledge;

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
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Plan brainiac-v6: the one conversion of a schema-4 observation into the policy network's tensors (tensor ABI 4, or 5
 * with eight floats per token), shared by the training harness (bench PolicyTensorsV4, which adds the native batch
 * packing) and production Brainiac (which adds the browser wire format, {@link Tensors#browserBytes()}).
 */
public final class ActorTensors {
    private static final List<String> TYPES = List.of("EMPTY", "INTEGER", "MULTIPLE_CHOICE", "ARBITRARY_CARDS",
        "CARD_ACTION_CHOICE", "ACTION_CHOICE", "CARD_SELECTION");
    /** Browser payload magic, "SWT4" / "SWT5" little-endian: header, head, tok, tokf (4 or 8 floats), opt, kgrid. */
    public static final int BROWSER_MAGIC_4 = 0x34545753;
    public static final int BROWSER_MAGIC_5 = 0x35545753;

    private ActorTensors() {
    }

    /** A table bundle's card index and vocabularies, with its contract (tensor ABI, context version, table SHA). */
    public static final class Tables {
        public final Map<String, Integer> cards;
        public final Map<String, Integer> q;
        public final Map<String, Integer> o;
        public final Map<String, Integer> context;
        public final int contextVersion;
        public int tensorAbi = 4;
        public String tableSha;
        /** Plan titles-and-learner-ablation: normalizer v2 reads the engine's marked card names (NameMarks). */
        public boolean nameMarks;

        public Tables(Map<String, Integer> cards, Map<String, Integer> q, Map<String, Integer> o,
                Map<String, Integer> context, int contextVersion) {
            this.cards = cards;
            this.q = q;
            this.o = o;
            this.context = context;
            this.contextVersion = contextVersion;
        }

        /** Loads and validates a bundle directory (vocab.json, cards.json, bundle.json) of vocabulary schema 4. */
        public static Tables load(Path directory) throws IOException {
            JsonObject vocabulary = JsonParser.parseString(Files.readString(directory.resolve("vocab.json")))
                .getAsJsonObject();
            JsonObject cards = JsonParser.parseString(Files.readString(directory.resolve("cards.json")))
                .getAsJsonObject();
            if (!vocabulary.has("schema_version") || vocabulary.get("schema_version").getAsInt() != 4) {
                throw new IllegalArgumentException("Schema-4 vocabulary required");
            }
            Tables tables = new Tables(map(cards), map(vocabulary.getAsJsonObject("q")),
                map(vocabulary.getAsJsonObject("o")),
                vocabulary.has("context") ? map(vocabulary.getAsJsonObject("context")) : Map.of(),
                vocabulary.has("context_version") ? vocabulary.get("context_version").getAsInt() : 4);
            validate(tables, vocabulary, directory);
            return tables;
        }

        private static Map<String, Integer> map(JsonObject object) {
            Map<String, Integer> result = new java.util.HashMap<>(object.size() * 2);
            for (Map.Entry<String, JsonElement> entry : object.entrySet()) {
                result.put(entry.getKey(), entry.getValue().getAsInt());
            }
            return java.util.Collections.unmodifiableMap(result);
        }
    }

    /** One decision's tensors. tok32 [T, 8], tokf [T, floatWidth], opt [O, 6], head [22], kgrid [K, 153]. */
    public static final class Tensors {
        public final int tokens;
        public final int options;
        public final int floatWidth;
        public final int[] tok32;
        public final float[] tokf;
        public final int[] opt;
        public final int[] head = new int[22];
        public final boolean[] isNone;
        public final byte[] kgrid;
        public final int gridRows;
        public final String tableSha;
        public final boolean privileged;

        public Tensors(int tokens, int options, int floatWidth, int[] tok32, float[] tokf, int[] opt, int[] head,
                boolean[] isNone, byte[] kgrid, int gridRows, String tableSha, boolean privileged) {
            this.tokens = tokens;
            this.options = options;
            this.floatWidth = floatWidth;
            this.tok32 = tok32;
            this.tokf = tokf;
            this.opt = opt;
            System.arraycopy(head, 0, this.head, 0, 22);
            this.isNone = isNone;
            this.kgrid = kgrid;
            this.gridRows = gridRows;
            this.tableSha = tableSha;
            this.privileged = privileged;
        }

        private Tensors(int tokens, int options, int floatWidth, int gridRows, String tableSha, boolean privileged) {
            this.tokens = tokens;
            this.options = options;
            this.floatWidth = floatWidth;
            tok32 = new int[tokens * 8];
            tokf = new float[tokens * floatWidth];
            opt = new int[options * 6];
            isNone = new boolean[options];
            kgrid = new byte[gridRows * KnowledgeGrid.WIDTH];
            this.gridRows = gridRows;
            this.tableSha = tableSha;
            this.privileged = privileged;
        }

        /** The browser wire format (actor view only): magic, T, O, K, head, tok32, tokf, opt, kgrid; little-endian. */
        public byte[] browserBytes() {
            if (privileged) {
                throw new IllegalArgumentException("Privileged tensors rejected by browser actor path");
            }
            ByteBuffer buffer = ByteBuffer.allocate(4 * (4 + 22) + 4 * tok32.length + 4 * tokf.length
                + 4 * opt.length + kgrid.length).order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(floatWidth == 8 ? BROWSER_MAGIC_5 : BROWSER_MAGIC_4).putInt(tokens).putInt(options)
                .putInt(gridRows);
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

    /** The bundle contract: vocabulary hashes, OOV layout, closed context keys, card-map hash, table SHA. */
    public static void validate(Tables tables, JsonObject vocabulary, Path directory) throws IOException {
        String metadata = Files.readString(directory.resolve("bundle.json"));
        JsonObject bundle = JsonParser.parseString(metadata).getAsJsonObject();
        String sha = HexFormat.of().formatHex(VocabularyV4.fingerprint(vocabulary));
        tables.tensorAbi = bundle.has("tensor_abi") ? bundle.get("tensor_abi").getAsInt() : 4;
        tables.nameMarks = TitleMatcher.NORMALIZER_V2.equals(vocabulary.get("normalizer_version").getAsString());
        if (tables.tensorAbi != 4 && tables.tensorAbi != 5) {
            throw new IllegalArgumentException("Unsupported tensor ABI " + tables.tensorAbi);
        }
        if (vocabulary.get("tensor_abi").getAsInt() != 4 || vocabulary.get("table_format").getAsInt() != 2
                || vocabulary.get("oov_buckets").getAsInt() != 512
                || !TitleMatcher.normalizers().contains(vocabulary.get("normalizer_version").getAsString())
                || !vocabulary.get("oov_hash").getAsString().equals(VocabularyV4.HASH)
                || !sha.equals(vocabulary.get("sha").getAsString())
                || !sha.equals(bundle.get("vocabulary_sha").getAsString())) {
            throw new IllegalArgumentException("ABI-4 vocabulary contract/hash mismatch");
        }
        // Plan actor-v6-inputs: the context family may reserve no OOV buckets (context_oov_buckets 0).
        int contextBuckets = vocabulary.has("context_oov_buckets") ? vocabulary.get("context_oov_buckets").getAsInt()
            : 512;
        for (Map<String, Integer> family : List.of(tables.q, tables.o, tables.context)) {
            int reserved = family == tables.context ? contextBuckets : 512;
            Set<Integer> ids = new HashSet<>(family.values());
            if (ids.size() != family.size()) {
                throw new IllegalArgumentException("Duplicate template IDs");
            }
            for (int index = 0; index < family.size(); index++) {
                if (!ids.contains(index) || index < reserved && !Integer.valueOf(index).equals(
                        family.get(String.format("<oov:%03d>", index)))) {
                    throw new IllegalArgumentException("Noncontiguous or changed OOV IDs");
                }
            }
            if (family.keySet().stream().anyMatch(key -> key.startsWith("<oov:")
                    && family.get(key) >= reserved)) {
                throw new IllegalArgumentException("OOV bucket outside the reserved rows");
            }
        }
        if (contextBuckets == 0) {
            // Closed context keys: fail at load, before any game, if the engine can emit a key the table lacks.
            List<String> missing = StoredContextV5.keys().stream()
                .filter(key -> !tables.context.containsKey(key)).toList();
            if (!missing.isEmpty()) {
                throw new IllegalArgumentException("Context vocabulary lacks " + missing.size()
                    + " engine keys (grow the tables): " + missing.subList(0, Math.min(10, missing.size())));
            }
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            new TreeMap<>(tables.cards).forEach((key, value) -> digest.update((key + "\t" + value + "\n")
                .getBytes(java.nio.charset.StandardCharsets.UTF_8)));
            if (!HexFormat.of().formatHex(digest.digest()).equals(bundle.get("card_map_sha").getAsString())) {
                throw new IllegalArgumentException("ABI-4 card-map hash mismatch");
            }
        } catch (java.security.NoSuchAlgorithmException exception) {
            throw new IllegalStateException(exception);
        }
        tables.tableSha = bundle.get("table_sha").getAsString();
        if (tables.tableSha.length() != 64 || HexFormat.of().parseHex(tables.tableSha).length != 32) {
            throw new IllegalArgumentException("Invalid table SHA");
        }
    }

    public static int card(String blueprint, Tables tables) {
        String stripped = ObservationV2.strip(blueprint);
        Integer index = tables.cards.get(stripped);
        if (index == null) {
            if (stripped.matches("[0-9]+_.*") && Integer.parseInt(stripped.split("_", 2)[0]) >= 200) {
                throw new IllegalArgumentException("Virtual blueprint requires explicit table growth: " + blueprint);
            }
            throw new IllegalArgumentException("Blueprint absent from this card table: " + blueprint);
        }
        return index;
    }

    public static Tensors convert(List<ObservationV2.Token> rows, List<int[]> grid, List<Object[]> options,
            String type, String question, int[] globals, int side, Tables tables, boolean privileged) {
        if (tables.tableSha == null) {
            throw new IllegalArgumentException("ABI-4 own-table bundle required");
        }
        rows = contextRows(rows, tables.contextVersion);
        int width = tables.tensorAbi == 5 ? 8 : 4;
        Tensors result = new Tensors(rows.size(), options.size(), width, grid.size(), tables.tableSha, privileged);
        for (int index = 0; index < rows.size(); index++) {
            ObservationV2.Token row = rows.get(index);
            int slot = row.kind == 7 ? VocabularyV4.lookup(tables.context, "context", row.contextKey) : row.slot + 1;
            int[] values = {row.kind, card(row.bp, tables), row.zone, row.mine, slot,
                row.parent + 1, row.count, row.flags};
            System.arraycopy(values, 0, result.tok32, index * 8, 8);
            float[] floats = {row.a, row.b, row.c, row.d};
            System.arraycopy(floats, 0, result.tokf, index * width, 4);
            if (width == 8 && row.state() != null) {
                System.arraycopy(row.state(), 0, result.tokf, index * width + 4, 4);
            }
            if (!privileged && row.kind == 7 && (row.contextKey.startsWith("rules.")
                    || row.contextKey.startsWith("priv."))) {
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
                : blueprint.equals("inPlay") || blueprint.equals("rules") ? 1 : card(blueprint, tables);
            result.isNone[index] = template.equals("<none>") || template.equals("<pass>");
            // Plan multiselect-count stage B: column 1 (formerly reserved, always 0) = the option's card is already
            // selected in a step-by-step card selection (an optional sixth option element, 0 when absent).
            int selected = option.length > 5 && option[5] != null ? (Integer) option[5] : 0;
            int[] values = {VocabularyV4.lookup(tables.o, "o", template), selected, (Integer) option[2] + 1,
                blueprintId, option[4] == null ? 0 : Math.addExact((Integer) option[4], 1),
                result.isNone[index] ? 1 : 0};
            System.arraycopy(values, 0, result.opt, index * 6, 6);
        }
        result.head[0] = TYPES.indexOf(type);
        result.head[1] = VocabularyV4.lookup(tables.q, "q", question);
        result.head[3] = side;
        if (globals.length != 18) {
            throw new IllegalArgumentException("Invalid global width");
        }
        System.arraycopy(globals, 0, result.head, 4, 18);
        validate(result, tables, !privileged);
        return result;
    }

    /**
     * Plan context-overhaul: only the context rows of this vocabulary's context version (an observation may carry
     * both versions, in a match between models of different versions). A row whose parent row is dropped goes too;
     * the remaining references are re-indexed on copies, so the observation itself is unchanged.
     */
    public static List<ObservationV2.Token> contextRows(List<ObservationV2.Token> rows, int version) {
        int[] position = new int[rows.size()];
        List<ObservationV2.Token> kept = new ArrayList<>(rows.size());
        boolean dropped = false;
        for (int index = 0; index < rows.size(); index++) {
            ObservationV2.Token row = rows.get(index);
            boolean drop = row.kind == 7 && (!StoredContextV5.belongs(row.contextKey, version)
                || row.parent >= 0 && row.parent < index && position[row.parent] < 0);
            position[index] = drop ? -1 : kept.size();
            if (drop) {
                dropped = true;
            } else {
                kept.add(row);
            }
        }
        if (!dropped) {
            return rows;
        }
        for (int index = 0; index < kept.size(); index++) {
            ObservationV2.Token row = kept.get(index);
            boolean edge = row.kind == 8 && row.a > 0;
            if (row.parent < 0 && !edge) {
                continue;
            }
            ObservationV2.Token copy = copy(row);
            if (row.parent >= 0) {
                copy.parent = position[row.parent];
            }
            if (edge) {
                copy.a = position[(int) row.a - 1] + 1;
            }
            if (copy.parent < -1 || row.parent >= 0 && copy.parent < 0 || edge && copy.a <= 0) {
                throw new IllegalStateException("A kept row references a dropped context row: " + row.json());
            }
            kept.set(index, copy);
        }
        return kept;
    }

    private static ObservationV2.Token copy(ObservationV2.Token row) {
        ObservationV2.Token copy = new ObservationV2.Token();
        copy.kind = row.kind;
        copy.zone = row.zone;
        copy.mine = row.mine;
        copy.slot = row.slot;
        copy.parent = row.parent;
        copy.count = row.count;
        copy.flags = row.flags;
        copy.bp = row.bp;
        copy.contextKey = row.contextKey;
        copy.a = row.a;
        copy.b = row.b;
        copy.c = row.c;
        copy.d = row.d;
        copy.state = row.state();
        return copy;
    }

    /** A schema-4 record (12 fields: actor; 15: training envelope with the privileged view at 12-14). */
    public static Tensors fromRecord(JsonArray record, Tables tables, boolean privileged) {
        if (record.get(0).getAsInt() != 4 || record.size() != 12 && record.size() != 15
                || privileged && record.size() != 15) {
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
            if (values.size() == 16) {
                row.state = new float[]{values.get(12).getAsFloat(), values.get(13).getAsFloat(),
                    values.get(14).getAsFloat(), values.get(15).getAsFloat()};
            }
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
                    option.get(5).isJsonNull() ? null : option.get(5).getAsInt(),
                    option.size() > 6 ? option.get(6).getAsInt() : 0});
            }
        }
        return convert(rows, grid, options, record.get(4).getAsString(), record.get(5).getAsString(),
            ints(record.get(7).getAsJsonArray()), record.get(3).getAsString().equals("DS") ? 0 : 1, tables,
            privileged);
    }

    private static int[] ints(JsonArray values) {
        int[] result = new int[values.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = values.get(index).getAsInt();
        }
        return result;
    }

    public static void validate(Tensors tensors, Tables tables, boolean actor) {
        if (actor && tensors.privileged || !tables.tableSha.equals(tensors.tableSha)) {
            throw new IllegalArgumentException("Privileged payload or table mismatch in actor path");
        }
        if (tensors.tok32 == null || tensors.tok32.length != 8 * tensors.tokens
                || tensors.floatWidth != (tables.tensorAbi == 5 ? 8 : 4)
                || tensors.tokf.length != tensors.floatWidth * tensors.tokens || tensors.opt.length != 6 * tensors.options
                || tensors.kgrid.length != 153 * tensors.gridRows || tensors.head[0] < 0 || tensors.head[0] >= 7
                || tensors.head[1] < 0 || tensors.head[1] >= tables.q.size() || tensors.head[2] != 0
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
            int f = index * tensors.floatWidth;
            float endpoint = tensors.tokf[f];
            if (kind < 0 || kind > 10 || kind == 4 || kind == 6 || tensors.tok32[start + 2] < 0
                    || tensors.tok32[start + 2] >= 64 || tensors.tok32[start + 3] < 0
                    || tensors.tok32[start + 3] > 1 || tensors.tok32[start + 1] < 0
                    || tensors.tok32[start + 1] >= tables.cards.size()) {
                throw new IllegalArgumentException("Invalid ABI-4 token");
            }
            for (int column = 0; column < tensors.floatWidth; column++) {
                if (!Float.isFinite(tensors.tokf[f + column])) {
                    throw new IllegalArgumentException("Nonfinite ABI-4 float");
                }
            }
            if (kind == 8 && (parent < 1 || parent > tensors.tokens || endpoint < 1
                    || endpoint > tensors.tokens || endpoint != (int) endpoint || slot < 1 || slot > 14)) {
                throw new IllegalArgumentException("Invalid physical edge");
            }
            if (kind == 7 && (slot < 0 || slot >= tables.context.size() || parent < 0 || parent > tensors.tokens
                    || flags < 0 || flags > 127)) {
                throw new IllegalArgumentException("Invalid inline context edge/key");
            }
            if (kind == 7) {
                float[] indices={endpoint,tensors.tokf[f+2],tensors.tokf[f+3],tensors.tok32[start+6]};
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
                        || endpoint!=0 || tensors.tokf[f+1]!=0 || tensors.tokf[f+2]!=0
                        || tensors.tokf[f+3]!=0) {
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
                    || tensors.opt[index*6] >= tables.o.size() || (tensors.opt[index*6+1] & ~1) != 0
                    || tensors.opt[index*6+3] < 0 || tensors.opt[index*6+3] >= tables.cards.size()) {
                throw new IllegalArgumentException("Invalid option source");
            }
        }
    }
}
