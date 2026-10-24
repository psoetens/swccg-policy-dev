package com.gempukku.swccgo.game.knowledge;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.nio.file.Files;
import java.nio.file.Path;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/** ABI-4 vocabulary hashing; reserved OOV buckets are deliberately not identities. */
public final class VocabularyV4 {
    public static final int OOV_BUCKETS = 512;
    public static final String HASH = "fnv1a32-utf8";
    private static final Map<String, Map<String, Long>> UNKNOWN = new TreeMap<>();
    private static final Map<String, Long> OVERFLOW = new TreeMap<>();
    private static int unknownKeys;

    /**
     * Plan actor-v6-inputs: unknown keys are always recorded. The report goes to {@code -Dpolicy.unknownKeysV4} if
     * set, else to {@code <policy.unknownKeysDir or SWCCG_UNKNOWN_KEYS_DIR or runs/unknown-keys>/<host>-<pid>.json};
     * written every 10 minutes and at exit, and only once a key was unknown. scripts/unknown_keys.py merges them.
     */
    private static final Path REPORT = reportPath();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(VocabularyV4::writeReport, "vocabulary-v4-unknowns"));
        java.util.concurrent.ScheduledExecutorService flush = java.util.concurrent.Executors
            .newSingleThreadScheduledExecutor(runnable -> {
                Thread thread = new Thread(runnable, "vocabulary-v4-unknowns-flush");
                thread.setDaemon(true);
                return thread;
            });
        flush.scheduleAtFixedRate(VocabularyV4::writeReport, 10, 10, java.util.concurrent.TimeUnit.MINUTES);
    }

    private static Path reportPath() {
        String explicit = System.getProperty("policy.unknownKeysV4");
        if (explicit != null) {
            return Path.of(explicit);
        }
        String directory = System.getProperty("policy.unknownKeysDir", System.getenv().getOrDefault(
            "SWCCG_UNKNOWN_KEYS_DIR", "runs/unknown-keys"));
        String host;
        try {
            host = java.net.InetAddress.getLocalHost().getHostName();
        } catch (java.io.IOException exception) {
            host = "host";
        }
        return Path.of(directory, host + "-" + ProcessHandle.current().pid() + ".json");
    }

    private static void writeReport() {
        String json;
        synchronized (VocabularyV4.class) {
            if (unknownKeys == 0 && OVERFLOW.isEmpty()) {
                return;
            }
            json = unknownJson();
        }
        try {
            if (REPORT.getParent() != null) {
                Files.createDirectories(REPORT.getParent());
            }
            Path temporary = REPORT.resolveSibling(REPORT.getFileName() + ".tmp");
            Files.writeString(temporary, json);
            Files.move(temporary, REPORT, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (java.io.IOException exception) {
            System.err.println("ABI-4 unknown-key report failed: " + exception);
        }
    }

    private static synchronized void recordUnknown(String family, String key) {
        Map<String, Long> keys = UNKNOWN.computeIfAbsent(family, ignored -> new TreeMap<>());
        if (keys.containsKey(key)) {
            keys.merge(key, 1L, Long::sum);
        } else if (unknownKeys < Integer.getInteger("policy.maxUnknownKeysV4", 100_000) && key.length() <= 65536) {
            keys.put(key, 1L);
            unknownKeys++;
        } else {
            OVERFLOW.merge(family, 1L, Long::sum);
        }
    }

    public static synchronized String unknownJson() {
        return new com.google.gson.Gson().toJson(Map.of("normalizer_version", TitleMatcher.NORMALIZER_VERSION,
            "families", UNKNOWN, "overflow_lookups", OVERFLOW, "distinct_keys", unknownKeys)) + "\n";
    }

    public static int unknown(String key) {
        int hash = 0x811c9dc5;
        for (byte value : key.getBytes(StandardCharsets.UTF_8)) {
            hash = (hash ^ (value & 255)) * 16777619;
        }
        return (int) (Integer.toUnsignedLong(hash) % OOV_BUCKETS);
    }

    public static int lookup(Map<String, Integer> vocabulary, String family, String key) {
        int known = TemplateCoverage.lookup(vocabulary, family, key);
        if (vocabulary.containsKey(key)) {
            return known;
        }
        if (!vocabulary.containsKey("<oov:000>")) {
            // Plan actor-v6-inputs: a family without OOV buckets (context version 5) has a closed key set.
            throw new IllegalStateException("Unknown " + family + " key in a vocabulary without OOV buckets: " + key);
        }
        recordUnknown(family, key);
        return unknown(key);
    }

    private static String asciiQuote(String value) {
        StringBuilder result = new StringBuilder("\"");
        for (char character : value.toCharArray()) {
            switch (character) {
                case '\\', '"' -> result.append('\\').append(character);
                case '\b' -> result.append("\\b");
                case '\t' -> result.append("\\t");
                case '\n' -> result.append("\\n");
                case '\f' -> result.append("\\f");
                case '\r' -> result.append("\\r");
                default -> {
                    if (character < 32 || character > 126) {
                        result.append(String.format("\\u%04x", (int) character));
                    } else {
                        result.append(character);
                    }
                }
            }
        }
        return result.append('"').toString();
    }

    private static String canonical(JsonElement element) {
        if (element.isJsonNull()) {
            return "null";
        }
        if (element.isJsonObject()) {
            java.util.StringJoiner result = new java.util.StringJoiner(",", "{", "}");
            Map<String, JsonElement> sorted = new TreeMap<>();
            element.getAsJsonObject().entrySet().forEach(entry -> sorted.put(entry.getKey(), entry.getValue()));
            sorted.forEach((key, value) -> result.add(asciiQuote(key) + ":" + canonical(value)));
            return result.toString();
        }
        if (element.isJsonArray()) {
            java.util.StringJoiner result = new java.util.StringJoiner(",", "[", "]");
            element.getAsJsonArray().forEach(value -> result.add(canonical(value)));
            return result.toString();
        }
        return element.getAsJsonPrimitive().isString() ? asciiQuote(element.getAsString()) : element.toString();
    }

    public static byte[] fingerprint(JsonObject payload) {
        JsonObject fields = payload.deepCopy();
        fields.remove("sha");
        try {
            return MessageDigest.getInstance("SHA-256").digest(canonical(fields).getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(exception);
        }
    }

    public static void validate(Map<String, Map<String, Integer>> families, int buckets, String hash) {
        if (buckets != OOV_BUCKETS || !HASH.equals(hash)) {
            throw new IllegalArgumentException("Unsupported frozen OOV contract");
        }
        for (String family : List.of("q", "o", "context")) {
            Map<String, Integer> entries = families.get(family);
            if (entries == null || entries.size() < OOV_BUCKETS) {
                throw new IllegalArgumentException("Missing reserved vocabulary buckets: " + family);
            }
            for (int index = 0; index < OOV_BUCKETS; index++) {
                String key = String.format("<oov:%03d>", index);
                if (!Integer.valueOf(index).equals(entries.get(key))) {
                    throw new IllegalArgumentException("Invalid reserved vocabulary bucket: " + family + " " + key);
                }
            }
            boolean[] seen = new boolean[entries.size()];
            for (Integer id : entries.values()) {
                if (id == null || id < 0 || id >= seen.length || seen[id]) {
                    throw new IllegalArgumentException("Vocabulary IDs must be unique and contiguous: " + family);
                }
                seen[id] = true;
            }
        }
    }

    public static byte[] fingerprint(Map<String, Map<String, Integer>> families, int buckets, String hash) {
        validate(families, buckets, hash);
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(("schema=4\nabi=4\noov_buckets=" + buckets + "\noov_hash=" + hash + "\n")
                .getBytes(StandardCharsets.UTF_8));
            for (String family : List.of("q", "o", "context")) {
                for (Map.Entry<String, Integer> entry : new TreeMap<>(families.get(family)).entrySet()) {
                    digest.update((family + "\t" + entry.getKey() + "\t" + entry.getValue() + "\n")
                        .getBytes(StandardCharsets.UTF_8));
                }
            }
            return digest.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(exception);
        }
    }
}
