import com.gempukku.swccgo.game.knowledge.VocabularyV4;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Cross-language hash vectors include UTF-8 and UTF-16 ordering edge cases. */
public final class VocabularyV4Check {
    public static void main(String[] args) {
        List<String> keys = List.of("", "hello", "Deploy •Dagobah", "é", "\uD83D\uDE80", "\uE000");
        Map<String, Map<String, Integer>> families = new TreeMap<>();
        for (String family : List.of("q", "o", "context")) {
            Map<String, Integer> entries = new TreeMap<>();
            for (int index = 0; index < VocabularyV4.OOV_BUCKETS; index++) {
                entries.put(String.format("<oov:%03d>", index), index);
            }
            int next = VocabularyV4.OOV_BUCKETS;
            for (String key : keys.stream().sorted().toList()) {
                entries.put(key, next++);
            }
            families.put(family, entries);
        }
        for (String key : keys) {
            System.out.println(VocabularyV4.unknown(key));
        }
        System.out.println(HexFormat.of().formatHex(VocabularyV4.fingerprint(
            families, VocabularyV4.OOV_BUCKETS, VocabularyV4.HASH)));
        for (String key : keys) {
            if (VocabularyV4.lookup(families.get("q"), "q", key) != families.get("q").get(key)) {
                throw new AssertionError("Known key was hashed: " + key);
            }
        }
        if (VocabularyV4.lookup(families.get("q"), "q", "unknown") != VocabularyV4.unknown("unknown")) {
            throw new AssertionError("Unknown key did not use its reserved bucket");
        }
        Map<String, Map<String, Integer>> invalid = new TreeMap<>(families);
        invalid.put("q", new TreeMap<>(families.get("q")));
        invalid.get("q").put("hello", 0);
        try {
            VocabularyV4.validate(invalid, 512, VocabularyV4.HASH);
            throw new AssertionError("Duplicate/reserved vocabulary ID accepted");
        } catch (IllegalArgumentException expected) {
            // A hash collision in OOV space is allowed; duplicate frozen IDs are not.
        }
    }
}
