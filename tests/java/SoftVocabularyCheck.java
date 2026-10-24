import com.gempukku.swccgo.game.knowledge.VocabularyV4;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.Map;

public final class SoftVocabularyCheck {
    public static void main(String[] args) throws Exception {
        var payload = JsonParser.parseString(Files.readString(Path.of(args[0]))).getAsJsonObject();
        String expected = payload.get("sha").getAsString();
        String actual = HexFormat.of().formatHex(VocabularyV4.fingerprint(payload));
        if (!expected.equals(actual)) {
            throw new AssertionError("Python/Java soft-vocabulary hash differs: " + expected + " / " + actual);
        }
        System.setProperty("policy.maxUnknownKeysV4", "2");
        for (String key : new String[]{"first", "second", "third", "first"}) {
            VocabularyV4.lookup(Map.of(), "q", key);
        }
        var report = JsonParser.parseString(VocabularyV4.unknownJson()).getAsJsonObject();
        if (report.get("distinct_keys").getAsInt() != 2
                || report.getAsJsonObject("families").getAsJsonObject("q").get("first").getAsInt() != 2
                || report.getAsJsonObject("overflow_lookups").get("q").getAsInt() != 1) {
            throw new AssertionError("Complete/bounded unknown telemetry failed");
        }
        System.out.println("{\"passed\":true,\"sha\":\"" + actual + "\",\"bounded_unknowns\":true}");
    }
}
