import com.gempukku.swccgo.game.knowledge.NameMarks;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Plan titles-and-learner-ablation A6: marker round trip, stripping, the raw scope, and v2 matching of marked names. */
public final class NameMarksCheck {
    private static String mark(String blueprint, String name) {
        return NameMarks.START + blueprint + NameMarks.TITLE + name + NameMarks.END;
    }

    public static void main(String[] args) {
        String text = "Stack " + mark("1_45", "3B3-888") + " on " + mark("", "Sense") + " here";
        if (!NameMarks.strip(text).equals("Stack 3B3-888 on Sense here")) {
            throw new AssertionError("strip: " + NameMarks.strip(text));
        }
        if (NameMarks.strip("plain $1 \\ text") != "plain $1 \\ text" || NameMarks.strip(null) != null) {
            throw new AssertionError("unmarked text must pass through unchanged (same object)");
        }
        if (!NameMarks.strip(mark("1_1", "Price $5 \\ odd")).equals("Price $5 \\ odd")) {
            throw new AssertionError("replacement must be literal");
        }

        Map<String, String[]> parameters = new LinkedHashMap<>();
        parameters.put("zeta", new String[] {"a"});
        parameters.put("actionText", new String[] {text, null, "plain"});
        parameters.put("alpha", null);
        Map<String, String[]> read = NameMarks.read(parameters);
        if (!List.copyOf(read.keySet()).equals(List.of("zeta", "actionText", "alpha"))
                || !read.get("actionText")[0].equals("Stack 3B3-888 on Sense here") || read.get("actionText")[1] != null
                || read.get("alpha") != null || !parameters.get("actionText")[0].equals(text)) {
            throw new AssertionError("parameters: stripped copy in the original order, original untouched");
        }
        Map<String, String[]> plain = Map.of("results", new String[] {"OK"});
        if (NameMarks.read(plain) != plain) {
            throw new AssertionError("unmarked parameters must be returned as they are");
        }
        String raw = NameMarks.raw(true, () -> NameMarks.read(text));
        String outside = NameMarks.raw(false, () -> NameMarks.read(text));
        if (!raw.equals(text) || !outside.equals(NameMarks.strip(text)) || !NameMarks.read(text).equals(outside)) {
            throw new AssertionError("raw scope");
        }
        if (NameMarks.raw(true, () -> NameMarks.read(parameters)) != parameters) {
            throw new AssertionError("raw parameters must be the originals");
        }

        TitleMatcher matcher = new TitleMatcher(Map.of(
            "Sense", List.of("1_1"),
            "3B3-888", List.of("1_45"),
            "Han Solo", List.of("1_2", "200_2")));
        TitleMatcher.References references = new TitleMatcher.References();
        String template = matcher.template(text, references, Set.of());
        if (!template.equals("Stack <c> on <c> here") || !references.equals(List.of("1_45", "1_1"))) {
            throw new AssertionError("marked names: " + template + references);
        }
        references = new TitleMatcher.References();
        template = matcher.template("Cancel " + mark("", "Unknown Thing") + "'s text, " + mark("7_9", "Sense")
            + " and Sense", references, Set.of());
        if (!template.equals("Cancel Unknown Thing's text, <c> and Sense") || !references.equals(List.of("7_9"))) {
            throw new AssertionError("unknown title stays text; exact blueprint wins: " + template + references);
        }
        references = new TitleMatcher.References();
        template = matcher.template("Replace " + mark("", "Han Solo") + " with Han Solo", references, Set.of("200_2"));
        if (!template.equals("Replace <c> with <c>") || !references.equals(List.of("200_2", "200_2"))) {
            throw new AssertionError("title-only mark resolves like a title: " + template + references);
        }
        references = new TitleMatcher.References();
        template = matcher.template("Broken x Sense", references, Set.of());
        if (template.indexOf('') >= 0 || template.indexOf('') >= 0) {
            throw new AssertionError("malformed marks lose their markers: " + template);
        }
        System.out.println("NameMarksCheck passed");
    }
}
