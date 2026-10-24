import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class TitleMatcherCheck {
    public static void main(String[] args) {
        TitleMatcher matcher = new TitleMatcher(Map.of(
            "Sense", List.of("1_1"),
            "Han Solo", List.of("1_2", "200_2"),
            "Han Solo With Heavy Blaster Pistol", List.of("200_3"),
            "Dagobah: Bog Clearing", List.of("2_1"),
            "Dagobah: Swamp", List.of("2_2")
        ));
        TitleMatcher.References references = new TitleMatcher.References();
        String text = matcher.template(
            "Sense then 'Sense', ••Han Solo With Heavy Blaster Pistol and <>Han Solo",
            references, Set.of("200_2")
        );
        if (!text.equals("Sense then '<c>', <c> and <c>")
                || !references.equals(List.of("1_1", "200_3", "200_2"))
                || TitleMatcher.flag(references, 2) != 0) {
            throw new AssertionError("Longest/title/quote matching failed: " + text + references);
        }
        references = new TitleMatcher.References();
        text = matcher.template(
            "On which side of •Dagobah: Bog Clearing do you want to deploy •Dagobah: Swamp",
            references, Set.of()
        );
        if (!text.equals("On which side of <c> do you want to deploy <c>") || references.size() != 2) {
            throw new AssertionError("Real deployment template was not normalized");
        }
        references = new TitleMatcher.References();
        text = matcher.template(
            "Han Solo before <div class='cardHint' value='1_1'>Sense</div> and Han Solo",
            references, Set.of()
        );
        if (!text.equals("<c> before <c> and <c>") || !references.equals(List.of("1_2", "1_1", "1_2"))) {
            throw new AssertionError("Markup/plain reference order failed: " + text + references);
        }
        if (TitleMatcher.flag(references, 0) != TitleMatcher.CANONICAL
                || TitleMatcher.flag(references, 1) != 0) {
            throw new AssertionError("Canonical reference flags lost");
        }
        references = new TitleMatcher.References();
        String unique = "Reveal up to two unique (•) aliens";
        if (!matcher.template(unique, references, Set.of()).equals(unique) || !references.isEmpty()) {
            throw new AssertionError("Literal uniqueness notation became a title reference");
        }
        // Hidden cards cannot enter resolution: the API only accepts the allowed set.
        // Neither a hidden reprint nor collection order changes the canonical choice.
        var unknown = matcher.resolve("Han Solo", Set.of());
        var ambiguous = matcher.resolve("Han Solo", Set.of("200_2", "1_2"));
        if (!unknown.equals(ambiguous) || !unknown.blueprint().equals("1_2")
                || unknown.flags() != TitleMatcher.CANONICAL) {
            throw new AssertionError("Ambiguous resolution is not canonical");
        }
        TitleMatcher amended = new TitleMatcher(Map.of(
            "Swamp", List.of("3_1"), "Ralltiir: Forest", List.of("3_2"),
            "Tatooine: Docking Bay 94", List.of("3_3"), "Docking Bay 94", List.of("3_4"),
            "Tatooine", List.of("3_5"), "Han Solo", List.of("1_2", "200_2"),
            "Han Solo (V)", List.of("200_2")
        ));
        references = new TitleMatcher.References();
        text = amended.template("◇Swamp and ◇Ralltiir: Forest and ◇Tatooine: Docking Bay 94",
            references, Set.of());
        if (!text.equals("<c> and <c> and <c>") || !references.equals(List.of("3_1", "3_2", "3_3"))) {
            throw new AssertionError("Diamond titles did not match from their markers: " + text + references);
        }
        references = new TitleMatcher.References();
        String unknownMarker = "◇Tatooine: Unknown Docking Bay 94";
        if (!amended.template(unknownMarker, references, Set.of()).equals(unknownMarker)
                || !references.isEmpty()) {
            throw new AssertionError("Unknown marked title matched a shorter title inside it");
        }
        references = new TitleMatcher.References();
        if (!amended.contextTemplate(unknownMarker, references, Set.of()).equals(unknownMarker)
                || !references.isEmpty()) {
            throw new AssertionError("Unknown context title was partially replaced");
        }
        references = new TitleMatcher.References();
        text = amended.template("Deploy •Han Solo (V) and <div class='cardHint' value='1_2' "
            + "data-horizontal='true'>Han Solo</div>", references, Set.of("1_2"));
        if (!text.equals("Deploy <c> and <c>") || !references.equals(List.of("200_2", "1_2"))
                || TitleMatcher.flag(references, 0) != TitleMatcher.CANONICAL
                || TitleMatcher.flag(references, 1) != 0) {
            throw new AssertionError("Variant suffix or attributed markup lost: " + text + references);
        }
        references = new TitleMatcher.References();
        text = amended.template("About to lose 4 Force, draw 4 battle destiny (limited to 1 draw), "
            + "<mc>14 and SSA-306 OOM-9 2-1B and •Han Solo", references, Set.of("1_2"));
        if (!text.equals("About to lose <n> Force, draw <n> battle destiny (limited to <n> draw), "
                + "<mc><n> and SSA-306 OOM-9 2-1B and <c>") || references.size() != 5) {
            throw new AssertionError("Numeric normalization changed names or ordinals: " + text + references);
        }
        for (int index = 0; index < 4; index++) {
            float expected = new float[]{4, 4, 1, 14}[index];
            if (TitleMatcher.flag(references, index) != TitleMatcher.NUMERIC
                    || TitleMatcher.value(references, index) != expected || !references.get(index).equals("<pad>")) {
                throw new AssertionError("Numeric reference lost value or flag");
            }
        }
        System.out.println("{\"passed\":true,\"longest_match\":true,\"canonical_boundary\":true,"
            + "\"ordered_references\":true}");
    }
}
