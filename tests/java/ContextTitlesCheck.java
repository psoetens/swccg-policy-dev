import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.ContextTitles;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.PileKnowledge;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import com.gempukku.swccgo.game.state.GameState;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Stored strings retain every ordered title, scalar and tuple; counters keep their family/action. */
public final class ContextTitlesCheck {
    public static void main(String[] args) throws Exception {
        Path table = Files.createTempFile("swccg-title-check-", ".tsv");
        Files.writeString(table, "1_2\tHan Solo\n200_2\tHan Solo\n1_3\tLuke Skywalker\n");
        System.setProperty("policy.titleTable", table.toString());
        try {
            GameKnowledge knowledge = new GameKnowledge("ds", "ls");
            GameState state = new GameState() {
                @Override
                public GameKnowledge getKnowledge() {
                    return knowledge;
                }
            };
            SwccgGame game = (SwccgGame) Proxy.newProxyInstance(
                SwccgGame.class.getClassLoader(), new Class<?>[]{SwccgGame.class},
                (proxy, method, arguments) -> switch (method.getName()) {
                    case "getGameState" -> state;
                    case "getModifiersEnvironment" -> null;
                    default -> throw new AssertionError(method.getName());
                }
            );
            ObservationV2 observation = new ObservationV2(game, "ds", false, 4);
            observation.tokens.clear();
            observation.contextV4("states.message=•Han Solo meets Luke Skywalker", null, null, 7,
                new int[]{0, 3, 2, 1});
            ContextTitles.normalizeAll(observation);
            ObservationV2.Token first = observation.tokens.get(0);
            ObservationV2.Token second = observation.tokens.get(1);
            if (observation.tokens.size() != 2 || !first.contextKey.equals("states.message=<c> meets <c>")
                    || !first.bp.equals("1_2") || first.flags != ContextTitles.CANONICAL || first.b != 7
                    || first.a != 0 || first.c != 3 || first.d != 2 || first.count != 1) {
                throw new AssertionError("Primary context title/scalar/index tuple lost");
            }
            if (!second.contextKey.equals("context.titleReference") || !second.bp.equals("1_3")
                    || second.parent != 0 || second.b != 1
                    || second.flags != (ContextTitles.REFERENCE | ContextTitles.CANONICAL)
                    || second.a != -1 || second.c != -1 || second.d != -1 || second.count != -1) {
                throw new AssertionError("Additional ordered title reference lost");
            }
            knowledge.views[0][0].initialize(PileKnowledge.RESERVE, 1, List.of("200_2"));
            ObservationV2 counter = new ObservationV2(game, "ds", false, 4);
            counter.tokens.clear();
            counter.contextV4("usage._gameLimitCounters.Han Solo|DEPLOY_CARD_ACTION", null, null, 2, new int[0]);
            ContextTitles.normalizeAll(counter);
            ObservationV2.Token row = counter.tokens.get(0);
            if (counter.tokens.size() != 1 || !row.contextKey.equals("usage._gameLimitCounters|DEPLOY_CARD_ACTION")
                    || !row.bp.equals("200_2") || row.flags != 0 || row.b != 2) {
                throw new AssertionError("Counter family/action or permitted title resolution lost");
            }
            TitleMatcher.References references = new TitleMatcher.References();
            TitleMatcher.shared().template("Use 3 Force with •Han Solo", references, counter.titleBlueprints());
            counter.reference(references, 0, 0);
            counter.reference(references, 2, 0);
            counter.reference(references, 2, 1);
            ObservationV2.Token questionNumber = counter.tokens.get(1);
            ObservationV2.Token optionNumber = counter.tokens.get(2);
            ObservationV2.Token optionCard = counter.tokens.get(3);
            if (questionNumber.kind != 9 || optionNumber.kind != 10 || questionNumber.b != 3
                    || optionNumber.b != 3 || optionNumber.flags != TitleMatcher.NUMERIC
                    || questionNumber.flags != TitleMatcher.NUMERIC || !optionNumber.bp.equals("<pad>")
                    || optionNumber.a != 2 || optionNumber.slot != 0 || optionCard.slot != 1
                    || !optionCard.bp.equals("200_2")) {
                throw new AssertionError("Question/option numeric rows lost routing, quantity or shared ordinals");
            }
            System.out.println("{\"passed\":true,\"ordered_titles\":2,\"tuple_and_scalar_preserved\":true,"
                + "\"counter_identity_extracted\":true}");
        } finally {
            Files.delete(table);
            System.clearProperty("policy.titleTable");
        }
    }
}
