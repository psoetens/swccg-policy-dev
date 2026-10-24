import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.PileKnowledge;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import com.gempukku.swccgo.game.state.GameState;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;

/** A hidden card throws on identity access; only public metadata may be inspected. */
public final class TitleBoundaryCheck {
    public static void main(String[] args) {
        GameKnowledge knowledge = new GameKnowledge("ds", "ls");
        PhysicalCard hidden = (PhysicalCard) Proxy.newProxyInstance(
            PhysicalCard.class.getClassLoader(), new Class<?>[]{PhysicalCard.class},
            (proxy, method, arguments) -> switch (method.getName()) {
                case "getOwner" -> "ls";
                case "getZone" -> Zone.RESERVE_DECK;
                case "isNotShownOnUserInterface", "isInserted" -> false;
                case "getPermanentCardId", "getCardId" -> 42;
                default -> throw new AssertionError("Actor read hidden card field: " + method.getName());
            }
        );
        GameState state = new GameState() {
            @Override
            public GameKnowledge getKnowledge() {
                return knowledge;
            }

            @Override
            public List<PhysicalCard> getAllPermanentCards() {
                return List.of(hidden);
            }
        };
        SwccgGame game = (SwccgGame) Proxy.newProxyInstance(
            SwccgGame.class.getClassLoader(), new Class<?>[]{SwccgGame.class},
            (proxy, method, arguments) -> switch (method.getName()) {
                case "getGameState" -> state;
                case "getModifiersEnvironment" -> null;
                default -> throw new AssertionError("Unexpected simulator lookup: " + method.getName());
            }
        );
        knowledge.views[0][1].initialize(PileKnowledge.RESERVE, 1, List.of());
        knowledge.disclosed[1].initialize(PileKnowledge.RESERVE, 1, List.of());
        TitleMatcher matcher = new TitleMatcher(Map.of(
            "Han Solo", List.of("1_2", "200_2"), "Han Solo (V)", List.of("200_2")
        ));
        String previous = null;
        for (String privateBlueprint : List.of("1_2", "200_2")) {
            knowledge.views[1][1].initialize(PileKnowledge.RESERVE, 1, List.of(privateBlueprint));
            ObservationV2 actor = new ObservationV2(game, "ds", false, 4);
            String observation = actor.json()
                + com.gempukku.swccgo.game.knowledge.KnowledgeGrid.json(actor.kgrid);
            if (previous != null && !previous.equals(observation)) {
                throw new AssertionError("Opponent-private identity changed the actor observation");
            }
            previous = observation;
            TitleMatcher.References references = new TitleMatcher.References();
            String text = matcher.template("Deploy •Han Solo", references, TitleMatcher.permitted(actor));
            if (!text.equals("Deploy <c>") || !references.equals(List.of("1_2"))
                    || TitleMatcher.flag(references, 0) != TitleMatcher.CANONICAL) {
                throw new AssertionError("Hidden identity influenced canonical title resolution");
            }
            if (!matcher.resolve("Han Solo", actor.titleBlueprints()).blueprint().equals("1_2")) {
                throw new AssertionError("Context-title resolver used opponent-private knowledge");
            }
            var variant = matcher.resolve("Han Solo (V)", TitleMatcher.permitted(actor));
            if (!variant.blueprint().equals("200_2") || variant.flags() != TitleMatcher.CANONICAL) {
                throw new AssertionError("Variant suffix consulted a hidden card");
            }
        }
        knowledge.views[0][0].initialize(PileKnowledge.RESERVE, 1, List.of("200_2"));
        ObservationV2 actor = new ObservationV2(game, "ds", false, 4);
        for (var allowed : List.of(TitleMatcher.permitted(actor), actor.titleBlueprints())) {
            TitleMatcher.Resolved resolved = matcher.resolve("Han Solo", allowed);
            if (!resolved.blueprint().equals("200_2") || resolved.flags() != 0) {
                throw new AssertionError("Actor's own knowledge did not resolve the title");
            }
            if (matcher.resolve("Han Solo (V)", allowed).flags() != 0) {
                throw new AssertionError("Known variant remained canonical");
            }
        }
        System.out.println("{\"passed\":true,\"hidden_identity_accesses\":0,\"private_variants\":2,"
            + "\"own_knowledge_resolution\":true}");
    }
}
