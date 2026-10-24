import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.PileKnowledge;
import com.gempukku.swccgo.game.knowledge.KnowledgeGrid;
import com.gempukku.swccgo.game.state.GameState;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

/** Checks Amendment 1: duplicate bounds have an equivalent intersection.
 * Uses the real, unchanged schema-3 encoder. The game shell supplies no cards or
 * modifiers; this check concerns stored knowledge, not a simulated legal game.
 */
public final class KnowledgeGridCapacityCheck {
    private static PileKnowledge knowledge(boolean firstViaTransit, boolean secondViaTransit) {
        PileKnowledge knowledge = new PileKnowledge();
        knowledge.initialize(PileKnowledge.RESERVE, 4, List.of("1_1", "1_1", "1_2", "1_2"));
        knowledge.rememberInventory();
        for (boolean viaTransit : new boolean[]{firstViaTransit, secondViaTransit}) {
            if (viaTransit) {
                knowledge.move(PileKnowledge.RESERVE, PileKnowledge.TRANSIT, null, -1, -1);
                knowledge.move(PileKnowledge.TRANSIT, PileKnowledge.USED, null, -1, -1);
            } else {
                knowledge.move(PileKnowledge.RESERVE, PileKnowledge.USED, null, -1, -1);
            }
            knowledge.tighten();
        }
        return knowledge;
    }

    private static List<ObservationV2.Token> encode(PileKnowledge pileKnowledge, boolean privileged) {
        GameKnowledge knowledge = new GameKnowledge("ds", "ls");
        knowledge.views[0][0] = pileKnowledge.copy();
        GameState state = new GameState() {
            @Override
            public GameKnowledge getKnowledge() {
                return knowledge;
            }
        };
        SwccgGame game = (SwccgGame) Proxy.newProxyInstance(
            SwccgGame.class.getClassLoader(),
            new Class<?>[]{SwccgGame.class},
            (proxy, method, arguments) -> {
                if (method.getName().equals("getGameState")) {
                    return state;
                }
                if (method.getName().equals("getModifiersEnvironment")) {
                    return null;
                }
                throw new AssertionError("Unexpected game call: " + method.getName());
            }
        );
        ObservationV2 observation = new ObservationV2(game, "ds", privileged, 3);
        List<ObservationV2.Token> rows = new ArrayList<>();
        for (ObservationV2.Token token : observation.tokens) {
            if (token.kind == 4 && token.bp.equals("1_1") && token.mine == 1 && token.slot == 5) {
                rows.add(token);
            }
        }
        return rows;
    }

    private static String json(List<ObservationV2.Token> rows) {
        StringJoiner result = new StringJoiner(",", "[", "]");
        for (ObservationV2.Token row : rows) {
            result.add(row.json());
        }
        return result.toString();
    }

    public static void main(String[] args) {
        PileKnowledge mixed = knowledge(true, false);
        int[] grid = KnowledgeGrid.row("1_1", new PileKnowledge[]{mixed});
        int offset = KnowledgeGrid.offset(PileKnowledge.N + 1, 0);
        if (grid[offset] != 2 || grid[offset + 1] != 2 || grid[offset + 2] != 1) {
            throw new AssertionError("Actual schema-4 cell must be [2,2,1]");
        }
        List<ObservationV2.Token> actor = encode(mixed, false);
        List<ObservationV2.Token> critic = encode(mixed, true);
        for (List<ObservationV2.Token> rows : List.of(actor, critic)) {
            if (rows.size() != 2) {
                throw new AssertionError("Expected two distinct schema-3 rows, found " + json(rows));
            }
            ObservationV2.Token weaker = rows.get(0);
            ObservationV2.Token stronger = rows.get(1);
            if (weaker.count != 1 || weaker.b != 3 || stronger.count != 2 || stronger.b != 2) {
                throw new AssertionError("Unexpected bounds: " + json(rows));
            }
            if (weaker.flags != 2 || stronger.flags != 2) {
                throw new AssertionError("Both facts must have exactly OUR scope");
            }
            int lower = rows.stream().mapToInt(row -> row.count).max().orElseThrow();
            int upper = rows.stream().mapToInt(row -> (int) row.b).min().orElseThrow();
            if (lower != 2 || upper != 2) {
                throw new AssertionError("Expected intersection [2,2]");
            }
            for (int copies = 0; copies <= 4; copies++) {
                boolean conjunction = rows.stream().allMatch(row -> lower >= row.count && upper <= row.b);
                final int candidate = copies;
                boolean feasible = rows.stream().allMatch(row -> candidate >= row.count && candidate <= row.b);
                if (!conjunction || feasible != (copies >= lower && copies <= upper)) {
                    throw new AssertionError("Constraint equivalence failed");
                }
            }
        }
        for (boolean viaTransit : new boolean[]{false, true}) {
            if (encode(knowledge(viaTransit, viaTransit), false).size() != 1) {
                throw new AssertionError("Homogeneous transfers should coalesce to one row");
            }
        }
        System.out.println("{\"reproduced\":true,\"constraint_equivalence\":true,\"intersection\":[2,2],"
            + "\"controls_passed\":2,\"actor_rows\":" + json(actor)
            + ",\"critic_rows\":" + json(critic) + "}");
    }
}
