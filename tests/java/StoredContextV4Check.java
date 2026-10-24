import com.gempukku.swccgo.filters.Filter;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.StoredContextV4;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying;
import com.gempukku.swccgo.logic.modifiers.DefaultModifierFixture;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Checks raw leaf multiplicity, four-level tuples, stable names and cyclic aliases. */
public final class StoredContextV4Check {
    private static final GameKnowledge KNOWLEDGE = new GameKnowledge("ds", "ls");
    private static PhysicalCard testCard;
    private static final GameState STATE = new GameState() {
        @Override
        public GameKnowledge getKnowledge() {
            return KNOWLEDGE;
        }

        @Override
        public PhysicalCard findCardById(Integer id) {
            return id == 42 ? testCard : null;
        }

        @Override
        public PhysicalCard findCardByPermanentId(Integer id) {
            return id == 42 ? testCard : null;
        }
    };

    private static ObservationV2 observation(int schema) {
        SwccgGame game = (SwccgGame) Proxy.newProxyInstance(
            SwccgGame.class.getClassLoader(), new Class<?>[]{SwccgGame.class},
            (proxy, method, arguments) -> {
                if (method.getName().equals("getGameState")) {
                    return STATE;
                }
                if (method.getName().equals("getModifiersEnvironment")) {
                    return null;
                }
                throw new AssertionError(method.getName());
            }
        );
        ObservationV2 observation = new ObservationV2(game, "ds", false, schema);
        observation.tokens.clear();
        return observation;
    }

    private static ObservationV2 capture(int schema, Object value) throws ReflectiveOperationException {
        ObservationV2 observation = observation(schema);
        StoredContextV4.capture(observation, STATE, "fixture", value);
        return observation;
    }


    private static Filter nested() {
        return new Filter() {
            final int amount = 7;
            final Filter child = new Filter() {
                final int amount = 9;

                @Override
                public boolean accepts(GameState state, ModifiersQuerying querying, PhysicalCard card) {
                    throw new AssertionError("Reflection must never evaluate a filter");
                }
            };

            @Override
            public boolean accepts(GameState state, ModifiersQuerying querying, PhysicalCard card) {
                throw new AssertionError("Reflection must never evaluate a filter");
            }
        };
    }


    public static void main(String[] args) throws ReflectiveOperationException {
        String priorAudit = System.getProperty("policy.checkContextV4");
        System.setProperty("policy.checkContextV4", "unit-fixture");
        for (Object empty : Arrays.asList(null, Map.of(), List.of(), new Object[0], new int[0])) {
            ObservationV2 cleaned = capture(4, empty);
            if (!cleaned.tokens.isEmpty()) {
                throw new AssertionError("Empty container emitted tokens");
            }
        }
        Object deeplyEmpty = Map.of();
        for (int depth = 0; depth < 9; depth++) {
            deeplyEmpty = Map.of("child", deeplyEmpty);
        }
        ObservationV2 emptyAtLimit = capture(4, deeplyEmpty);
        if (!emptyAtLimit.tokens.isEmpty() || emptyAtLimit.contextOmissions.size() != 1) {
            throw new AssertionError("A1 empty depth-limit omission lacks audit provenance");
        }
        if (priorAudit == null) {
            System.clearProperty("policy.checkContextV4");
        } else {
            System.setProperty("policy.checkContextV4", priorAudit);
        }
        System.setProperty("policy.checkContextV4", "unit-fixture");
        for (int amount : new int[]{0, 7}) {
            DefaultModifierFixture modifier = new DefaultModifierFixture(amount);
            ObservationV2 defaults = capture(4, modifier);
            if (defaults.tokens.stream().noneMatch(row -> row.contextKey.contains(".type="))) {
                throw new AssertionError("Default-only modifier lost its type");
            }
        }
        if (priorAudit == null) {
            System.clearProperty("policy.checkContextV4");
        } else {
            System.setProperty("policy.checkContextV4", priorAudit);
        }
        Filter filter = nested();
        long types = capture(4, filter).tokens.stream().filter(row -> row.contextKey.contains(".type=")).count();
        if (types != 2) {
            throw new AssertionError("Meaningful anonymous filter types were lost");
        }
        for (ObservationV2.Token row : capture(4, filter).tokens) {
            if (row.contextKey.matches(".*\\$[0-9]+.*")) {
                throw new AssertionError("Unstable anonymous name: " + row.contextKey);
            }
        }
        Object tuple = List.of(List.of(List.of(List.of(3, 5))));
        ObservationV2 positions = capture(4, tuple);
        if (positions.tokens.get(1).count != 1) {
            throw new AssertionError("Fourth list index was not preserved");
        }
        try {
            capture(4, List.of(tuple));
            throw new AssertionError("Fifth list-index level was accepted");
        } catch (IllegalStateException expected) {
            if (!expected.getMessage().contains("four list-index levels")) {
                throw expected;
            }
        }
        Map<String, Object> cycle = new LinkedHashMap<>();
        cycle.put("amount", 4);
        cycle.put("self", cycle);
        ObservationV2 cyclic = capture(4, cycle);
        int aliases = 0;
        for (int index = 0; index < cyclic.tokens.size(); index++) {
            ObservationV2.Token row = cyclic.tokens.get(index);
            if (row.kind == 8) {
                throw new AssertionError("Context emitted a separate EDGE row");
            }
            if ((row.flags & StoredContextV4.ALIAS) != 0) {
                aliases++;
                if (index == 0 || row.parent != 0) {
                    throw new AssertionError("Alias source/target remapping failed");
                }
            }
        }
        if (aliases != 1) {
            throw new AssertionError("Expected one cyclic alias");
        }
        Filter emptyFilter = new Filter() {
            @Override
            public boolean accepts(GameState state, ModifiersQuerying querying, PhysicalCard card) {
                throw new AssertionError("Filter must not be evaluated");
            }
        };
        if (!capture(4, emptyFilter).tokens.isEmpty()) {
            throw new AssertionError("Non-opaque type without descendants was retained");
        }
        ObservationV2 opaque = capture(4, new Object());
        if (opaque.tokens.size() != 1 || !opaque.tokens.get(0).contextKey.equals("fixture.opaque=java.lang.Object")) {
            throw new AssertionError("Opaque stored object was lost");
        }
        ObservationV2 linked = observation(4);
        linked.tokens.add(new ObservationV2.Token());
        PhysicalCard source = (PhysicalCard) Proxy.newProxyInstance(
            PhysicalCard.class.getClassLoader(), new Class<?>[]{PhysicalCard.class},
            (proxy, method, arguments) -> {
                if (method.getName().equals("getPermanentCardId") || method.getName().equals("getCardId")) {
                    return 42;
                }
                throw new AssertionError(method.getName());
            }
        );
        Field permanent = ObservationV2.class.getDeclaredField("byPermanent");
        permanent.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<Integer, Integer> byPermanent = (Map<Integer, Integer>) permanent.get(linked);
        byPermanent.put(42, 0);
        testCard = source;
        linked.contextV4("source", null, source, 7, new int[]{0, 2, 4, 6});
        ObservationV2.Token leaf = linked.tokens.get(1);
        if (linked.tokens.size() != 2 || leaf.parent != 0 || leaf.a != 0 || leaf.c != 2
                || leaf.d != 4 || leaf.count != 6 || leaf.b != 7) {
            throw new AssertionError("Source edge or list tuple was not carried on the context row");
        }
        StoredContextV4.capture(linked, STATE, "map", Map.of(42, 9));
        ObservationV2.Token mapped = linked.tokens.get(2);
        if (!mapped.contextKey.equals("map.card") || mapped.parent != 0 || mapped.a != -1) {
            throw new AssertionError("Integer card ID became a numeric feature");
        }
        ObservationV2 literal = capture(4, Map.of(999, 7));
        if (!literal.tokens.get(0).contextKey.equals("fixture.999") || literal.tokens.get(0).a != -1) {
            throw new AssertionError("Unresolved numeric map key became a list index");
        }
        System.out.println("{\"passed\":true,\"empty_containers\":5,\"tuple_levels\":4,"
            + "\"cycle_aliases\":1,\"source_link\":true,\"leaf_multisets_equal\":true}");
        System.out.println(StoredContextV4.coverageJson());
    }
}
