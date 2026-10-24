package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.modifiers.DefaultLimitCounter;
import java.lang.reflect.Proxy;
import java.util.Map;

/** Duplicate counters require the same stored object and an exact semantic twin. */
public final class ContextCountersCheck {
    public static void main(String[] args) {
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
        DefaultLimitCounter counter = new DefaultLimitCounter();
        counter.incrementToLimit(3, 2);
        ObservationV2.Token publicToken = observation.contextV4(
            "usage._turnLimitCounters.PLAY", "ds", null, 2, new int[0]
        );
        observation.contextCounters.put(publicToken, counter);
        StoredContextV4.capture(observation, state, "rules", Map.of(
            "_turnLimitCounters", Map.of("none|ds|unused|PLAY", counter)
        ));
        ObservationV2.Token reference = observation.contextV4("fixture.alias", null, null, 1, new int[0]);
        reference.flags = StoredContextV4.ALIAS;
        reference.parent = 1;
        StoredContextV4.removeDuplicateCounters(observation);
        if (observation.tokens.size() != 2 || observation.contextDuplicates.size() != 1
                || observation.tokens.get(0) != observation.contextDuplicates.get(0).twin()
                || reference.parent != 0) {
            throw new AssertionError("Counter removal lost its twin or failed to remap an index");
        }
        ObservationV2.Token unmatched = observation.contextV4(
            "usage._turnLimitCounters.PLAY", "ds", null, 3, new int[0]
        );
        observation.contextCounters.put(unmatched, counter);
        StoredContextV4.removeDuplicateCounters(observation);
        if (!observation.tokens.contains(unmatched)) {
            throw new AssertionError("Different used value was removed");
        }
        try {
            StoredContextV4.verifyCounterTwin(unmatched, observation.tokens.get(0));
            throw new AssertionError("Invalid removal passed exact-twin check");
        } catch (IllegalStateException expected) {
            if (!expected.getMessage().contains("exact rules twin")) {
                throw expected;
            }
        }
        System.out.println("{\"passed\":true,\"verified_removals\":1,\"unmatched_retained\":true}");
    }
}
