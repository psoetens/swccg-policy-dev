package com.gempukku.swccgo.game.knowledge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import static com.gempukku.swccgo.game.knowledge.PileKnowledge.*;

/** Schema-4 bounds in pile/mask-major, scope-major, lower/upper/asserted order. */
public final class KnowledgeGrid {
    public static final int WIDTH = 153;
    public static final int[] MASKS = {3, 5, 6, 7};

    public static int offset(int quantity, int scope) {
        return (quantity * 3 + scope) * 3;
    }

    private static void put(int[] row, int quantity, int scope, int lower, int upper, boolean asserted) {
        if (lower < 0 || upper < lower || upper > 255) {
            throw new IllegalStateException("Grid bounds outside uint8 or empty intersection: " + lower + ":" + upper);
        }
        int start = offset(quantity, scope);
        row[start] = lower;
        row[start + 1] = upper;
        row[start + 2] = asserted ? 1 : 0;
    }

    public static int[] row(String blueprint, PileKnowledge[] scopes) {
        int[] row = new int[WIDTH];
        for (int scope = 0; scope < scopes.length; scope++) {
            PileKnowledge knowledge = scopes[scope];
            knowledge.tighten();
            for (int pile = 0; pile < N; pile++) {
                if (pile == TABLE) {
                    continue;
                }
                Bag bag = knowledge.pile(pile);
                Range range = bag.range(blueprint);
                boolean emitted = bag.cards.containsKey(blueprint) && !(range.hi == 0 && bag.complete());
                put(row, pile, scope, range.lo, range.hi, emitted || bag.complete());
            }
            for (int index = 0; index < MASKS.length; index++) {
                int mask = MASKS[index];
                int defaultLower = 0;
                int defaultUpper = 0;
                for (int pile = 0; pile < 3; pile++) {
                    if ((mask & (1 << pile)) != 0) {
                        Range range = knowledge.pile(pile).range(blueprint);
                        defaultLower += range.lo;
                        defaultUpper += range.hi;
                    }
                }
                int lower = 0;
                int upper = Integer.MAX_VALUE;
                int constraints = 0;
                boolean transit = false;
                for (Map.Entry<Integer, Bag> entry : knowledge.groups.entrySet()) {
                    if ((entry.getKey() & ~(1 << TRANSIT)) != mask) {
                        continue;
                    }
                    Range range = entry.getValue().cards.get(blueprint);
                    if (range == null || range.lo <= defaultLower && range.hi >= defaultUpper) {
                        continue;
                    }
                    if (range.lo < 0 || range.hi > 255) {
                        throw new IllegalStateException("Constraint bounds outside uint8");
                    }
                    lower = Math.max(lower, range.lo);
                    upper = Math.min(upper, range.hi);
                    constraints++;
                    transit |= (entry.getKey() & (1 << TRANSIT)) != 0;
                }
                if (constraints > 1 && transit && knowledge.pile(TRANSIT).size != 0) {
                    throw new IllegalStateException("TRANSIT-bearing constraints merged while Transit is nonempty");
                }
                if (constraints == 0) {
                    lower = defaultLower;
                    upper = defaultUpper;
                }
                put(row, N + index, scope, lower, upper, constraints != 0);
            }
        }
        return row;
    }

    public static void append(ObservationV2 observation, GameKnowledge knowledge, int viewer, boolean privileged) {
        observation.tokens.removeIf(token -> token.kind == 2 || token.kind == 4);
        for (int owner = 0; owner < 2; owner++) {
            PileKnowledge[] scopes = privileged
                ? new PileKnowledge[]{knowledge.views[viewer][owner], knowledge.disclosed[owner],
                    knowledge.views[1 - viewer][owner]}
                : new PileKnowledge[]{knowledge.views[viewer][owner], knowledge.disclosed[owner]};
            Set<String> blueprints = new TreeSet<>();
            for (PileKnowledge scope : scopes) {
                scope.tighten();
                for (Bag bag : scope.groups.values()) {
                    blueprints.addAll(bag.cards.keySet());
                }
            }
            for (String blueprint : blueprints) {
                ObservationV2.Token token = new ObservationV2.Token();
                token.kind = 2;
                token.bp = blueprint;
                token.mine = owner == viewer ? 1 : 0;
                token.slot = observation.kgrid.size();
                token.count = 0;
                observation.kgrid.add(row(blueprint, scopes));
                observation.tokens.add(token);
            }
        }
        validate(observation, privileged);
    }

    public static void validate(ObservationV2 observation, boolean privileged) {
        boolean[] used = new boolean[observation.kgrid.size()];
        Set<String> identities = new TreeSet<>();
        for (ObservationV2.Token token : observation.tokens) {
            if (token.kind == 4) {
                throw new IllegalArgumentException("Schema 4 cannot contain CONSTRAINT rows");
            }
            if (token.kind == 2) {
                if (token.slot < 0 || token.slot >= used.length || used[token.slot]
                        || !identities.add(token.mine + ":" + token.bp)) {
                    throw new IllegalArgumentException("Invalid or duplicate knowledge grid row");
                }
                used[token.slot] = true;
            }
        }
        for (int index = 0; index < used.length; index++) {
            int[] row = observation.kgrid.get(index);
            if (!used[index] || row.length != WIDTH) {
                throw new IllegalArgumentException("Unbound or malformed knowledge grid row");
            }
            for (int cell = 0; cell < WIDTH; cell += 3) {
                if (row[cell] < 0 || row[cell + 1] < row[cell] || row[cell + 1] > 255
                        || row[cell + 2] < 0 || row[cell + 2] > 1) {
                    throw new IllegalArgumentException("Malformed knowledge grid cell");
                }
                if (!privileged && cell / 3 % 3 == 2
                        && (row[cell] != 0 || row[cell + 1] != 0 || row[cell + 2] != 0)) {
                    throw new IllegalArgumentException("Actor grid contains opponent-private facts");
                }
            }
        }
    }

    public static String json(List<int[]> rows) {
        List<String> encoded = new ArrayList<>();
        for (int[] row : rows) {
            encoded.add(Arrays.toString(row));
        }
        return "[" + String.join(",", encoded) + "]";
    }
}
