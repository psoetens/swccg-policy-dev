package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.filters.Filter;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.modifiers.LimitCounter;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersLogic;
import com.gempukku.swccgo.logic.timing.Action;
import com.gempukku.swccgo.logic.timing.Effect;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/** Stored-field-only schema-4 reflection. Never evaluates callbacks or modifiers. */
public final class StoredContextV4 {
    public static final int ALIAS = 8;
    public static final int STRUCTURAL = 16;
    private static final Map<String, LongAdder> COVERAGE = new ConcurrentHashMap<>();
    private static final Set<String> PACKAGES = Set.of(
        "com.gempukku.swccgo.game.state", "com.gempukku.swccgo.logic.modifiers",
        "com.gempukku.swccgo.logic.evaluators", "com.gempukku.swccgo.logic.conditions",
        "com.gempukku.swccgo.game.state.actions", "com.gempukku.swccgo.cards.evaluators"
    );
    private static final ClassValue<Field[]> FIELDS = new ClassValue<>() {
        @Override
        protected Field[] computeValue(Class<?> type) {
            List<Field> fields = new ArrayList<>();
            for (Class<?> current = type; current != null && current != Object.class;
                    current = current.getSuperclass()) {
                for (Field field : current.getDeclaredFields()) {
                    if (Modifier.isStatic(field.getModifiers()) || field.getName().startsWith("this$")) {
                        continue;
                    }
                    if (Set.of("_game", "_swccgGame", "_gameState", "_action").contains(field.getName())) {
                        continue;
                    }
                    field.setAccessible(true);
                    fields.add(field);
                }
            }
            fields.sort(Comparator.comparing(Field::getName));
            return fields.toArray(new Field[0]);
        }
    };

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            String output = System.getProperty("policy.contextCoverageV4");
            if (output == null) {
                return;
            }
            try {
                Files.writeString(Path.of(output), coverageJson());
            } catch (java.io.IOException exception) {
                System.err.println("schema-4 context coverage write failed: " + exception);
            }
        }, "knowledge-v4-context-coverage"));
    }

    private static final class Anchor {
        final ObservationV2.Token token;
        boolean referenced;
        boolean keep;
        boolean meaningful;
        int end;

        Anchor(ObservationV2.Token token) {
            this.token = token;
        }
    }

    private final ObservationV2 observation;
    private final GameState state;
    private final IdentityHashMap<Object, Anchor> seen = new IdentityHashMap<>();
    private final IdentityHashMap<Object, Boolean> emptySeen = new IdentityHashMap<>();
    private final IdentityHashMap<ObservationV2.Token, Anchor> anchors = new IdentityHashMap<>();
    private final IdentityHashMap<ObservationV2.Token, Anchor> aliases = new IdentityHashMap<>();
    private final int start;

    private StoredContextV4(ObservationV2 observation, GameState state) {
        this.observation = observation;
        this.state = state;
        this.start = observation.tokens.size();
    }

    public static void append(ObservationV2 observation, SwccgGame game, List<PhysicalCard> cards) {
        StoredContextV4 context = new StoredContextV4(observation, game.getGameState());
        for (PhysicalCard card : cards) {
            context.value("card.whileInPlay", card.getWhileInPlayData(), card, 0, new int[0]);
            context.value("card.remainder", card.getForRemainderOfGameData(), card, 0, new int[0]);
        }
        context.value("states", context.state.knowledgeStoredStates(), null, 0, new int[0]);
        if (game.getModifiersEnvironment() instanceof ModifiersLogic modifiers) {
            context.value("rules", modifiers.knowledgeStoredContext(), null, 0, new int[0]);
        }
        context.finish();
    }

    /** Test seam also used by coverage checks; shares the exact production walker. */
    public static void capture(ObservationV2 observation, GameState state, String path, Object value) {
        StoredContextV4 context = new StoredContextV4(observation, state);
        context.value(path, value, null, 0, new int[0]);
        context.finish();
    }

    public static void appendPublicCounters(ObservationV2 observation, SwccgGame game, String viewer,
            List<PhysicalCard> visible) {
        if (!(game.getModifiersEnvironment() instanceof ModifiersLogic modifiers)) {
            return;
        }
        Set<String> titles = new HashSet<>();
        Set<Integer> ids = new HashSet<>();
        for (PhysicalCard card : visible) {
            if (!card.getBlueprintId(game.getGameState(), false).startsWith("-1_")) {
                titles.add(card.getBlueprint().getTitle());
                ids.add(card.getCardId());
            }
        }
        for (Map.Entry<String, Object> entry : modifiers.knowledgeStoredContext().entrySet()) {
            if (entry.getKey().endsWith("LimitCounters")) {
                publicCounter(observation, game.getGameState(), viewer, entry.getKey(), entry.getValue(), titles, ids);
            }
        }
    }

    private static void publicCounter(ObservationV2 observation, GameState state, String viewer,
            String path, Object value, Set<String> titles, Set<Integer> ids) {
        if (!(value instanceof Map<?, ?> map)) {
            return;
        }
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getValue() instanceof Map<?, ?>) {
                String key = entry.getKey() instanceof Enum<?> ? entry.getKey().toString() : "active";
                publicCounter(observation, state, viewer, path + "." + key, entry.getValue(), titles, ids);
                continue;
            }
            if (!(entry.getValue() instanceof LimitCounter counter) || counter.getUsedLimit() == 0) {
                continue;
            }
            String key = String.valueOf(entry.getKey());
            String[] pieces = key.split("\\|", -1);
            PhysicalCard card = null;
            String owner = null;
            if (pieces.length == 4) {
                owner = pieces[1];
                try {
                    int id = Integer.parseInt(pieces[0]);
                    if (!ids.contains(id)) {
                        continue;
                    }
                    card = state.findCardById(id);
                } catch (NumberFormatException exception) {
                    if (!viewer.equals(owner)) {
                        continue;
                    }
                }
                key = pieces[3];
            } else if (!titles.contains(pieces[0])) {
                continue;
            }
            ObservationV2.Token token = observation.contextV4(
                "usage." + path + "." + key, owner, card, counter.getUsedLimit(), new int[0]
            );
            observation.contextCounters.put(token, counter);
        }
    }

    private static String counterFamily(ObservationV2.Token token) {
        String key = token.contextKey.substring(token.contextKey.indexOf('.') + 1);
        return key.substring(0, key.indexOf("LimitCounters") + "LimitCounters".length());
    }

    private static String counterAction(ObservationV2.Token token) {
        String key = token.contextKey;
        int start = key.indexOf("LimitCounters") + "LimitCounters".length();
        String suffix = key.substring(start);
        int last = Math.max(suffix.lastIndexOf('.'), suffix.lastIndexOf('|'));
        return suffix.substring(last + 1);
    }

    public static void verifyCounterTwin(ObservationV2.Token removed, ObservationV2.Token twin) {
        if (!removed.contextKey.startsWith("usage.") || !twin.contextKey.startsWith("rules.")
                || !counterFamily(removed).equals(counterFamily(twin))
                || !counterAction(removed).equals(counterAction(twin))
                || !removed.bp.equals(twin.bp) || removed.parent != twin.parent
                || removed.b != twin.b || removed.flags != twin.flags) {
            throw new IllegalStateException("Duplicate counter has no exact rules twin: " + removed.json()
                + " / " + twin.json());
        }
    }

    public static ObservationV2.Token restoreDefault(ObservationV2.ContextDefault omission) {
        Field reflected = Arrays.stream(FIELDS.get(omission.type()))
            .filter(field -> field.getName().equals(omission.field())).findFirst().orElseThrow();
        String suffix = ".type=" + stableName(omission.type());
        String ownerKey = omission.owner().contextKey;
        ObservationV2.Token token = omission.omitted();
        if (Modifier.isStatic(reflected.getModifiers()) || !ownerKey.endsWith(suffix)
                || !token.contextKey.equals(ownerKey.substring(0, ownerKey.length() - suffix.length())
                    + "." + reflected.getName()) || token.b != 0 || token.flags != 0) {
            throw new IllegalStateException("Unjustified reflected default omission: " + token.json());
        }
        return token;
    }

    public static void removeDuplicateCounters(ObservationV2 observation) {
        IdentityHashMap<Object, List<ObservationV2.Token>> twins = new IdentityHashMap<>();
        observation.contextCounters.forEach((token, counter) -> {
            if (token.contextKey.startsWith("rules.")) {
                twins.computeIfAbsent(counter, ignored -> new ArrayList<>()).add(token);
            }
        });
        Set<ObservationV2.Token> removed = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (ObservationV2.Token token : observation.tokens) {
            if (token.kind != 7 || !token.contextKey.startsWith("usage.")) {
                continue;
            }
            List<ObservationV2.Token> candidates = twins.get(observation.contextCounters.get(token));
            if (candidates == null) {
                continue;
            }
            ObservationV2.Token twin = candidates.stream().filter(candidate ->
                counterFamily(token).equals(counterFamily(candidate))
                    && counterAction(token).equals(counterAction(candidate))
                    && token.bp.equals(candidate.bp) && token.parent == candidate.parent
                    && token.b == candidate.b && token.flags == candidate.flags
            ).findFirst().orElse(null);
            if (twin == null) {
                continue;
            }
            verifyCounterTwin(token, twin);
            observation.contextDuplicates.add(new ObservationV2.ContextDuplicate(token, twin));
            removed.add(token);
        }
        int[] positions = new int[observation.tokens.size()];
        int next = 0;
        for (int index = 0; index < positions.length; index++) {
            positions[index] = removed.contains(observation.tokens.get(index)) ? -1 : next++;
        }
        for (ObservationV2.Token token : observation.tokens) {
            if (token.kind == 7 && token.parent >= 0) {
                int mapped = positions[token.parent];
                if (mapped < 0) {
                    throw new IllegalStateException("A removed counter is the target of a context reference");
                }
                token.parent = mapped;
            }
        }
        observation.tokens.removeIf(removed::contains);
        count("removed.duplicate_counter", removed.size());
    }

    private static void count(String key) {
        count(key, 1);
    }

    private static void count(String key, long amount) {
        COVERAGE.computeIfAbsent(key, ignored -> new LongAdder()).add(amount);
    }

    public static String coverageJson() {
        StringJoiner result = new StringJoiner(",\n", "{\n", "\n}\n");
        new TreeMap<>(COVERAGE).forEach((key, value) -> result.add(TemplateCoverage.quote(key) + ":" + value.sum()));
        return result.toString();
    }

    public static String stableName(Class<?> type) {
        if (type.isArray()) {
            return stableName(type.getComponentType()) + "[]";
        }
        if (!type.isAnonymousClass() && !type.isLocalClass() && !type.isSynthetic()) {
            return type.getName();
        }
        Method method = null;
        Class<?> enclosing = type;
        while (enclosing.isAnonymousClass() || enclosing.isLocalClass() || enclosing.isSynthetic()) {
            if (method == null) {
                method = enclosing.getEnclosingMethod();
            }
            Class<?> parent = enclosing.getEnclosingClass();
            if (parent == null) {
                String name = enclosing.getName().split("\\$\\$Lambda", 2)[0];
                try {
                    Class<?> owner = Class.forName(name, false, type.getClassLoader());
                    while (owner.isAnonymousClass() || owner.isLocalClass()) {
                        owner = owner.getEnclosingClass();
                    }
                    name = owner.getName();
                } catch (ClassNotFoundException exception) {
                    throw new IllegalStateException("Cannot resolve synthetic enclosing class " + name, exception);
                }
                return name + "#lambda";
            }
            enclosing = parent;
        }
        String suffix = type.isSynthetic() ? "lambda" : method == null ? "initializer" : method.getName();
        return enclosing.getName() + "#" + suffix;
    }

    private String name(Class<?> type) {
        String name = stableName(type);
        if (!name.equals(type.getName())) {
            count("collapsed." + name);
        }
        return name;
    }

    private String key(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof PhysicalCard) {
            return "card";
        }
        String text = value instanceof String || value instanceof Number || value instanceof Enum<?>
            ? String.valueOf(value) : name(value.getClass());
        if (text.equals(state.getDarkPlayer())) {
            return "DS";
        }
        if (text.equals(state.getLightPlayer())) {
            return "LS";
        }
        return text;
    }

    private ObservationV2.Token emit(String path, PhysicalCard source, float value, int[] indices) {
        return observation.contextV4(path, null, source, value, indices);
    }

    private static boolean empty(Object value) {
        return value instanceof Map<?, ?> map && map.isEmpty()
            || value instanceof Collection<?> collection && collection.isEmpty()
            || value.getClass().isArray() && Array.getLength(value) == 0;
    }

    private static boolean meaningful(Object value) {
        return value instanceof Action || value instanceof Effect || value instanceof Filter
            || PACKAGES.contains(value.getClass().getPackageName());
    }

    private void opaque(String path, Object value, PhysicalCard source, int[] indices) {
        String name = name(value.getClass());
        count("opaque." + name);
        emit(path + ".opaque=" + name, source, 1, indices);
    }

    private static int[] position(int[] indices, int index, String path) {
        if (indices.length == 4) {
            throw new IllegalStateException("schema-4 context path exceeds four list-index levels: "
                + path + ".N " + Arrays.toString(indices) + " + " + index);
        }
        int[] result = Arrays.copyOf(indices, indices.length + 1);
        result[indices.length] = index;
        return result;
    }

    private void value(String path, Object value, PhysicalCard source, int depth, int[] indices) {
        if (value == null) {
            return;
        }
        if (empty(value)) {
            // V3 treats primitive arrays as opaque at every depth; other empty
            // containers become opaque only beyond its depth limit. A1 removes
            // those markers too; retain audit-only provenance.
            boolean primitiveArray = value.getClass().isArray() && !(value instanceof Object[]);
            if (emptySeen.put(value, true) == null && (depth > 8 || primitiveArray)
                    && System.getProperty("policy.checkContextV4") != null) {
                ObservationV2.Token omitted = emit(path + ".opaque=" + name(value.getClass()), source, 1, indices);
                observation.tokens.remove(observation.tokens.size() - 1);
                observation.contextOmissions.add(omitted);
                count("removed.empty_opaque");
            }
            return;
        }
        if (value instanceof GameState || value instanceof SwccgGame
                || value instanceof java.util.random.RandomGenerator) {
            opaque(path, value, source, indices);
            return;
        }
        if (value instanceof LimitCounter counter) {
            ObservationV2.Token token = emit(path, source, counter.getUsedLimit(), indices);
            observation.contextCounters.put(token, counter);
            return;
        }
        if (value instanceof Number number) {
            emit(path, source, number.floatValue(), indices);
            return;
        }
        if (value instanceof Boolean flag) {
            emit(path, source, flag ? 1 : 0, indices);
            return;
        }
        if (value instanceof String || value instanceof Enum<?>) {
            emit(path + "=" + key(value), source, 1, indices);
            return;
        }
        if (value instanceof PhysicalCard card) {
            emit(path, card, 1, indices);
            return;
        }
        boolean meaningful = meaningful(value);
        ObservationV2.Token token = emit(path + (meaningful ? ".type=" + name(value.getClass()) : ".object"),
            source, 1, indices);
        if (!meaningful) {
            token.flags |= STRUCTURAL;
        }
        Anchor anchor = new Anchor(token);
        anchor.meaningful = meaningful;
        anchors.put(token, anchor);
        Anchor previous = seen.get(value);
        if (previous != null) {
            previous.referenced = true;
            anchor.keep = true;
            ObservationV2.Token alias = emit(path + ".alias", null, 1, indices);
            alias.flags = ALIAS;
            aliases.put(alias, previous);
            anchor.end = observation.tokens.size();
            return;
        }
        seen.put(value, anchor);
        if (depth > 8) {
            opaque(path, value, source, indices);
        } else if (value instanceof Map<?, ?> map) {
            map(path, map, source, depth, indices);
        } else if (value instanceof Collection<?> collection) {
            int index = 0;
            for (Object item : collection) {
                boolean ordered = value instanceof List<?>;
                value(path + (ordered ? ".N" : ".member"), item, source, depth + 1,
                    ordered ? position(indices, index++, path) : indices);
            }
        } else if (value instanceof Object[] array) {
            for (int index = 0; index < array.length; index++) {
                value(path + ".N", array[index], source, depth + 1, position(indices, index, path));
            }
        } else if (meaningful) {
            for (Field field : FIELDS.get(value.getClass())) {
                try {
                    Object stored = field.get(value);
                    if (stored instanceof Number number && field.getName().equals("_permCardId")) {
                        stored = state.findCardByPermanentId(number.intValue());
                    }
                    if (Boolean.FALSE.equals(stored) || stored instanceof Number number
                            && number.doubleValue() == 0) {
                        anchor.keep = true;
                        count("removed.default_field." + name(value.getClass()) + "." + field.getName());
                        if (System.getProperty("policy.checkContextV4") != null) {
                            ObservationV2.Token omitted = emit(path + "." + field.getName(), source, 0, indices);
                            observation.tokens.remove(observation.tokens.size() - 1);
                            observation.contextDefaults.add(new ObservationV2.ContextDefault(
                                value.getClass(), field.getName(), token, omitted
                            ));
                        }
                        continue;
                    }
                    value(path + "." + field.getName(), stored, source, depth + 1, indices);
                } catch (IllegalAccessException exception) {
                    throw new IllegalStateException(exception);
                }
            }
        } else {
            opaque(path, value, source, indices);
        }
        anchor.end = observation.tokens.size();
    }

    private void map(String path, Map<?, ?> map, PhysicalCard source, int depth, int[] indices) {
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            Object mapKey = entry.getKey();
            String name = key(mapKey);
            PhysicalCard link = mapKey instanceof PhysicalCard card ? card : source;
            if (mapKey instanceof Integer number) {
                PhysicalCard card = state.findCardById(number);
                if (card == null) {
                    card = state.findCardByPermanentId(number);
                }
                if (card != null) {
                    if (System.getProperty("policy.checkContextV4") != null) {
                        observation.contextMapKeys.add(new ObservationV2.ContextMapKey(
                            path + "." + name, path + ".card", indices.clone(),
                            observation.byCardId.getOrDefault(card.getCardId(), -1)
                        ));
                    }
                    link = card;
                    name = "card";
                } else {
                    count("numeric_map_key." + path);
                }
            } else if (mapKey instanceof Number) {
                count("numeric_map_key." + path);
            }
            if (entry.getValue() instanceof LimitCounter counter) {
                if (counter.getUsedLimit() == 0) {
                    continue;
                }
                String[] pieces = name.split("\\|", -1);
                if (pieces.length == 4) {
                    try {
                        link = state.findCardById(Integer.parseInt(pieces[0]));
                    } catch (NumberFormatException ignored) {
                        // Same non-instance counter convention as the frozen walker.
                    }
                    name = key(pieces[1]) + "." + pieces[3];
                }
            }
            value(path + "." + name, entry.getValue(), link, depth + 1, indices);
        }
    }

    private void finish() {
        int[] retained = new int[observation.tokens.size() + 1];
        for (int index = observation.tokens.size() - 1; index >= start; index--) {
            Anchor anchor = anchors.get(observation.tokens.get(index));
            boolean keep = anchor == null || anchor.keep || anchor.referenced
                || anchor.meaningful && retained[index + 1] > retained[anchor.end];
            retained[index] = retained[index + 1] + (keep ? 1 : 0);
        }
        List<ObservationV2.Token> kept = new ArrayList<>(observation.tokens.subList(0, start));
        IdentityHashMap<ObservationV2.Token, Integer> positions = new IdentityHashMap<>();
        for (int index = start; index < observation.tokens.size(); index++) {
            ObservationV2.Token token = observation.tokens.get(index);
            if (retained[index] > retained[index + 1]) {
                positions.put(token, kept.size());
                kept.add(token);
            }
        }
        aliases.forEach((token, target) -> token.parent = positions.get(target.token));
        observation.tokens.clear();
        observation.tokens.addAll(kept);
    }
}
