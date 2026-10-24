package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.actions.AbstractAction;
import com.gempukku.swccgo.logic.actions.PlayCardAction;
import com.gempukku.swccgo.logic.effects.RespondablePlayingCardEffect;
import com.gempukku.swccgo.logic.modifiers.LimitCounter;
import com.gempukku.swccgo.logic.modifiers.ModifierType;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersLogic;
import com.gempukku.swccgo.logic.timing.Action;
import com.gempukku.swccgo.logic.timing.Effect;
import com.gempukku.swccgo.logic.timing.EffectResult;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Plan context-overhaul: the context version 5 rows. Typed keys from closed engine enumerations, each linked to a card
 * token where one exists; no reflection dumps and no per-card keys. Reads stored fields and plain getters only (the
 * action stack through its private fields), never rule evaluation.
 * <ul>
 * <li>actor and critic: {@code now.action.<Action.Type>}, {@code now.play.<Zone>}, {@code now.target},
 * {@code now.respond.<Effect.Type>}, {@code now.result.<EffectResult.Type>}, {@code mod.<ModifierType>}
 * (index a = the duration container);</li>
 * <li>critic only: {@code priv.target}, {@code priv.limit.<counter>}, {@code priv.card.whileInPlay},
 * {@code priv.card.remainder}.</li>
 * </ul>
 * The actor links only cards it can see; a hidden card never becomes a link, a blueprint or a token of its own.
 */
public final class StoredContextV5 {
    public static final int VERSION = 5;
    /** Duration containers of {@code ModifiersLogic}, in a fixed order (the index of a {@code mod.*} row). */
    public static final List<String> DURATIONS = durations();
    public static final List<String> LIMITS = limits();
    private static final Field ENVIRONMENT_STACK = field("com.gempukku.swccgo.game.state.actions.DefaultActionsEnvironment",
        "_actionStack");
    private static final Field STACK = field("com.gempukku.swccgo.logic.timing.ActionStack", "_actionStack");
    private static final Field PLAY_OUT_EFFECT = field("com.gempukku.swccgo.logic.timing.TurnProcedure$PlayOutEffect",
        "_effect");
    private static final Field EFFECT_RESULTS = field("com.gempukku.swccgo.logic.timing.PlayOutEffectResults",
        "_effectResults");

    private StoredContextV5() {
    }

    private static Field field(String type, String name) {
        try {
            Field field = Class.forName(type).getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException("context v5 needs " + type + "." + name, exception);
        }
    }

    private static List<String> durations() {
        List<String> result = new ArrayList<>();
        for (Field field : ModifiersLogic.class.getDeclaredFields()) {
            String name = field.getName();
            if ((name.startsWith("_untilEndOf") || name.startsWith("_untilStartOf")) && name.endsWith("Modifiers")) {
                result.add(name);
            }
        }
        result.sort(Comparator.naturalOrder());
        return List.copyOf(result);
    }

    private static List<String> limits() {
        List<String> result = new ArrayList<>();
        for (Field field : ModifiersLogic.class.getDeclaredFields()) {
            if (field.getName().endsWith("LimitCounters")) {
                result.add(limitName(field.getName()));
            }
        }
        result.sort(Comparator.naturalOrder());
        return List.copyOf(result);
    }

    private static String limitName(String field) {
        return field.substring(1, field.length() - "LimitCounters".length());
    }

    /** True for a key of the version-5 families (battle.* and drain.* are shared by both versions). */
    public static boolean owns(String key) {
        return key.startsWith("now.") || key.startsWith("mod.") || key.startsWith("priv.");
    }

    /** True if a context row with this key belongs to the given context version's vocabulary. */
    public static boolean belongs(String key, int version) {
        boolean shared = key.startsWith("battle.") || key.startsWith("drain.");
        return shared || (version == VERSION) == owns(key);
    }

    /** Every key this class can emit (the vocabulary's context family is built from this list). */
    public static List<String> keys() {
        TreeSet<String> keys = new TreeSet<>();
        for (Action.Type type : Action.Type.values()) {
            keys.add("now.action." + type.name());
        }
        for (Zone zone : Zone.values()) {
            keys.add("now.play." + zone.name());
        }
        keys.add("now.target");
        for (Effect.Type type : Effect.Type.values()) {
            keys.add("now.respond." + type.name());
        }
        for (EffectResult.Type type : EffectResult.Type.values()) {
            keys.add("now.result." + type.name());
        }
        for (ModifierType type : ModifierType.values()) {
            keys.add("mod." + type.name());
        }
        keys.add("priv.target");
        for (String limit : LIMITS) {
            keys.add("priv.limit." + limit);
        }
        keys.add("priv.card.whileInPlay");
        keys.add("priv.card.remainder");
        return new ArrayList<>(keys);
    }

    /** Writes {@link #keys()} as a JSON list to the file given (the vocabulary builder reads it). */
    public static void main(String[] args) throws java.io.IOException {
        java.util.StringJoiner json = new java.util.StringJoiner(",\n ", "[", "]\n");
        for (String key : keys()) {
            json.add(TemplateCoverage.quote(key));
        }
        java.nio.file.Files.writeString(java.nio.file.Path.of(args[0]), json.toString());
        System.out.println("context v5 keys: " + keys().size() + " (durations " + DURATIONS.size() + ", limits "
            + LIMITS.size() + ")");
    }

    /** The actor's visibility rule of {@link ObservationV2}: a card this player may see on the table or in hand. */
    static boolean visible(PhysicalCard card, String player, GameState state) {
        Zone zone = card.getZone();
        return zone != null && !card.isNotShownOnUserInterface()
            && (zone.isPublic() || player.equals(card.getOwner()) && zone.isVisibleByOwner())
            && !card.getBlueprintId(state, false).startsWith("-1_");
    }

    @SuppressWarnings("unchecked")
    private static List<Action> stack(SwccgGame game) {
        if (!ENVIRONMENT_STACK.getDeclaringClass().isInstance(game.getActionsEnvironment())) {
            throw new IllegalStateException("context v5 needs DefaultActionsEnvironment");
        }
        try {
            Object stack = ENVIRONMENT_STACK.get(game.getActionsEnvironment());
            return stack == null ? List.of() : new ArrayList<>((Collection<Action>) STACK.get(stack));
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException(exception);
        }
    }

    public static void append(ObservationV2 observation, SwccgGame game, String player, boolean privileged) {
        GameState state = game.getGameState();
        List<Action> actions = stack(game);
        for (int index = 0; index < actions.size(); index++) {
            // Position 0 is the top of the stack (the action resolving now).
            Action action = actions.get(actions.size() - 1 - index);
            int[] position = {index};
            if (PLAY_OUT_EFFECT.getDeclaringClass().isInstance(action)) {
                respond(observation, state, player, privileged, read(PLAY_OUT_EFFECT, action), position);
                continue;
            }
            if (EFFECT_RESULTS.getDeclaringClass().isInstance(action)) {
                results(observation, state, read(EFFECT_RESULTS, action), position);
                continue;
            }
            if (action.getType() == null || action.getType() == Action.Type.GAME_PROCESS) {
                continue;
            }
            PhysicalCard source = action.getActionSource();
            link(observation, state, player, privileged, "now.action." + action.getType().name(),
                action.getPerformingPlayer(), source, 1, position);
            if (action instanceof PlayCardAction play && play.getPlayedCard() != null) {
                Zone from = play.getPlayingFromZone();
                if (from != null) {
                    // The played card becomes public when the play is announced (targets chosen); before that the
                    // opponent's actor sees only that a card is being played from that zone. The owner always knows
                    // its own card, also while it sits in a zone the visibility rule does not show (e.g. VOID).
                    boolean announced = action instanceof AbstractAction abstractAction
                        && abstractAction.isChoosingTargetsComplete();
                    PhysicalCard played = play.getPlayedCard();
                    boolean shown = privileged || announced || visible(played, player, state)
                        || player.equals(played.getOwner());
                    ObservationV2.Token token = observation.contextV4("now.play." + from.name(),
                        action.getPerformingPlayer(), shown ? played : null, 1, position);
                    // The critic and the owner know the true card, also in a hidden pile (a location deployed from
                    // Reserve Deck); the opponent's actor gets the public identity only.
                    String blueprint = privileged || player.equals(played.getOwner())
                        ? GameKnowledge.blueprint(played) : played.getBlueprintId(state, false);
                    if (shown && !blueprint.startsWith("-1_")) {
                        token.bp = ObservationV2.strip(blueprint);
                    }
                }
            }
            if (action instanceof AbstractAction targeting) {
                targets(observation, state, player, privileged, targeting, index);
            }
        }
        if (game.getModifiersEnvironment() instanceof ModifiersLogic modifiers) {
            Map<String, Object> stored = modifiers.knowledgeStoredContext();
            for (int duration = 0; duration < DURATIONS.size(); duration++) {
                List<com.gempukku.swccgo.logic.modifiers.Modifier> found = new ArrayList<>();
                collect(stored.get(DURATIONS.get(duration)), found, 0);
                for (com.gempukku.swccgo.logic.modifiers.Modifier modifier : found) {
                    if (modifier.getModifierType() == null) {
                        continue;
                    }
                    link(observation, state, player, privileged, "mod." + modifier.getModifierType().name(),
                        modifier.getForPlayer(), modifier.getSource(state), 1, new int[]{duration});
                }
            }
            if (privileged) {
                for (Map.Entry<String, Object> entry : stored.entrySet()) {
                    if (entry.getKey().endsWith("LimitCounters")) {
                        limits(observation, state, "priv.limit." + limitName(entry.getKey()), entry.getValue());
                    }
                }
            }
        }
        if (privileged) {
            for (PhysicalCard card : state.getAllPermanentCards()) {
                if (card.getWhileInPlayData() != null) {
                    observation.contextV4("priv.card.whileInPlay", card.getOwner(), card, 1, new int[0]);
                }
                if (card.getForRemainderOfGameData() != null && !card.getForRemainderOfGameData().isEmpty()) {
                    observation.contextV4("priv.card.remainder", card.getOwner(), card, 1, new int[0]);
                }
            }
        }
    }

    private static Object read(Field field, Object owner) {
        try {
            return field.get(owner);
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException(exception);
        }
    }

    /** One row; for the actor, a card it cannot see is dropped from the row (the row itself stays). */
    private static void link(ObservationV2 observation, GameState state, String player, boolean privileged,
            String key, String owner, PhysicalCard source, float value, int[] indices) {
        PhysicalCard shown = source == null || privileged || visible(source, player, state) ? source : null;
        observation.contextV4(key, owner, shown, value, indices);
    }

    private static void respond(ObservationV2 observation, GameState state, String player, boolean privileged,
            Object value, int[] position) {
        if (!(value instanceof Effect effect) || effect.getType() == null) {
            return;
        }
        Action cause = effect.getAction();
        PhysicalCard source = effect instanceof RespondablePlayingCardEffect playing ? playing.getCard()
            : cause == null ? null : cause.getActionSource();
        link(observation, state, player, privileged, "now.respond." + effect.getType().name(),
            cause == null ? null : cause.getPerformingPlayer(), source, 1, position);
    }

    private static void results(ObservationV2 observation, GameState state, Object value, int[] position) {
        if (!(value instanceof Collection<?> results)) {
            return;
        }
        List<EffectResult> sorted = new ArrayList<>();
        for (Object result : results) {
            if (result instanceof EffectResult effectResult && effectResult.getType() != null) {
                sorted.add(effectResult);
            }
        }
        // A hash set: a fixed order keeps the observation deterministic.
        sorted.sort(Comparator.comparing((EffectResult result) -> result.getType().name())
            .thenComparing(result -> String.valueOf(result.getPerformingPlayerId())));
        for (EffectResult result : sorted) {
            observation.contextV4("now.result." + result.getType().name(), result.getPerformingPlayerId(), null, 1,
                position);
        }
    }

    private static void targets(ObservationV2 observation, GameState state, String player, boolean privileged,
            AbstractAction action, int index) {
        Map<Integer, ? extends Map<PhysicalCard, ?>> groups = action.getAllPrimaryTargetCards();
        if (groups == null || groups.isEmpty()) {
            return;
        }
        boolean announced = action.isChoosingTargetsComplete();
        for (Integer group : new TreeSet<>(groups.keySet())) {
            List<PhysicalCard> cards = new ArrayList<>(groups.get(group).keySet());
            cards.sort(Comparator.comparingInt(PhysicalCard::getPermanentCardId));
            for (PhysicalCard card : cards) {
                boolean shown = announced && visible(card, player, state);
                if (shown) {
                    observation.contextV4("now.target", action.getPerformingPlayer(), card, 1,
                        new int[]{index, group});
                } else if (privileged) {
                    observation.contextV4("priv.target", action.getPerformingPlayer(), card, announced ? 1 : 0,
                        new int[]{index, group});
                }
            }
        }
    }

    private static void collect(Object value, List<com.gempukku.swccgo.logic.modifiers.Modifier> found, int depth) {
        if (value == null || depth > 3) {
            return;
        }
        if (value instanceof com.gempukku.swccgo.logic.modifiers.Modifier modifier) {
            found.add(modifier);
        } else if (value instanceof Map<?, ?> map) {
            for (Object item : map.values()) {
                collect(item, found, depth + 1);
            }
        } else if (value instanceof Collection<?> collection) {
            for (Object item : collection) {
                collect(item, found, depth + 1);
            }
        }
    }

    private static void limits(ObservationV2 observation, GameState state, String key, Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return;
        }
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getValue() instanceof Map<?, ?>) {
                limits(observation, state, key, entry.getValue());
                continue;
            }
            if (!(entry.getValue() instanceof LimitCounter counter) || counter.getUsedLimit() == 0) {
                continue;
            }
            String[] pieces = String.valueOf(entry.getKey()).split("\\|", -1);
            PhysicalCard card = null;
            String owner = null;
            if (pieces.length == 4) {
                owner = pieces[1];
                try {
                    card = state.findCardById(Integer.parseInt(pieces[0]));
                } catch (NumberFormatException ignored) {
                    // Title-keyed counter: no card instance.
                }
            }
            observation.contextV4(key, owner, card, counter.getUsedLimit(), new int[0]);
        }
    }
}
