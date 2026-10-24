package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;

/**
 * Plan context-overhaul, debug only ({@code -Dpolicy.checkContextV5=<summary.json>}), at every decision:
 * <ol>
 * <li>both-version observations filtered to one version (the tensor builder's {@code contextRows}) equal the
 * observations built for that version alone, actor and critic;</li>
 * <li>every version-5 key is in the vocabulary key list; the actor has no {@code priv.*} row;</li>
 * <li>information boundary: swapping an unseen opponent hand card with a buried Reserve card leaves the actor's
 * version-5 observation byte-identical (same swap as {@link InformationBoundaryCheck}), while the critic's changes.</li>
 * </ol>
 * Any failure throws. The summary counts views, rows seen per family and view, and the swaps checked.
 */
public final class ContextV5Check {
    private static final Set<String> KEYS = new HashSet<>(StoredContextV5.keys());
    private static final AtomicLong VIEWS = new AtomicLong();
    private static final AtomicLong WITH_CONTEXT = new AtomicLong();
    private static final AtomicLong SWAPS = new AtomicLong();
    private static final AtomicLong CRITIC_UNCHANGED = new AtomicLong();
    private static final Map<String, AtomicLong> ROWS = new ConcurrentHashMap<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            String output = System.getProperty("policy.checkContextV5");
            try {
                Files.writeString(Path.of(output), summary());
            } catch (java.io.IOException exception) {
                System.err.println("context v5 check summary write failed: " + exception);
            }
        }, "context-v5-check"));
    }

    private ContextV5Check() {
    }

    private static List<ObservationV2.Token> build(SwccgGame game, String player, boolean privileged, String context) {
        ObservationV2.CONTEXT.set(context);
        try {
            return new ObservationV2(game, player, privileged, 4).tokens;
        } finally {
            ObservationV2.CONTEXT.remove();
        }
    }

    private static String json(List<ObservationV2.Token> tokens) {
        StringJoiner result = new StringJoiner(",", "[", "]");
        for (ObservationV2.Token token : tokens) {
            result.add(token.json());
        }
        return result.toString();
    }

    public static void check(SwccgGame game, String player,
            BiFunction<List<ObservationV2.Token>, Integer, List<ObservationV2.Token>> contextRows) {
        for (boolean privileged : new boolean[]{false, true}) {
            List<ObservationV2.Token> both = build(game, player, privileged, "both");
            List<ObservationV2.Token> five = build(game, player, privileged, "5");
            String view = privileged ? "critic" : "actor";
            for (int version : new int[]{4, 5}) {
                String alone = json(version == 5 ? five : build(game, player, privileged, "4"));
                if (!alone.equals(json(contextRows.apply(both, version)))) {
                    throw new AssertionError("context v" + version + " " + view
                        + ": both-version rows filtered differ from the version alone\n" + alone + "\n"
                        + json(contextRows.apply(both, version)));
                }
            }
            boolean any = false;
            for (ObservationV2.Token token : five) {
                if (token.kind != 7 || !StoredContextV5.owns(token.contextKey)) {
                    continue;
                }
                any = true;
                if (!KEYS.contains(token.contextKey)) {
                    throw new AssertionError("context v5 key outside the vocabulary list: " + token.contextKey);
                }
                if (!privileged && token.contextKey.startsWith("priv.")) {
                    throw new AssertionError("privileged context row in the actor view: " + token.json());
                }
                ROWS.computeIfAbsent(view + "\t" + token.contextKey, ignored -> new AtomicLong()).incrementAndGet();
            }
            if (!privileged) {
                VIEWS.incrementAndGet();
                if (any) {
                    WITH_CONTEXT.incrementAndGet();
                }
            }
        }
        swap(game, player);
    }

    @SuppressWarnings("unchecked")
    private static void swap(SwccgGame game, String viewer) {
        GameState state = game.getGameState();
        String opponent = game.getOpponent(viewer);
        try {
            Field handsField = GameState.class.getDeclaredField("_hands");
            Field reservesField = GameState.class.getDeclaredField("_reserveDecks");
            handsField.setAccessible(true);
            reservesField.setAccessible(true);
            List<PhysicalCard> hand = ((Map<String, List<PhysicalCard>>) handsField.get(state)).get(opponent);
            List<PhysicalCard> reserve = ((Map<String, List<PhysicalCard>>) reservesField.get(state)).get(opponent);
            if (hand.isEmpty() || reserve.size() < 2) {
                return;
            }
            // A card being played is public once announced; swapping it out of the hand is not a valid counterfactual.
            Set<PhysicalCard> played = new HashSet<>();
            for (ObservationV2.Token token : build(game, viewer, true, "5")) {
                if (token.kind == 7 && token.contextKey.startsWith("now.play.") && !token.bp.equals("<pad>")) {
                    for (PhysicalCard card : hand) {
                        if (GameKnowledge.blueprint(card).equals(token.bp)
                                || ObservationV2.strip(card.getBlueprintId(state, false)).equals(token.bp)) {
                            played.add(card);
                        }
                    }
                }
            }
            PhysicalCard a = null;
            for (PhysicalCard card : hand) {
                if (!played.contains(card)) {
                    a = card;
                    break;
                }
            }
            if (a == null) {
                return;
            }
            int handPosition = hand.indexOf(a);
            PhysicalCard b = null;
            int position = -1;
            for (int index = 1; index < reserve.size(); index++) {
                if (!reserve.get(index).isInserted()
                        && !GameKnowledge.blueprint(a).equals(GameKnowledge.blueprint(reserve.get(index)))) {
                    b = reserve.get(index);
                    position = index;
                    break;
                }
            }
            if (b == null) {
                return;
            }
            List<ObservationV2.Token> actorTokens = build(game, viewer, false, "5");
            String actor = json(actorTokens);
            String critic = json(build(game, viewer, true, "5"));
            Zone zoneA = a.getZone();
            Zone zoneB = b.getZone();
            try {
                hand.set(handPosition, b);
                reserve.set(position, a);
                a.setZone(zoneB);
                b.setZone(zoneA);
                List<ObservationV2.Token> swapped = build(game, viewer, false, "5");
                if (!actor.equals(json(swapped))) {
                    throw new AssertionError("context v5 actor view leaked an unseen hand/Reserve swap ("
                        + GameKnowledge.blueprint(a) + " <-> " + GameKnowledge.blueprint(b) + "): "
                        + difference(actorTokens, swapped));
                }
                if (critic.equals(json(build(game, viewer, true, "5")))) {
                    CRITIC_UNCHANGED.incrementAndGet();
                }
                SWAPS.incrementAndGet();
            } finally {
                hand.set(handPosition, a);
                reserve.set(position, b);
                a.setZone(zoneA);
                b.setZone(zoneB);
            }
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(exception);
        }
    }

    private static String difference(List<ObservationV2.Token> before, List<ObservationV2.Token> after) {
        StringJoiner result = new StringJoiner("; ");
        for (int index = 0; index < Math.max(before.size(), after.size()); index++) {
            String left = index < before.size() ? before.get(index).json() : "-";
            String right = index < after.size() ? after.get(index).json() : "-";
            if (!left.equals(right)) {
                result.add(index + ": " + left + " -> " + right);
            }
            if (result.length() > 2000) {
                break;
            }
        }
        return before.size() + "/" + after.size() + " rows; " + result;
    }

    private static String summary() {
        StringJoiner rows = new StringJoiner(",\n  ", "{\n  ", "\n }");
        new TreeMap<>(ROWS).forEach((key, value) -> rows.add(TemplateCoverage.quote(key) + ": " + value.get()));
        return "{\"views\": " + VIEWS.get() + ", \"views_with_v5_context\": " + WITH_CONTEXT.get()
            + ", \"boundary_swaps\": " + SWAPS.get() + ", \"critic_unchanged_by_swap\": " + CRITIC_UNCHANGED.get()
            + ", \"vocabulary_keys\": " + KEYS.size() + ",\n \"rows\": " + rows + "}\n";
    }
}
