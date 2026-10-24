package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan actor-v6-inputs (BL-15), debug only ({@code -Dpolicy.checkHandOwners=<summary.json>}): at every decision, counts
 * cards in a hand or stacked whose zone owner differs from their card owner (an opponent's card held by a player).
 * Before the BL-15 fix the knowledge tracker files such a hand card in its owner's hand.
 */
public final class HandOwnerCheck {
    private static final AtomicLong DECISIONS = new AtomicLong();
    private static final Map<String, AtomicLong> DECISIONS_WITH = new ConcurrentHashMap<>();
    private static final Set<Integer> GAMES = ConcurrentHashMap.newKeySet();
    private static final Map<String, Set<Integer>> GAMES_WITH = new ConcurrentHashMap<>();
    private static final Map<String, AtomicLong> CARDS = new ConcurrentHashMap<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                Files.writeString(Path.of(System.getProperty("policy.checkHandOwners")), summary());
            } catch (java.io.IOException exception) {
                System.err.println("hand owner check summary write failed: " + exception);
            }
        }, "hand-owner-check"));
    }

    private HandOwnerCheck() {
    }

    public static void check(SwccgGame game) {
        int id = System.identityHashCode(game);
        GAMES.add(id);
        DECISIONS.incrementAndGet();
        Set<String> found = new java.util.HashSet<>();
        for (PhysicalCard card : game.getGameState().getAllPermanentCards()) {
            int cell = GameKnowledge.cell(card);
            String zoneOwner = card.getZoneOwner();
            if ((cell == PileKnowledge.HAND || cell == PileKnowledge.STACK) && zoneOwner != null
                    && !zoneOwner.equals(card.getOwner())) {
                String kind = cell == PileKnowledge.HAND ? "hand" : "stack";
                found.add(kind);
                CARDS.computeIfAbsent(kind + "\t" + GameKnowledge.blueprint(card), ignored -> new AtomicLong())
                    .incrementAndGet();
            }
        }
        for (String kind : found) {
            DECISIONS_WITH.computeIfAbsent(kind, ignored -> new AtomicLong()).incrementAndGet();
            GAMES_WITH.computeIfAbsent(kind, ignored -> ConcurrentHashMap.newKeySet()).add(id);
        }
    }

    private static String summary() {
        StringJoiner cards = new StringJoiner(", ", "{", "}");
        new TreeMap<>(CARDS).forEach((key, value) -> cards.add(TemplateCoverage.quote(key) + ": " + value.get()));
        StringJoiner kinds = new StringJoiner(", ", "{", "}");
        for (String kind : new String[]{"hand", "stack"}) {
            kinds.add("\"" + kind + "\": {\"decisions\": " + DECISIONS_WITH.getOrDefault(kind, new AtomicLong()).get()
                + ", \"games\": " + GAMES_WITH.getOrDefault(kind, Set.of()).size() + "}");
        }
        return "{\"games\": " + GAMES.size() + ", \"decisions\": " + DECISIONS.get() + ", \"mismatch\": " + kinds
            + ", \"cards\": " + cards + "}\n";
    }
}
