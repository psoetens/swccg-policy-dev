package com.gempukku.swccgo.game.rules;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.decisions.CardsSelectionDecision;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan activation-and-cancel, rule change B: a card selection that offers "click 'Done' to cancel" must pick a card
 * when every selectable card is visible to the deciding player. The cancel stays when any selectable card sits in a zone
 * the decider cannot see (e.g. Reserve Deck, Force Pile, opponent's hand).
 *
 * Plan permanent-play-rules (user 2026-10-10, "for us this is a permanent rule"): always on in training, for every
 * player; production Brainiac applies it to its own seat only. No flag.
 */
public final class VisibleCancel {
    public static final AtomicLong RESTRICTED = new AtomicLong(), KEPT_HIDDEN = new AtomicLong();
    private static final Field MINIMUM, CARDS, PARAMS;
    /** Per decision text: [restricted, kept because a selectable card is hidden]; printed at JVM exit for the report. */
    private static final Map<String, AtomicLong[]> BY_TEXT = new ConcurrentHashMap<>();

    static {
        try {
            MINIMUM = CardsSelectionDecision.class.getDeclaredField("_minimum");
            CARDS = CardsSelectionDecision.class.getDeclaredField("_physicalCards");
            PARAMS = com.gempukku.swccgo.logic.decisions.AbstractAwaitingDecision.class.getDeclaredField("_params");
            MINIMUM.setAccessible(true);
            CARDS.setAccessible(true);
            PARAMS.setAccessible(true);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                StringBuilder b = new StringBuilder("visible-cancel summary (text\trestricted\tkept_hidden):\n");
                BY_TEXT.forEach((text, n) -> b.append(text).append('\t').append(n[0]).append('\t').append(n[1]).append('\n'));
                System.err.print(b);
            }));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private VisibleCancel() {
    }

    private static void count(String text, int slot) {
        // Card names vary per decision; keep only the question wording for the tally.
        String key = text.replaceAll("<[^>]*>", "").replaceAll("\\s+", " ").trim();
        BY_TEXT.computeIfAbsent(key, k -> new AtomicLong[]{new AtomicLong(), new AtomicLong()})[slot].incrementAndGet();
    }

    static boolean visibleTo(PhysicalCard card, String player) {
        Zone zone = card.getZone();
        if (zone == null) return false;
        if (zone.isPublic()) return true;
        return player.equals(card.getOwner()) && zone.isVisibleByOwner();
    }

    /** Raises the minimum of an all-visible "Done to cancel" card selection from 0 to 1, in place. */
    @SuppressWarnings("unchecked")
    public static void restrict(AwaitingDecision decision, String player) {
        if (!(decision instanceof CardsSelectionDecision) || decision.getText() == null
                || !decision.getText().contains("click 'Done' to cancel")) return;
        try {
            if (MINIMUM.getInt(decision) != 0) return;
            List<PhysicalCard> cards = (List<PhysicalCard>) CARDS.get(decision);
            if (cards.isEmpty()) return;
            for (PhysicalCard card : cards) {
                if (!visibleTo(card, player)) {
                    KEPT_HIDDEN.incrementAndGet();
                    count(decision.getText(), 1);
                    return;
                }
            }
            MINIMUM.setInt(decision, 1);
            ((Map<String, String[]>) PARAMS.get(decision)).put("min", new String[]{"1"});
            RESTRICTED.incrementAndGet();
            count(decision.getText(), 0);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }
}
