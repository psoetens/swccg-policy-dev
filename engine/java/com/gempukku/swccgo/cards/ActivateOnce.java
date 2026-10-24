package com.gempukku.swccgo.cards;

import com.gempukku.swccgo.game.SwccgGame;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Plan activation-and-cancel, rule change A (both players): the "Activate Force" card-pile action is used at most once
 * per turn and activates the chosen amount without the opponent's "allow" cap; the opponent reacts after the whole
 * activation. Used by the overlay copy of AbstractSwccgCardBlueprint.
 *
 * Plan critic-replay (user 2026-10-08, "always active"): no longer flag-guarded. The own Activate phase starts with the
 * amount question (overlay copy of PlayersPlayPhaseActionsInOrderGameProcess) and any answer, 0 included, is the turn's
 * activation. Plan permanent-play-rules: the property is gone; production Brainiac plays the rule on its own seat
 * (ai.brainiac.ActivationAdapter).
 */
public final class ActivateOnce {
    private static final Map<SwccgGame, Map<String, Integer>> USED = Collections.synchronizedMap(new WeakHashMap<>());

    private ActivateOnce() {
    }

    /** True if {@code player} already used the Activate Force action during its current turn. */
    public static boolean used(SwccgGame game, String player) {
        Map<String, Integer> turns = USED.get(game);
        Integer turn = turns == null ? null : turns.get(player);
        return turn != null && turn == game.getGameState().getPlayersLatestTurnNumber(player);
    }

    public static void mark(SwccgGame game, String player) {
        USED.computeIfAbsent(game, g -> Collections.synchronizedMap(new HashMap<>()))
            .put(player, game.getGameState().getPlayersLatestTurnNumber(player));
    }
}
