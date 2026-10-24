package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.decisions.AwaitingDecisionType;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan brainiac-v6 (user 2026-10-10): Brainiac's seat plays Force activation the way its policy was trained. Training
 * always runs the critic-replay activation rule (ActivateOnce): the player's own Activate phase starts with "Choose
 * amount of Force to activate", there is one activation per turn, the opponent has no "allow" cap and there is no
 * "You have not activated Force" check. Stock Gemp offers "Activate Force" as a card-pile action instead. For Brainiac's
 * seat only (the human plays stock rules):
 * <ul>
 *   <li>the first own Activate-phase action choice of a turn takes "Activate Force" itself; the policy then answers the
 *       amount question, exactly the decision it was trained on;</li>
 *   <li>later "Activate Force" actions that turn are hidden from the policy (not encoded, not answerable);</li>
 *   <li>the opponent's activation: Brainiac allows the full amount (no cap, as in training);</li>
 *   <li>"You have not activated Force. Do you want to Pass?" (after an amount of 0): Yes, the policy chose to pass.</li>
 * </ul>
 */
final class ActivationAdapter {
    static final String ACTIVATE = "Activate Force";
    static final String ALLOW = "Choose amount of Force to allow opponent to activate";
    static final String PASS_CHECK = "You have not activated Force. Do you want to Pass?";
    static final AtomicLong ACTIVATIONS = new AtomicLong(), ALLOWED = new AtomicLong(), PASS_CHECKS = new AtomicLong(),
            HIDDEN = new AtomicLong();

    private int activatedTurn = -1;

    /** The adapter's own answer to this decision, or null when the policy decides. */
    String forced(SwccgGame game, String player, AwaitingDecision decision) {
        String text = decision.getText() == null ? "" : decision.getText();
        Map<String, String[]> parameters = decision.getDecisionParameters();
        if (decision.getDecisionType() == AwaitingDecisionType.INTEGER && text.startsWith(ALLOW)) {
            ALLOWED.incrementAndGet();
            return parameters.get("max")[0];
        }
        if (decision.getDecisionType() == AwaitingDecisionType.MULTIPLE_CHOICE && text.equals(PASS_CHECK)) {
            PASS_CHECKS.incrementAndGet();
            return "0";
        }
        if (ownActivatePhase(game, player, decision) && turn(game, player) != activatedTurn) {
            Set<String> ids = activateIds(decision);
            if (!ids.isEmpty()) {
                activatedTurn = turn(game, player);
                ACTIVATIONS.incrementAndGet();
                return ids.iterator().next();
            }
        }
        return null;
    }

    /** Action ids the policy must not see: "Activate Force" after this turn's activation. */
    Set<String> hidden(SwccgGame game, String player, AwaitingDecision decision) {
        if (!ownActivatePhase(game, player, decision) || turn(game, player) != activatedTurn) {
            return Set.of();
        }
        Set<String> ids = activateIds(decision);
        if (!ids.isEmpty()) {
            HIDDEN.incrementAndGet();
        }
        return ids;
    }

    private static boolean ownActivatePhase(SwccgGame game, String player, AwaitingDecision decision) {
        GameState state = game.getGameState();
        return decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE
                && state.getCurrentPhase() == Phase.ACTIVATE && player.equals(state.getCurrentPlayerId());
    }

    private static int turn(SwccgGame game, String player) {
        return game.getGameState().getPlayersLatestTurnNumber(player);
    }

    private static Set<String> activateIds(AwaitingDecision decision) {
        Map<String, String[]> parameters = decision.getDecisionParameters();
        String[] ids = parameters.getOrDefault("actionId", new String[0]);
        String[] texts = parameters.getOrDefault("actionText", new String[ids.length]);
        Set<String> result = new HashSet<>();
        for (int index = 0; index < ids.length && index < texts.length; index++) {
            if (ACTIVATE.equals(texts[index])) {
                result.add(ids[index]);
            }
        }
        return result;
    }
}
