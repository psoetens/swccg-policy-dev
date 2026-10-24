package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import java.util.Arrays;
import java.util.List;
import com.gempukku.swccgo.game.knowledge.KnowledgeTiming;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;

/**
 * Actor-only production adapter around the shared observation encoder: schema 4 (plan brainiac-v6; the context version
 * is the engine's policy.context, 5 for Brainiac).
 */
public final class KnowledgeStateEncoder {
    public static StateEncoder.Encoded encode(SwccgGame game, String player, AwaitingDecision decision,
            java.util.Set<String> hiddenActions) {
        GameState state = game.getGameState();
        String opponent = game.getOpponent(player);
        if (state.getKnowledge() == null) {
            throw new IllegalStateException("knowledge must be enabled at game initialization; "
                + "current board cannot recover inspection history");
        }
        state.getKnowledge().synchronizeOwners(state);
        if (Boolean.getBoolean("policy.validateKnowledge")) {
            state.getKnowledge().validate(state);
        }
        if (Boolean.getBoolean("policy.checkInformationBoundary")) {
            com.gempukku.swccgo.game.knowledge.InformationBoundaryCheck.check(game, player);
        }
        StateEncoder.Encoded encoded = new StateEncoder.Encoded();
        encoded.version = 4;
        encoded.type = decision.getDecisionType().name();
        encoded.hiddenActions.addAll(hiddenActions);
        long actorStart = KnowledgeTiming.start();
        encoded.observation = new ObservationV2(game, player, false, 4);
        TitleMatcher.References references = new TitleMatcher.References();
        String question = TitleMatcher.shared().template(decision.getText(), references,
            TitleMatcher.permitted(encoded.observation));
        for (int index = 0; index < references.size(); index++) {
            encoded.observation.reference(references, 0, index);
        }
        StateEncoder.encodeOptions(encoded, decision, encoded.observation.byCardId, state);
        String options = encoded.prefix.toString();
        encoded.prefix.setLength(0);
        StringBuilder prefix = encoded.prefix;
        prefix.append(StateEncoder.q(game.getDarkPlayer().equals(player) ? "DS" : "LS")).append(',')
            .append(StateEncoder.q(encoded.type)).append(',').append(StateEncoder.q(question)).append(',');
        StateEncoder.appendStrings(prefix, references);
        prefix.append(',');
        int[] globals = {
            state.getPlayersLatestTurnNumber(player), state.getPlayersLatestTurnNumber(opponent),
            state.getCurrentPhase() == null ? -1 : state.getCurrentPhase().ordinal(),
            player.equals(state.getCurrentPlayerId()) ? 1 : 0,
            state.getPlayerLifeForce(player), state.getPlayerLifeForce(opponent),
            state.getReserveDeckSize(player), state.getReserveDeckSize(opponent),
            state.getForcePileSize(player), state.getForcePileSize(opponent),
            state.getUsedPile(player).size(), state.getUsedPile(opponent).size(),
            state.getLostPile(player).size(), state.getLostPile(opponent).size(),
            state.getHand(player).size(), state.getHand(opponent).size(),
            state.isDuringBattle() ? 1 : 0, state.getForceDrainState() != null ? 1 : 0
        };
        prefix.append(Arrays.toString(globals)).append(',').append(encoded.observation.json())
            .append(',').append(options);
        KnowledgeTiming.end("actor_encode", actorStart);
        for (com.gempukku.swccgo.game.PhysicalCard card : state.getHand(player)) {
            encoded.hand.add(new String[]{card.getBlueprintId(true), card.getBlueprint().getTitle()});
        }
        return encoded;
    }
}
