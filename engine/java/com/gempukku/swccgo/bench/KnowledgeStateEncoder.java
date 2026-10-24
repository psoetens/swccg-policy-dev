package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import java.util.Arrays;
import com.gempukku.swccgo.game.knowledge.KnowledgeTiming;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;

/** Training harness adapter around the shared production observation encoder. Schema 4 (V3.1) only: plan
 * remove-old-generations removed the schema-2 / 3 paths and their parity / context-preservation checks. */
public final class KnowledgeStateEncoder {
    public static StateEncoder.Encoded encode(SwccgGame game, String player, AwaitingDecision decision) {
        return encode(game, player, decision, Integer.getInteger("policy.schema", 4), true);
    }
    static StateEncoder.Encoded comparison(SwccgGame game, String player, AwaitingDecision decision, int schema) {
        long start = KnowledgeTiming.start();
        var result = encode(game, player, decision, schema, false);
        KnowledgeTiming.end("paired_encode_v" + schema, start);
        return result;
    }
    private static StateEncoder.Encoded encode(SwccgGame game, String player, AwaitingDecision decision,
            int schema, boolean checks) {
        if (schema != 4) {
            throw new IllegalArgumentException("knowledge encoder supports schema 4 only, not " + schema);
        }
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
        if (checks && schema == 4 && System.getProperty("policy.checkContextV5") != null) {
            com.gempukku.swccgo.game.knowledge.ContextV5Check.check(game, player, PolicyTensorsV4::contextRows);
        }
        if (checks && System.getProperty("policy.checkHandOwners") != null) {
            com.gempukku.swccgo.game.knowledge.HandOwnerCheck.check(game);
        }
        if (checks && Boolean.getBoolean("policy.checkInformationBoundary")) {
            com.gempukku.swccgo.game.knowledge.InformationBoundaryCheck.check(game, player);
        }
        StateEncoder.Encoded encoded = new StateEncoder.Encoded();
        encoded.version = schema;
        encoded.type = decision.getDecisionType().name();
        long actorStart = KnowledgeTiming.start();
        encoded.observation = new ObservationV2(game, player, false, schema);
        TitleMatcher.References references = new TitleMatcher.References();
        String question = TitleMatcher.shared().template(decision.getText(), references,
            TitleMatcher.permitted(encoded.observation));
        for (int index = 0; index < references.size(); index++) {
            encoded.observation.reference(references, 0, index);
        }
        StateEncoder.encodeOptions(encoded, decision, encoded.observation.byCardId);
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
        // BINARY_ONLY_BEGIN
        if (Boolean.getBoolean("policy.binaryTrajectory")) {
            encoded.binary = new BinaryObservation(encoded.observation, question, references, globals,
                game.getDarkPlayer().equals(player) ? 0 : 1);
            if (schema == 4 && state.getKnowledge().shadow != null) {
                encoded.binary.legacy = legacyView(game, player, decision, state.getKnowledge().shadow, encoded, globals);
            }
        }
        // BINARY_ONLY_END
        prefix.append(Arrays.toString(globals)).append(',').append(encoded.observation.json())
            .append(',').append(options);
        KnowledgeTiming.end("actor_encode", actorStart);
        if (Boolean.getBoolean("policy.captureCritic")) {
            long criticStart = KnowledgeTiming.start();
            ObservationV2 critic = new ObservationV2(game, player, true, schema);
            for (int index = 0; index < references.size(); index++) {
                critic.reference(references, 0, index);
            }
            // BINARY_ONLY_BEGIN
            if (encoded.binary != null) {
                encoded.binary.privileged = critic;
            }
            if (encoded.binary != null && encoded.binary.legacy != null) {
                encoded.binary.legacy.privileged = critic;  // the critic's privileged view is the same under both rules
            }
            // BINARY_ONLY_END
            encoded.criticTokens = Boolean.getBoolean("policy.binaryTrajectory")
                && !Boolean.getBoolean("policy.checkProductionParity") ? null : critic.json();
            encoded.knowledgeSequence = state.getKnowledge().sequence;
            if (schema == 4) {
                encoded.criticGrid = com.gempukku.swccgo.game.knowledge.KnowledgeGrid.json(critic.kgrid);
            }
            KnowledgeTiming.end("critic_encode", criticStart);
        }
        if (checks && schema == 4 && BrainiacEncoderCheck.enabled()) {
            BrainiacEncoderCheck.check(encoded, game, player, decision);
        }
        if (checks && schema == 4 && CounterfactualProbe.enabled()) {
            CounterfactualProbe.probe(game, player, decision, encoded);
        }
        if (schema == 4 && ReplayControl.observing()) {
            ReplayControl.observe(game, player, encoded);  // plan critic-followups BL-49: rollout outcome states
        }
        return encoded;
    }

        // BINARY_ONLY_BEGIN (the production staging strips this block)
    /** Plan critic-noise stage 0: the actor view of the same decision under the shadow knowledge (the legacy rule), with
     * its own question template. Its card tokens must be the current view's, in the same order (the options' source
     * tokens index both), else the pair is unusable and the game stops. */
    private static BinaryObservation legacyView(SwccgGame game, String player, AwaitingDecision decision,
                                                com.gempukku.swccgo.game.knowledge.GameKnowledge shadow,
                                                StateEncoder.Encoded encoded, int[] globals) {
        ObservationV2 legacy = new ObservationV2(game, player, false, 4, shadow);
        TitleMatcher.References references = new TitleMatcher.References();
        String question = TitleMatcher.shared().template(decision.getText(), references, TitleMatcher.permitted(legacy));
        for (int index = 0; index < references.size(); index++) {
            legacy.reference(references, 0, index);
        }
        // The options' text references (kind 10) were appended to the current view by StateEncoder.encodeOptions; they
        // name cards in option texts and do not depend on the knowledge rule.
        for (ObservationV2.Token token : encoded.observation.tokens) {
            if (token.kind == 10) legacy.tokens.add(token);
        }
        // Card tokens must sit at the same positions (option sources index them); the knowledge tokens after them may
        // differ in number under the legacy rule (e.g. a stack row that is no longer exact).
        if (!legacy.byCardId.equals(encoded.observation.byCardId)) {
            java.util.Map<Integer,Integer> a = new java.util.TreeMap<>(legacy.byCardId), b = new java.util.TreeMap<>(encoded.observation.byCardId);
            java.util.Set<Integer> only = new java.util.TreeSet<>(b.keySet()); only.removeAll(a.keySet());
            java.util.Set<Integer> extra = new java.util.TreeSet<>(a.keySet()); extra.removeAll(b.keySet());
            int moved = 0; for (Integer id : a.keySet()) if (b.containsKey(id) && !a.get(id).equals(b.get(id))) moved++;
            throw new IllegalStateException("legacy view tokens out of step with the current view: tokens "
                + legacy.tokens.size() + " vs " + encoded.observation.tokens.size() + ", cards " + a.size() + " vs " + b.size()
                + ", only current " + only + ", only legacy " + extra + ", moved " + moved
                + ", kinds " + kinds(legacy) + " vs " + kinds(encoded.observation) + ", question " + question
                + " | " + decision.getText());
        }
        return new BinaryObservation(legacy, question, references, globals, game.getDarkPlayer().equals(player) ? 0 : 1);
    }
    private static java.util.Map<Integer, Integer> kinds(ObservationV2 observation) {
        java.util.Map<Integer, Integer> out = new java.util.TreeMap<>();
        for (ObservationV2.Token token : observation.tokens) out.merge(token.kind, 1, Integer::sum);
        return out;
    }
        // BINARY_ONLY_END
}
