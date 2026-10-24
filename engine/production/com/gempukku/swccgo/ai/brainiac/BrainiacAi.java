package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.ai.AsyncAiController;
import com.gempukku.swccgo.ai.models.BeginnerAi;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.rules.FreeMoveRule;
import com.gempukku.swccgo.game.rules.VisibleCancel;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The Brainiac bot: a policy network trained by self-play (swccg-policy), run in the opponent's browser. For each decision with
 * two or more options the server encodes the bot's view ({@link StateEncoder}, the shared ActorTensors) and sends it to the
 * browser, which runs the network and answers with the options ranked best first. The server then picks the answer, exactly as
 * the training harness does:
 * <ul>
 *   <li>multi-card selections take the top k options, k = max(min, 1); a "none"/"pass" option ranked first answers nothing;
 *       step-by-step selections (preselected cards) keep the preselection and add the best new card(s), see
 *       {@link StepwiseCandidates};</li>
 *   <li>loop guard: only an identical situation counts as a repeat (a hash of the decision type and the exact encoded input,
 *       kept per game). From its (K+1)th occurrence (K = 10) the best-ranked answer not yet tried there is taken;</li>
 *   <li>an answer the game rejects counts as tried, and the next candidate is submitted without asking the browser again;</li>
 *   <li>BeginnerAi answers only when every candidate has been tried (a stuck policy), or when there is nothing to rank;</li>
 *   <li>Brainiac's seat plays the permanent training rules (plan permanent-play-rules; the human's seat plays stock
 *       Gemp): Force activation ({@link ActivationAdapter}), no "Done to cancel" when every selectable card is visible
 *       ({@link VisibleCancel}) and no free move back into a position held earlier in the move phase
 *       ({@link FreeMoveRule}). The latter two filter Brainiac's own decision before it is encoded.</li>
 * </ul>
 */
public class BrainiacAi implements AsyncAiController {
    public static final AtomicLong DECISIONS = new AtomicLong(), REQUESTS = new AtomicLong(), LAST_RESORT = new AtomicLong(),
            LOOP_GUARD = new AtomicLong(), REJECTED = new AtomicLong(), RULE_CANCEL = new AtomicLong(),
            RULE_FREE_MOVE = new AtomicLong();
    static final int LOOP_K = 10;
    private static final Logger LOG = LogManager.getLogger(BrainiacAi.class);

    private final BrainiacRelease release;
    private final BeginnerAi lastResort = new BeginnerAi();
    private final ActivationAdapter activation = new ActivationAdapter();
    // Gemp gives every decision the same id (1): a decision is identified by the object itself.
    private AwaitingDecision forcedDecision;
    private AwaitingDecision ruledDecision;
    private String forcedAnswer;
    private final Map<Long, Integer> seen = new HashMap<>();
    private final Map<Long, Set<String>> tried = new HashMap<>();
    private SwccgGame game;
    private int requestedDecisionId = -1;
    private StateEncoder.Encoded requested;
    private long requestedFingerprint;
    private int requestedRepeats;
    private int requestedK;
    private List<String> candidates;
    private String lastAnswer;

    public BrainiacAi(BrainiacRelease release) {
        this.release = release;
    }

    @Override
    public void setGame(SwccgGame game) {
        this.game = game;
        lastResort.setGame(game);
    }

    @Override
    public String requestDecision(String playerId, AwaitingDecision decision, SwccgGame game) {
        setGame(game);
        applyRules(playerId, decision);
        String forced = activation.forced(game, playerId, decision);
        if (forced != null) {
            // Answered by decide() without the browser.
            forcedDecision = decision;
            forcedAnswer = forced;
            LOG.debug("Brainiac activation adapter: \"" + decision.getText() + "\" -> " + forced);
            return null;
        }
        StateEncoder.Encoded e = StateEncoder.encode(game, playerId, decision, activation.hidden(game, playerId, decision));
        if (e.answers.size() < 2)
            return null;
        long fp = fingerprint(e);
        boolean multi = (e.type.equals("CARD_SELECTION") || e.type.equals("ARBITRARY_CARDS")) && e.max > 1;
        requested = e;
        requestedDecisionId = decision.getAwaitingDecisionId();
        requestedFingerprint = fp;
        requestedRepeats = seen.merge(fp, 1, Integer::sum);
        requestedK = StepwiseCandidates.applies(e) ? StepwiseCandidates.picks(e) : multi ? Math.max(e.min, 1) : 1;
        candidates = null;
        REQUESTS.incrementAndGet();

        var t = release.tensors(e.line("brainiac", 0, List.of()));
        JsonObject p = new JsonObject();
        p.addProperty("release", release.name);
        p.addProperty("sha256", release.modelSha256);
        p.addProperty("runtime", release.runtime);
        p.addProperty("k", requestedK);
        p.addProperty("t", Base64.getEncoder().encodeToString(t.browserBytes()));
        p.addProperty("question", decision.getText() == null ? "" : decision.getText().replaceAll("<[^>]+>", ""));
        JsonArray labels = new JsonArray();
        for (String l : e.labels) labels.add(l);
        p.add("labels", labels);
        JsonArray hand = new JsonArray();
        for (String[] h : e.hand) {
            JsonArray c = new JsonArray();
            c.add(h[0]);
            c.add(h[1]);
            hand.add(c);
        }
        p.add("hand", hand);
        return p.toString();
    }

    @Override
    public int getRequestedDecisionId() {
        return requestedDecisionId;
    }

    @Override
    public String answer(String playerId, AwaitingDecision decision, String response) {
        DECISIONS.incrementAndGet();
        List<Integer> ranking = new ArrayList<>();
        if (response != null)
            for (String s : response.split(","))
                if (!s.isBlank()) {
                    try {
                        ranking.add(Integer.parseInt(s.trim()));
                    } catch (NumberFormatException ignored) {
                    }
                }
        StateEncoder.Encoded e = requested;
        if (StepwiseCandidates.applies(e)) {
            candidates = StepwiseCandidates.candidates(e, ranking);
            LOG.debug("Brainiac step-by-step selection \"" + decision.getText() + "\": preselected " + e.preselectedAnswers
                    + ", min " + e.min + ", max " + e.max + ", answer " + (candidates.isEmpty() ? "-" : candidates.get(0)));
        } else {
            candidates = candidates(e, ranking, requestedK);
        }
        if (candidates.isEmpty())
            return lastResort(playerId, decision, "empty ranking");
        String answer = candidates.get(0);
        int rep = requestedRepeats;
        if (rep > LOOP_K) {
            Set<String> t = tried.computeIfAbsent(requestedFingerprint, x -> new HashSet<>());
            answer = null;
            for (String c : candidates)
                if (!t.contains(canonical(c))) {
                    answer = c;
                    break;
                }
            LOOP_GUARD.incrementAndGet();
            if (answer == null)
                return lastResort(playerId, decision, "loop guard: all " + candidates.size() + " answers tried");
            t.add(canonical(answer));
        } else if (rep == LOOP_K) {
            tried.computeIfAbsent(requestedFingerprint, x -> new HashSet<>()).add(canonical(answer));
        }
        lastAnswer = answer;
        FreeMoveRule.decided(game, decision, answer);
        return answer;
    }

    @Override
    public String rejected(String playerId, AwaitingDecision decision, SwccgGame game) {
        REJECTED.incrementAndGet();
        LOG.info("Brainiac answer rejected: \"" + lastAnswer + "\" for decision \"" + decision.getText() + "\"");
        setGame(game);
        if (candidates != null && lastAnswer != null) {
            Set<String> t = tried.computeIfAbsent(requestedFingerprint, x -> new HashSet<>());
            t.add(canonical(lastAnswer));
            for (String c : candidates)
                if (!t.contains(canonical(c))) {
                    lastAnswer = c;
                    FreeMoveRule.decided(game, decision, c);
                    return c;
                }
        }
        return lastResort(playerId, decision, "every answer rejected");
    }

    /** Decisions with fewer than two options: the only answer, or BeginnerAi if there is nothing to choose from. */
    @Override
    public String decide(String playerId, AwaitingDecision decision, GameState gameState) {
        applyRules(playerId, decision);
        String forced = decision == forcedDecision ? forcedAnswer : activation.forced(game, playerId, decision);
        forcedDecision = null;
        if (forced != null) {
            candidates = null;
            lastAnswer = forced;
            FreeMoveRule.decided(game, decision, forced);
            return forced;
        }
        StateEncoder.Encoded e = StateEncoder.encode(game, playerId, decision, activation.hidden(game, playerId, decision));
        if (e.answers.size() == 1) {
            candidates = null;
            lastAnswer = e.answers.get(0);
            FreeMoveRule.decided(game, decision, lastAnswer);
            return lastAnswer;
        }
        candidates = null;
        lastAnswer = null;
        String fallback = lastResort.decide(playerId, decision, gameState);
        FreeMoveRule.decided(game, decision, fallback);
        return fallback;
    }

    /**
     * The permanent training rules on Brainiac's own decision, once per decision, before anything reads it. They change
     * the decision in place (fewer actions or embark targets; minimum 1 on an all-visible cancel); only Brainiac answers
     * it.
     */
    private void applyRules(String playerId, AwaitingDecision decision) {
        if (decision == ruledDecision)
            return;
        ruledDecision = decision;
        long cancels = VisibleCancel.RESTRICTED.get();
        int before = optionCount(decision);
        VisibleCancel.restrict(decision, playerId);
        FreeMoveRule.restrict(game, decision);
        if (VisibleCancel.RESTRICTED.get() != cancels) {
            RULE_CANCEL.incrementAndGet();
            LOG.debug("Brainiac rule no-visible-cancel: \"" + decision.getText() + "\" needs a card");
        }
        int after = optionCount(decision);
        if (after < before) {
            RULE_FREE_MOVE.addAndGet(before - after);
            LOG.debug("Brainiac rule free-move: " + (before - after) + " return move(s) removed from \"" + decision.getText() + "\"");
        }
    }

    private static int optionCount(AwaitingDecision decision) {
        String[] actions = decision.getDecisionParameters().get("actionId");
        String[] cards = decision.getDecisionParameters().get("cardId");
        return actions != null ? actions.length : cards != null ? cards.length : 0;
    }

    private String lastResort(String playerId, AwaitingDecision decision, String why) {
        LAST_RESORT.incrementAndGet();
        LOG.info("Brainiac last resort (BeginnerAi): " + why + "; decision \"" + decision.getText() + "\"");
        candidates = null;
        lastAnswer = null;
        String fallback = lastResort.decide(playerId, decision, game.getGameState());
        FreeMoveRule.decided(game, decision, fallback);
        return fallback;
    }

    /**
     * The answers to try, in order: the ranking itself, then the ranking rotated by 1, 2, ... (distinct answers only; a
     * multi-card answer is the same answer in any order).
     */
    static List<String> candidates(StateEncoder.Encoded e, List<Integer> ranking, int k) {
        int n = e.answers.size();
        Map<String, String> out = new LinkedHashMap<>();
        for (int shift = 0; shift < Math.max(1, ranking.size()); shift++) {
            List<String> parts = new ArrayList<>();
            String answer = null;
            for (int j = 0; j < ranking.size(); j++) {
                int i = ranking.get((j + shift) % ranking.size());
                if (i < 0 || i >= n) continue;
                if (e.isNone.get(i)) {
                    if (parts.isEmpty()) {
                        answer = "";
                        break;
                    }
                    continue;
                }
                parts.add(e.answers.get(i));
                if (parts.size() >= k) break;
            }
            if (answer == null && !parts.isEmpty()) answer = String.join(",", parts);
            if (answer != null) out.putIfAbsent(canonical(answer), answer);
        }
        return new ArrayList<>(out.values());
    }

    static String canonical(String answer) {
        if (answer.indexOf(',') < 0) return answer;
        String[] parts = answer.split(",");
        Arrays.sort(parts);
        return String.join(",", parts);
    }

    /** 64-bit FNV-1a hash of the decision type and the exact encoded model input (state, prompt, options). */
    static long fingerprint(StateEncoder.Encoded e) {
        long h = 0xcbf29ce484222325L;
        for (CharSequence cs : new CharSequence[]{e.type, "\0", e.prefix})
            for (int i = 0; i < cs.length(); i++) {
                h ^= cs.charAt(i);
                h *= 0x100000001b3L;
            }
        return h;
    }
}
