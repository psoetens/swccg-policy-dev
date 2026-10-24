package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.ai.SwccgAiController;
import com.gempukku.swccgo.ai.models.BeginnerAi;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Bot backed by the Python policy server (swccg-policy/src/swccg_policy/server.py). Each decision with two or more options is
 * encoded with StateEncoder and sent as one line; the server answers with a JSON array of option indices, best first. Decisions the
 * encoder cannot express (EMPTY, or no options) and re-sent decisions after an invalid answer fall back to BeginnerAi.
 *
 * RL mode (constructed with a model key, for swccg-policy/src/swccg_policy/rl.py): each query is an envelope
 * {"m":modelKey,"g":gameId,"s":side,"k":picks,"r":repeats,"rec":<record>}, and at game end {@link #finish} sends
 * {"end":gameId,"s":side,"m":modelKey,"o":outcome,"w":winner}. The server samples (keys without "@greedy") and returns the
 * sampled option(s) first. The envelope carries the repeat count "r" of the situation, so the server does not record steps
 * whose action the loop guard overrides (r > the server's --guard-threshold, which must equal policy.loopGuardThreshold).
 *
 * Step-by-step selections (preselected cards): the answer keeps the preselection and adds the best new card(s), see
 * {@link StepwiseCandidates}; the envelope tells the server which options it may not sample ("x") so the recorded action is the
 * one played, or not to record the step ("skip"). Plan multiselect-count: finishing is an explicit Done option, and card
 * multi-selections whose count is a choice are answered step by step inside PolicyAi ({@link #emulated}).
 *
 * Loop guard (both modes): only an identical situation counts as a repeat, identified by a 64-bit hash of the decision type and
 * the exact encoded model input (state, prompt and options), kept per game. Up to K = policy.loopGuardThreshold (default 10)
 * occurrences the policy decides as usual; from occurrence K + 1 on it takes its best-ranked answer not yet tried in that
 * situation (remembered for the rest of the game), and only once every answer has been tried does BeginnerAi decide. An answer
 * the game rejects (the same decision object is re-sent) counts as tried too, and the next untried answer is submitted.
 */
public class PolicyAi implements SwccgAiController {
    public static final AtomicLong DECISIONS = new AtomicLong(), QUERIES = new AtomicLong(), FALLBACKS = new AtomicLong(),
            QUERY_NANOS = new AtomicLong(), LOOP_GUARD = new AtomicLong(), REAL_LOOPS = new AtomicLong(), REJECTED = new AtomicLong();
    // Round-trip latency histogram of policy queries: bucket i covers [2^(i/4), 2^((i+1)/4)) microseconds (±9 % resolution).
    public static final java.util.concurrent.atomic.AtomicLongArray QUERY_HIST = new java.util.concurrent.atomic.AtomicLongArray(128);
    // -Dpolicy.checkEncoder=true: every encoded decision is also encoded without the query cache and compared (and the
    // options-only answers with the full encoding's); mismatches are counted in ENCODER_MISMATCHES and logged to stderr.
    // -Dpolicy.countFirst=false: encode the state before counting the options (the old order; for the replay comparison)
    static final boolean COUNT_FIRST = !"false".equals(System.getProperty("policy.countFirst"));
    static final boolean CHECK_ENCODER = Boolean.getBoolean("policy.checkEncoder");
    public static final AtomicLong ENCODER_CHECKS = new AtomicLong(), ENCODER_MISMATCHES = new AtomicLong();
    private static final boolean LOOP_GUARD_ON = !"false".equals(System.getProperty("policy.loopGuard"));
    static final int LOOP_K = Integer.getInteger("policy.loopGuardThreshold", 10);
    // -Dpolicy.loopLog=FILE: one line per loop-guard override (game, player, prompt, occurrence, answer, tried answers)
    private static final PrintWriter LOOP_LOG = openLoopLog(System.getProperty("policy.loopLog"));
    // Per game: occurrences of each situation (fingerprint), and for situations beyond K (or with a rejected answer) the answers
    // already tried there. Only hashes are kept, never the encoding itself.
    private final Map<Long, Integer> seen = new HashMap<>();
    private final Map<Long, Set<String>> tried = new HashMap<>();
    private long lastFingerprint;
    private List<String> lastCandidates;
    private String lastAnswer;
    private final String host;
    private final int port;
    private final BeginnerAi fallback = new BeginnerAi();
    private SwccgGame game;
    private Socket socket;
    private BufferedWriter out;
    private BufferedReader in;
    private AwaitingDecision lastDecision;
    private int lastDecisionRepeats;
    private final String modelKey; // null: legacy protocol (server.py)
    private final PolicyEngine engine; // non-null: the policy runs in this JVM (native backend), no socket
    private String gameId = "play";
    private String side;

    public PolicyAi(String host, int port) {
        this(host, port, null);
    }

    public PolicyAi(String host, int port, String modelKey) {
        this.host = host;
        this.port = port;
        this.modelKey = modelKey;
        this.engine = modelKey == null ? null : PolicyEngine.get();
    }

    public void setGameId(String gameId) {
        this.gameId = gameId;
        seen.clear();
        tried.clear();
        lastCandidates = null;
    }

    /** RL mode: tell the server how the game ended (it closes this side's trajectory). */
    public void finish(String outcome, String winner) {
        finish(outcome, winner, null);
    }

    /**
     * RL mode: tell the server how the game ended: outcome, winner, the opponent's kind ("op", e.g. advanced / random, for the
     * learner's statistics) and both remaining Life Forces ("lf": [own, opponent], for the Life Force reward term).
     */
    public void finish(String outcome, String winner, String opponentKind) {
        if (modelKey == null || side == null || (engine == null && socket == null)) return;
        String lf = "";
        int[] lfs = null;
        try {
            GameState gs = game.getGameState();
            lfs = new int[]{gs.getPlayerLifeForce(side), gs.getPlayerLifeForce(game.getOpponent(side))};
            lf = ",\"lf\":[" + lfs[0] + "," + lfs[1] + "]";
        } catch (RuntimeException ignored) {
            // game aborted or not started: no Life Force term
        }
        if (engine != null) {
            engine.finish(gameId, side, modelKey, outcome, winner == null ? "" : winner, opponentKind, lfs);
            return;
        }
        try {
            send("{\"end\":" + StateEncoder.q(gameId) + ",\"s\":" + StateEncoder.q(side) + ",\"m\":" + StateEncoder.q(modelKey)
                    + ",\"o\":" + StateEncoder.q(outcome) + ",\"w\":" + StateEncoder.q(winner == null ? "" : winner)
                    + (opponentKind == null ? "" : ",\"op\":" + StateEncoder.q(opponentKind)) + lf + "}");
        } catch (IOException ignored) {
        }
    }

    @Override
    public void setGame(SwccgGame game) {
        this.game = game;
        fallback.setGame(game);
    }

    @Override
    public String decide(String playerId, AwaitingDecision decision, GameState gameState) {
        DECISIONS.incrementAndGet();
        // The same decision object again = the game rejected the previous answer (compared by reference: identity hashes can
        // collide, and are all equal under -XX:hashCode=2, which the seeded replay check uses).
        lastDecisionRepeats = decision == lastDecision ? lastDecisionRepeats + 1 : 0;
        lastDecision = decision;
        if (lastDecisionRepeats > 0) {
            // The game rejected the previous answer and re-sent the decision: count it as tried, submit the next untried answer.
            REJECTED.incrementAndGet();
            if (lastCandidates != null) {
                Set<String> t = tried.computeIfAbsent(lastFingerprint, x -> new HashSet<>());
                t.add(canonical(lastAnswer));
                for (String c : lastCandidates)
                    if (!t.contains(canonical(c))) {
                        lastAnswer = c;
                        return c;
                    }
            }
            FALLBACKS.incrementAndGet();
            return fallback.decide(playerId, decision, gameState);
        }
        // Count the options before encoding the state: decisions with fewer than two options (most of them, e.g. response
        // windows where only "pass" is possible) never reach the policy. For a single option the fingerprint only keys the
        // rejected-answer set, which for one answer always ends in the fallback, so the options-only fingerprint suffices.
        StateEncoder.Encoded o = COUNT_FIRST ? StateEncoder.optionsOnly(decision) : StateEncoder.encode(game, playerId, decision);
        int n = o.answers.size();
        if (n == 0) {
            lastCandidates = null;
            FALLBACKS.incrementAndGet();
            return fallback.decide(playerId, decision, gameState);
        }
        if (n == 1) {
            lastFingerprint = fingerprint(o);
            lastCandidates = List.of(o.answers.get(0));
            lastAnswer = o.answers.get(0);
            return lastAnswer;
        }
        StateEncoder.Encoded e = COUNT_FIRST ? StateEncoder.encode(game, playerId, decision) : o;
        if (CHECK_ENCODER) checkEncoder(e, playerId, decision, o);
        long fp = fingerprint(e);
        try {
            side = playerId;
            boolean multi = (e.type.equals("CARD_SELECTION") || e.type.equals("ARBITRARY_CARDS")) && e.max > 1;
            boolean stepwise = StepwiseCandidates.applies(e);
            int rep = LOOP_GUARD_ON ? seen.merge(fp, 1, Integer::sum) : 1;
            List<String> candidates;
            if (stepwise) candidates = gempStepwise(e, playerId, decision.getText(), rep);
            else if (multi && countChoice(e, decision.getText())) candidates = emulated(e, playerId, decision.getText(), rep);
            else {
                int k = multi ? multiPicks(e, decision.getText()) : 1;
                // The server returns the options ranked best-first (in RL mode the sampled option(s) first).
                candidates = candidates(e, ask(e, playerId, k, rep, false, null), k);
            }
            if (candidates.isEmpty()) {
                lastCandidates = null;
                FALLBACKS.incrementAndGet();
                return fallback.decide(playerId, decision, gameState);
            }
            String answer = candidates.get(0);
            if (rep > LOOP_K) {
                // The same situation for the (K+1)th time or more: the policy's best answer not yet tried here.
                if (rep == LOOP_K + 1) REAL_LOOPS.incrementAndGet();
                Set<String> t = tried.computeIfAbsent(fp, x -> new HashSet<>());
                answer = null;
                for (String c : candidates)
                    if (!t.contains(canonical(c))) {
                        answer = c;
                        break;
                    }
                LOOP_GUARD.incrementAndGet();
                if (LOOP_LOG != null) synchronized (LOOP_LOG) {
                    LOOP_LOG.println(gameId + "\t" + playerId + "\t" + e.type + "\t" + decision.getText() + "\t" + rep + "\t"
                            + (answer == null ? "<BeginnerAi>" : answer) + "\t" + t + "\t" + String.join("|", e.answers));
                    LOOP_LOG.flush();
                }
                if (answer == null) {
                    lastCandidates = null;
                    FALLBACKS.incrementAndGet();
                    return fallback.decide(playerId, decision, gameState);
                }
                t.add(canonical(answer));
            } else if (rep == LOOP_K) {
                tried.computeIfAbsent(fp, x -> new HashSet<>()).add(canonical(answer));
            }
            lastFingerprint = fp;
            lastCandidates = candidates;
            lastAnswer = answer;
            return answer;
        } catch (IOException ex) {
            throw new UncheckedIOException("policy server " + host + ":" + port, ex);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the policy", ex);
        }
    }

    /**
     * Test seam (plan multiselect-count, Java checks): ranks a query instead of the policy. Arguments as {@link #ask}; returns
     * the option indices, the answer first.
     */
    interface Ranker {
        List<Integer> rank(StateEncoder.Encoded e, int k, int repeats, boolean skip, List<Integer> excluded);
    }

    Ranker ranker;

    /**
     * One policy query: the options ranked best first (in RL mode the sampled option(s) first). k = picks; rep = the loop
     * guard's occurrence count of the decision; undo = do not record the step ("skip"); excluded = options that may not be
     * sampled ("x"; null = none).
     */
    private List<Integer> ask(StateEncoder.Encoded e, String playerId, int k, int rep, boolean undo, List<Integer> excluded)
            throws IOException, InterruptedException {
        int flags = 0;
        for (boolean selected : e.selected) if (selected) flags++;
        if (flags > 0) {
            FLAGGED_QUERIES.incrementAndGet();
            FLAGGED_OPTIONS.addAndGet(flags);
            decisionFlags += flags;
        }
        if (ranker != null) {
            QUERIES.incrementAndGet();
            return ranker.rank(e, k, rep, undo, excluded);
        }
        long t0 = System.nanoTime();
        String line = e.line(modelKey == null ? "play" : gameId, 0, List.of());
        List<Integer> ranking;
        if (engine != null) {
            // in-JVM policy: the same fields as the envelope below ("skip" / "x" as in envelopeFields)
            ranking = engine.query(modelKey, gameId, playerId, line, k, rep, undo,
                    excluded);
        } else {
            if (modelKey != null)
                line = "{\"m\":" + StateEncoder.q(modelKey) + ",\"g\":" + StateEncoder.q(gameId) + ",\"s\":" + StateEncoder.q(playerId)
                        + ",\"k\":" + k + ",\"r\":" + rep + envelopeFields(undo, excluded) + ",\"rec\":" + line + "}";
            ranking = query(line);
        }
        long dt = System.nanoTime() - t0;
        QUERY_NANOS.addAndGet(dt);
        recordLatency(dt);
        QUERIES.incrementAndGet();
        return ranking;
    }

    /** The first index of a ranking that is a valid option and not excluded, or -1. */
    static int first(List<Integer> ranking, int n, java.util.Collection<Integer> excluded) {
        for (int i : ranking)
            if (i >= 0 && i < n && !excluded.contains(i)) return i;
        return -1;
    }

    // Plan multiselect-count: count-choice statistics (decisions, policy queries, how they ended), and with
    // -Dpolicy.multiselectLog=FILE one JSON line per such decision.
    public static final AtomicLong MS_DECISIONS = new AtomicLong(), MS_STEPS = new AtomicLong(), MS_DONE = new AtomicLong(),
            MS_MAX = new AtomicLong(), MS_CANCEL = new AtomicLong(), GEMP_STEPWISE = new AtomicLong(),
            GEMP_STEPWISE_DONE_OFFERED = new AtomicLong(), GEMP_STEPWISE_DONE = new AtomicLong(),
            // stage B: queries with at least one option marked "already selected", and the marks in them
            FLAGGED_QUERIES = new AtomicLong(), FLAGGED_OPTIONS = new AtomicLong();
    private int decisionFlags; // marks in this decision's queries (for the log)
    private static final PrintWriter MS_LOG = openLoopLog(System.getProperty("policy.multiselectLog"));

    private void logSelection(String mode, String text, StateEncoder.Encoded e, int cards, int queries, String end, int answered) {
        if (MS_LOG == null) return;
        String line = "{\"g\":" + StateEncoder.q(gameId) + ",\"p\":" + StateEncoder.q(side) + ",\"mode\":" + StateEncoder.q(mode)
                + ",\"type\":" + StateEncoder.q(e.type) + ",\"text\":" + StateEncoder.q(text) + ",\"min\":" + e.min + ",\"max\":" + e.max
                + ",\"cards\":" + cards + ",\"preselected\":" + e.preselectedAnswers.size() + ",\"queries\":" + queries
                + ",\"end\":" + StateEncoder.q(end) + ",\"answered\":" + answered + ",\"flagged\":" + decisionFlags + "}";
        decisionFlags = 0;
        synchronized (MS_LOG) {
            MS_LOG.println(line);
            MS_LOG.flush();
        }
    }

    /**
     * Plan multiselect-count: a card multi-selection whose count the policy chooses (answered step by step, {@link #emulated}):
     * a CARD_SELECTION, or an ARBITRARY_CARDS without preselection, that allows more than max(min, 1) cards and offers more
     * cards than that, except the two templates answered with max cards ({@link #multiPicks}).
     */
    static boolean countChoice(StateEncoder.Encoded e, String text) {
        if (StepwiseCandidates.applies(e) || takesMax(text)) return false;
        int cards = 0;
        for (boolean none : e.isNone) if (!none) cards++;
        return e.max > Math.max(e.min, 1) && cards > Math.max(e.min, 1);
    }

    /**
     * Plan multiselect-count: a card multi-selection answered step by step inside PolicyAi, one policy query per added card,
     * then one answer to Gemp. Each query is the decision's own encoding with the cards selected so far excluded from
     * sampling; once the count reaches min (and at least one card is selected) a Done option ({@link StateEncoder#appendDone},
     * "stop with the current selection") is added as the last option. Cancel stays the decision's "none" option (the empty
     * answer, only where Gemp allows it: min 0), offered on the first step only (later it would drop the selection; Done
     * with nothing selected would be the same answer as cancel, so Done is not offered then). Max reached (or no card left)
     * finishes without a query, and so does a step with a single allowed option. Every query is a normal policy step (recorded
     * in RL mode), Done a normal option row.
     * Candidates: the built answer, then the one-shot answers of the first step's ranking (for a rejected answer).
     */
    private List<String> emulated(StateEncoder.Encoded e, String playerId, String text, int rep)
            throws IOException, InterruptedException {
        MS_DECISIONS.incrementAndGet();
        decisionFlags = 0;
        List<Integer> cards = new ArrayList<>(), none = new ArrayList<>();
        for (int i = 0; i < e.answers.size(); i++) (e.isNone.get(i) ? none : cards).add(i);
        int hi = Math.min(e.max, cards.size());
        List<Integer> selected = new ArrayList<>();
        List<Integer> firstRanking = null;
        int done = -1, queries = 0;
        String end;
        while (true) {
            if (selected.size() >= hi) {
                end = "max";
                break;
            }
            if (done < 0 && !selected.isEmpty() && selected.size() >= e.min) done = StateEncoder.appendDone(e);
            List<Integer> excluded = new ArrayList<>(selected);
            if (!selected.isEmpty()) excluded.addAll(none);
            int allowed = e.answers.size() - excluded.size();
            int pick;
            if (allowed <= 1) {
                pick = first(range(e.answers.size()), e.answers.size(), excluded);
            } else {
                StateEncoder.setSelected(e, selected); // stage B: the cards selected so far are model input
                List<Integer> ranking = ask(e, playerId, 1, rep, false, excluded.isEmpty() ? null : excluded);
                queries++;
                MS_STEPS.incrementAndGet();
                if (firstRanking == null) firstRanking = ranking;
                pick = first(ranking, e.answers.size(), excluded);
            }
            if (pick < 0) {
                end = selected.size() >= Math.max(e.min, 1) ? "fallback-finish" : "fallback";
                break;
            }
            if (pick == done) {
                end = "done";
                break;
            }
            if (e.isNone.get(pick)) {
                end = "cancel";
                break;
            }
            selected.add(pick);
        }
        if (end.equals("done")) MS_DONE.incrementAndGet();
        else if (end.equals("max")) MS_MAX.incrementAndGet();
        else if (end.equals("cancel")) MS_CANCEL.incrementAndGet();
        List<String> parts = new ArrayList<>();
        for (int i : selected) parts.add(e.answers.get(i));
        String answer = end.equals("cancel") ? "" : end.equals("fallback") ? null : String.join(",", parts);
        logSelection("emulated", text, e, cards.size(), queries, end, answer == null ? -1 : parts.size());
        Map<String, String> out = new java.util.LinkedHashMap<>();
        if (answer != null) out.put(canonical(answer), answer);
        if (firstRanking != null) {
            List<Integer> base = new ArrayList<>(firstRanking);
            base.remove(Integer.valueOf(done));
            for (String c : candidates(e, base, Math.max(e.min, 1))) out.putIfAbsent(canonical(c), c);
        }
        return new ArrayList<>(out.values());
    }

    /**
     * Gemp's own step-by-step selections (ARBITRARY_CARDS with a preselection, {@link StepwiseCandidates}). When a card must be
     * added: one query, the preselected cards excluded (as before). When the preselection may be answered as it is and a new
     * card may be added (plan multiselect-count): the Done option is offered, the preselected (and "none") options excluded;
     * Done = resubmit the preselection. When only finishing is possible: finish without a query. Undo: not recorded (as before).
     */
    private List<String> gempStepwise(StateEncoder.Encoded e, String playerId, String text, int rep)
            throws IOException, InterruptedException {
        GEMP_STEPWISE.incrementAndGet();
        decisionFlags = 0;
        // Stage B: the preselected cards are marked "already selected" (model input) in every query of the decision.
        List<Integer> pre = new ArrayList<>();
        for (int i = 0; i < e.preselected.size(); i++) if (e.preselected.get(i)) pre.add(i);
        StateEncoder.setSelected(e, pre);
        if (StepwiseCandidates.isUndo(e)) {
            List<String> out = StepwiseCandidates.candidates(e, ask(e, playerId, StepwiseCandidates.picks(e), rep, true, null));
            logSelection("gemp", text, e, 0, 1, "undo", out.isEmpty() ? -1 : out.get(0).isEmpty() ? 0 : out.get(0).split(",").length);
            return out;
        }
        List<Integer> x = StepwiseCandidates.excluded(e);
        if (StepwiseCandidates.needed(e) > 0) {
            List<String> out = StepwiseCandidates.candidates(e, ask(e, playerId, StepwiseCandidates.picks(e), rep, false, x));
            logSelection("gemp", text, e, 0, 1, "add-needed", out.isEmpty() ? -1 : out.get(0).split(",").length);
            return out;
        }
        if (!StepwiseCandidates.offersDone(e)) {
            logSelection("gemp", text, e, 0, 0, "forced-finish", e.preselectedAnswers.size());
            return List.of(StepwiseCandidates.finish(e));
        }
        GEMP_STEPWISE_DONE_OFFERED.incrementAndGet();
        int done = StateEncoder.appendDone(e);
        List<Integer> ranking = ask(e, playerId, 1, rep, false, x);
        List<String> candidates = StepwiseCandidates.candidates(e, ranking, done);
        boolean finished = first(ranking, e.answers.size(), x) == done;
        if (finished) GEMP_STEPWISE_DONE.incrementAndGet();
        logSelection("gemp", text, e, 0, 1, finished ? "done" : "add", candidates.isEmpty() ? -1
                : candidates.get(0).isEmpty() ? 0 : candidates.get(0).split(",").length);
        return candidates;
    }

    private static List<Integer> range(int n) {
        List<Integer> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(i);
        return out;
    }

    private void checkEncoder(StateEncoder.Encoded e, String playerId, AwaitingDecision decision, StateEncoder.Encoded o) {
        StateEncoder.Encoded ref = StateEncoder.encodeImpl(game, playerId, decision);
        ENCODER_CHECKS.incrementAndGet();
        boolean same = ref.prefix.toString().contentEquals(e.prefix) && ref.answers.equals(e.answers) && ref.isNone.equals(e.isNone)
                && o.answers.equals(e.answers) && o.isNone.equals(e.isNone) && o.min == e.min && o.max == e.max;
        if (!same && ENCODER_MISMATCHES.incrementAndGet() <= 20)
            System.err.println("ENCODER MISMATCH " + gameId + " " + playerId + "\n  cached   " + e.prefix + "\n  uncached " + ref.prefix
                    + "\n  options  " + o.answers + " vs " + e.answers);
    }

    /** The question of Gemp's StackCardsFromOutsideDeckEffect / StackCardsFromReserveDeckEffect with max > 1. */
    static final String STACK_UNDER = "Choose cards to stack under ";
    /** Squadron replacement (ReplacementAction): exactly the squadron's replacement count, which is the decision's max. */
    static final String REPLACE = "Choose cards to replace";
    /**
     * Collision! (Card1_074, the only user of this question): Gemp's real minimum is every opponent's starship at the system
     * (TargetCardsAtSameLocationEffect, target all: min = max = their number) and any smaller answer cancels the play
     * (plan multiselect-count, Amendment 1: take max, like the squadron replacement).
     */
    static final String COLLISION = "Choose starships";

    /**
     * Cards answered for a card multi-selection that is not stepwise: max(min, 1), except (plan facedown-fixes; the
     * census in reports/facedown-fixes.md documents every other template):
     * <ul>
     *   <li>the set-up stacks "deploy with up to N cards from outside your deck under here" (Fear Is My Ally, An Unusual
     *       Amount Of Fear: up to 10 Defensive Shields; the any-number / up-to-15 shield Effects) stack as many as
     *       possible. Gemp stacks every card without asking when all fit, and asks only when more are available than may
     *       be stacked (its max is already min(N, available));</li>
     *   <li>squadron replacement needs exactly the replacement count (3) of the cards offered; the decision allows "Done"
     *       (min 0 / 1) and Gemp aborts the action for fewer cards, so one card never replaces;</li>
     *   <li>Collision! ("Choose starships"): all of the opponent's starships at the system or the play is cancelled
     *       (plan multiselect-count, Amendment 1).</li>
     * </ul>
     * All answer max cards, the policy choosing which. Plan multiselect-count: every other selection whose count is a
     * choice ({@link #countChoice}) is answered step by step ({@link #emulated}); what remains here (min = max, or no more
     * cards than max(min, 1)) is a forced count.
     */
    static int multiPicks(StateEncoder.Encoded e, String text) {
        if (takesMax(text)) {
            int cards = 0;
            for (boolean none : e.isNone) if (!none) cards++;
            return Math.max(Math.min(e.max, cards), 1);
        }
        return Math.max(e.min, 1);
    }

    /** The templates answered with max cards (above). */
    static boolean takesMax(String text) {
        return text != null && (text.startsWith(STACK_UNDER) || text.startsWith(REPLACE) || text.equals(COLLISION)
                || text.equals(COLLISION + ", or click 'Done' to cancel"));
    }

    /**
     * RL envelope fields of a step-by-step selection: "x" = options the server may not sample (selected / preselected cards,
     * see {@link StepwiseCandidates#excluded} and {@link #emulated}), "skip" = don't record this step (undo, which follows a
     * rule instead of the sample).
     */
    static String envelopeFields(boolean skip, List<Integer> x) {
        if (skip) return ",\"skip\":true";
        if (x == null || x.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(",\"x\":[");
        for (int i = 0; i < x.size(); i++) sb.append(i == 0 ? "" : ",").append(x.get(i));
        return sb.append(']').toString();
    }

    /**
     * The answers to try, in order: the ranking itself, then the ranking rotated by 1, 2, ... (distinct answers only; a
     * multi-card answer is the same answer in any order). Card
     * selections allowing several cards take the top k, k = {@link #multiPicks}; a "none"/"pass" option ranked first means answering
     * with nothing.
     */
    static List<String> candidates(StateEncoder.Encoded e, List<Integer> ranking, int k) {
        int n = e.answers.size();
        Map<String, String> out = new java.util.LinkedHashMap<>();
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

    /** An answer as an order-insensitive key: a card selection "b,a" is the same answer as "a,b". */
    static String canonical(String answer) {
        if (answer.indexOf(',') < 0) return answer;
        String[] parts = answer.split(",");
        java.util.Arrays.sort(parts);
        return String.join(",", parts);
    }

    static void recordLatency(long nanos) {
        double us = Math.max(1.0, nanos / 1e3);
        QUERY_HIST.incrementAndGet(Math.min(127, (int) (4 * Math.log(us) / Math.log(2))));
    }

    /** Latency quantile q (0..1) of the recorded policy queries in ms (bucket midpoint, geometric), or 0 without queries. */
    public static double latencyQuantileMs(double q) {
        long total = 0;
        for (int i = 0; i < QUERY_HIST.length(); i++) total += QUERY_HIST.get(i);
        if (total == 0) return 0;
        long target = (long) Math.ceil(q * total), acc = 0;
        for (int i = 0; i < QUERY_HIST.length(); i++) {
            acc += QUERY_HIST.get(i);
            if (acc >= target) return Math.pow(2, (i + 0.5) / 4) / 1e3;
        }
        return Math.pow(2, 127.5 / 4) / 1e3;
    }

    private static PrintWriter openLoopLog(String path) {
        try {
            return path == null ? null : new PrintWriter(new FileWriter(path, StandardCharsets.UTF_8, true));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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

    private List<Integer> query(String line) throws IOException {
        String resp = send(line);
        List<Integer> idx = new ArrayList<>();
        String s = resp.trim().replace("[", "").replace("]", "");
        if (!s.isEmpty()) for (String t : s.split(",")) idx.add(Integer.parseInt(t.trim()));
        return idx;
    }

    private String send(String line) throws IOException {
        if (socket == null) {
            socket = new Socket(host, port);
            socket.setTcpNoDelay(true);
            out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
            in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
        }
        out.write(line);
        out.write('\n');
        out.flush();
        String resp = in.readLine();
        if (resp == null) throw new EOFException("policy server closed the connection");
        return resp;
    }

    public void close() {
        try {
            if (socket != null) socket.close();
        } catch (IOException ignored) {
        }
    }
}
