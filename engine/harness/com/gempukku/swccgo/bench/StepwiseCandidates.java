package com.gempukku.swccgo.bench;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Derived from com.gempukku.swccgo.ai.brainiac.StepwiseCandidates (feat/brainiac), the rule for the training harness. Plan
 * multiselect-count replaced its "a preselected card ranked first means finish" convention by an explicit Done option
 * (production Brainiac keeps the old rule until the next release).
 * Answers for step-by-step card selections: an ARBITRARY_CARDS decision with preselected cards, which Gemp asks one card at a time
 * (card combinations). An answer must keep the preselected cards: a policy's top-k cards alone would often replace them with an
 * invalid set. The rule, from the options ranked best first:
 * <ul>
 *   <li>add: all preselected cards plus the best-ranked new card(s) up to min; one candidate per next-best new card, so a rejected
 *       add tries the next card;</li>
 *   <li>finish: if the preselection already reaches min, answering it alone (resubmitting the preselection); first if the Done
 *       option ({@link StateEncoder#appendDone}, which PolicyAi offers when {@link #offersDone}) is ranked above every new card
 *       or no new card is left, otherwise after the adds;</li>
 *   <li>undo (Gemp's recovery state: nothing may be added and min is below the preselection): drop one preselected card, the
 *       lowest-ranked first; then an empty answer if allowed.</li>
 * </ul>
 * In sampling (training), {@link #excluded} and {@link #picks} make the sampled options exactly what is played: the preselected
 * options are always excluded (outside undo), and when finishing is allowed a sampled Done means finish.
 */
public final class StepwiseCandidates {
    private StepwiseCandidates() {
    }

    /** True for a step-by-step selection (an ARBITRARY_CARDS decision with preselected cards). */
    public static boolean applies(StateEncoder.Encoded e) {
        return e.type.equals("ARBITRARY_CARDS") && !e.preselectedAnswers.isEmpty();
    }

    /** Gemp's recovery state: nothing may be added, and cards must be removed. */
    public static boolean isUndo(StateEncoder.Encoded e) {
        return e.max <= e.preselectedAnswers.size() && e.min < e.preselectedAnswers.size();
    }

    /** New cards that must be added (0 if the preselection may be answered as it is). */
    public static int needed(StateEncoder.Encoded e) {
        return Math.max(e.min - e.preselectedAnswers.size(), 0);
    }

    /** Number of options to sample: the new cards to add, or 1 (add one card, or Done). */
    public static int picks(StateEncoder.Encoded e) {
        return Math.max(needed(e), 1);
    }

    /**
     * Options excluded from sampling (outside undo): the preselected ones; when finishing is allowed also the "none" options
     * (an empty answer would drop the preselection, which the rule never answers).
     */
    public static List<Integer> excluded(StateEncoder.Encoded e) {
        List<Integer> out = new ArrayList<>();
        boolean finish = needed(e) == 0;
        for (int i = 0; i < e.preselected.size(); i++)
            if (e.preselected.get(i) || finish && e.isNone.get(i)) out.add(i);
        return out;
    }

    /** True when the policy chooses between adding a new card and finishing: PolicyAi then offers the Done option. */
    public static boolean offersDone(StateEncoder.Encoded e) {
        if (isUndo(e) || needed(e) > 0 || e.max - e.preselectedAnswers.size() < 1) return false;
        for (int i = 0; i < e.answers.size(); i++)
            if (!e.preselected.get(i) && !e.isNone.get(i)) return true;
        return false;
    }

    /** The answer that finishes with the preselection (what Done means). */
    public static String finish(StateEncoder.Encoded e) {
        return String.join(",", e.preselectedAnswers);
    }

    public static List<String> candidates(StateEncoder.Encoded e, List<Integer> ranking) {
        return candidates(e, ranking, -1);
    }

    /** As {@link #candidates(StateEncoder.Encoded, List)}; done = the index of the Done option, or -1 without one. */
    public static List<String> candidates(StateEncoder.Encoded e, List<Integer> ranking, int done) {
        int n = e.answers.size();
        List<Integer> rank = new ArrayList<>();
        for (int i : ranking)
            if (i >= 0 && i < n && !rank.contains(i)) rank.add(i);
        List<Integer> preRanked = new ArrayList<>(), newRanked = new ArrayList<>();
        for (int i : rank) {
            if (i == done) continue;
            if (e.preselected.get(i)) preRanked.add(i);
            else if (!e.isNone.get(i)) newRanked.add(i);
        }
        List<String> pre = e.preselectedAnswers;
        Map<String, String> out = new LinkedHashMap<>();
        if (isUndo(e)) {
            for (int j = preRanked.size() - 1; j >= 0; j--) {
                List<String> keep = new ArrayList<>(pre);
                keep.remove(e.answers.get(preRanked.get(j)));
                if (keep.size() >= e.min) put(out, String.join(",", keep));
            }
            if (e.min == 0) put(out, "");
            return new ArrayList<>(out.values());
        }
        int need = needed(e), maxAdd = e.max - pre.size();
        List<String> adds = new ArrayList<>();
        if (maxAdd >= 1) {
            int k = Math.min(Math.max(need, 1), maxAdd);
            for (int first = 0; first < newRanked.size(); first++) {
                List<String> answer = new ArrayList<>(pre);
                answer.add(e.answers.get(newRanked.get(first)));
                for (int j = 0; j < newRanked.size() && answer.size() < pre.size() + k; j++)
                    if (j != first) answer.add(e.answers.get(newRanked.get(j)));
                if (answer.size() == pre.size() + k) adds.add(String.join(",", answer));
            }
        }
        String finish = need == 0 ? finish(e) : null;
        boolean finishFirst = finish != null && (adds.isEmpty()
                || done >= 0 && rank.contains(done) && rank.indexOf(done) < rank.indexOf(newRanked.get(0)));
        if (finishFirst) put(out, finish);
        for (String a : adds) put(out, a);
        if (finish != null) put(out, finish);
        return new ArrayList<>(out.values());
    }

    private static void put(Map<String, String> out, String answer) {
        out.putIfAbsent(canonical(answer), answer);
    }

    /** An answer as an order-insensitive key: a card selection "b,a" is the same answer as "a,b". */
    public static String canonical(String answer) {
        if (answer.indexOf(',') < 0) return answer;
        String[] parts = answer.split(",");
        Arrays.sort(parts);
        return String.join(",", parts);
    }
}
