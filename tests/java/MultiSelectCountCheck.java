package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.framework.StartingSetup;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.PhysicalCardImpl;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Plan multiselect-count §1.2: the count choice with an explicit Done option, one real Gemp game per behaviour. PolicyAi
 * decides through its test seam (PolicyAi.Ranker: a scripted ranking instead of the policy); every policy query is
 * recorded (options, Done row, excluded options, k) and the one answer goes to Gemp, whose result is checked.
 * <ul>
 *   <li>docking bay transit (Docking Bay 94, "transit from here = 1 Force") with three Rebel Troopers: choose 2, then Done;
 *       Done at min (1 card); max reached (3 cards, no Done query after the third); cancel where Gemp's own min 0 allows
 *       it (the decision as Gemp sends it, without the training rule VisibleCancel); min 0, a card, then Done (cancel is
 *       not offered after the first card). Each transit pays 1 Force whatever the number of cards: the cost is per use;</li>
 *   <li>a Gemp-native stepwise decision (I've Lost Artoo! (V): a device that deploys on a character and/or an Effect with
 *       'skill', ARBITRARY_CARDS with a preselection): finished with Done (the preselection is resubmitted, one card taken),
 *       and adding the second card;</li>
 *   <li>StepwiseCandidates without the old convention: a preselected card ranked first no longer means finish.</li>
 * </ul>
 *
 *   SWCCG_V31_CLASSES=DIR bash scripts/efficient_java_check.sh com.gempukku.swccgo.bench.MultiSelectCountCheck
 */
public final class MultiSelectCountCheck {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** One recorded policy query. */
    static final class Step {
        final int options, done, k;
        final List<Integer> excluded;
        final List<String> answers;
        final boolean noneOffered;
        /** Stage B: the options marked "already selected" (model input), and the same read back from the record line. */
        final List<Integer> flagged = new ArrayList<>(), flaggedInLine = new ArrayList<>(), flaggedInTensors = new ArrayList<>();

        Step(StateEncoder.Encoded e, int k, List<Integer> excluded) {
            options = e.answers.size();
            check(e.tensorOptions.size() == options, "tensor options follow the options");
            int d = -1;
            for (int i = 0; i < options; i++)
                if (StateEncoder.DONE.equals(e.tensorOptions.get(i)[0])) d = i;
            check(d < 0 || d == options - 1, "Done is the last option");
            check(d < 0 || e.prefix.toString().endsWith("[\"" + StateEncoder.DONE + "\",[],-1,null,\"\",null]]"),
                "the record line carries the Done option row");
            done = d;
            this.k = k;
            this.excluded = excluded == null ? List.of() : List.copyOf(excluded);
            answers = List.copyOf(e.answers);
            boolean none = false;
            for (int i = 0; i < options; i++) if (e.isNone.get(i) && !this.excluded.contains(i)) none = true;
            noneOffered = none;
            for (int i = 0; i < options; i++) if (e.selected.get(i)) flagged.add(i);
            // the option array that ends the record line: a selected option carries a seventh field 1
            com.google.gson.JsonArray line = com.google.gson.JsonParser.parseString(e.line("g", 0, List.of())).getAsJsonArray();
            com.google.gson.JsonArray opts = line.get(9).getAsJsonArray();
            check(opts.size() == options, "record options follow the options");
            for (int i = 0; i < options; i++) {
                com.google.gson.JsonArray o = opts.get(i).getAsJsonArray();
                check(o.size() == 6 || o.size() == 7 && o.get(6).getAsInt() == 1, "option field count");
                if (o.size() == 7) flaggedInLine.add(i);
            }
            List<Object[]> tensors = BinaryObservation.withSelected(e);
            for (int i = 0; i < options; i++) if ((Integer) tensors.get(i)[5] == 1) flaggedInTensors.add(i);
            check(flagged.equals(flaggedInLine) && flagged.equals(flaggedInTensors), "flags agree: " + flagged + " "
                + flaggedInLine + " " + flaggedInTensors);
        }
    }

    /** Scripted policy: per query, the chosen option (by the step and the encoding) first, then the other options. */
    static final class Script implements PolicyAi.Ranker {
        final List<Step> steps = new ArrayList<>();
        final BiFunction<Integer, StateEncoder.Encoded, Integer> choose;

        Script(BiFunction<Integer, StateEncoder.Encoded, Integer> choose) {
            this.choose = choose;
        }

        @Override
        public List<Integer> rank(StateEncoder.Encoded e, int k, int repeats, boolean skip, List<Integer> excluded) {
            Step s = new Step(e, k, excluded);
            steps.add(s);
            int pick = choose.apply(steps.size() - 1, e);
            check(!s.excluded.contains(pick), "the script never picks an excluded option");
            List<Integer> out = new ArrayList<>(List.of(pick));
            for (int i = 0; i < e.answers.size(); i++) if (i != pick) out.add(i);
            return out;
        }
    }

    static int indexOf(StateEncoder.Encoded e, String answer) {
        int i = e.answers.indexOf(answer);
        check(i >= 0, "option " + answer + " offered");
        return i;
    }

    static int done(StateEncoder.Encoded e) {
        int last = e.answers.size() - 1;
        check(StateEncoder.DONE.equals(e.tensorOptions.get(last)[0]), "Done offered");
        return last;
    }

    static int none(StateEncoder.Encoded e) {
        for (int i = 0; i < e.answers.size(); i++) if (e.isNone.get(i)) return i;
        throw new AssertionError("no cancel option");
    }

    // ---------------------------------------------------------------- docking bay transit

    /**
     * Passes until the player's own phase action question in that phase (the framework's SkipToPhase does not know the
     * training overlay's activation rule, which asks the off-player an activation amount): activation questions take the
     * maximum, action questions pass, anything else BeginnerAi answers.
     */
    static void skipTo(VirtualTableScenario scn, String player, Phase phase) {
        com.gempukku.swccgo.ai.models.BeginnerAi ai = new com.gempukku.swccgo.ai.models.BeginnerAi();
        ai.setGame(scn.game());
        for (int i = 0; i < 1000; i++) {
            if (scn.gameState().getCurrentPhase() == phase && player.equals(scn.gameState().getCurrentPlayerId())) {
                AwaitingDecision own = scn.userFeedback().getAwaitingDecision(player);
                if (own != null && own.getText() != null && own.getText().contains("action or Pass")) return;
            }
            String who = scn.userFeedback().getAwaitingDecision(scn.DS) != null ? scn.DS : scn.LS;
            AwaitingDecision d = scn.userFeedback().getAwaitingDecision(who);
            check(d != null, "a decision while skipping");
            String type = d.getDecisionType().name();
            String answer;
            if (type.equals("INTEGER")) answer = d.getDecisionParameters().get("max")[0];
            else if (type.equals("CARD_ACTION_CHOICE") || type.equals("ACTION_CHOICE") && d.getText().contains("action or Pass"))
                answer = "";
            else answer = ai.decide(who, d, scn.gameState());
            scn.PlayerDecided(who, answer);
        }
        throw new AssertionError("did not reach " + phase + " of " + player);
    }

    static VirtualTableScenario transitScenario() {
        HashMap<String, String> ls = new HashMap<>();
        ls.put("tat_db", "1_129"); // Docking Bay 94: docking bay transit from here = 1 Force
        ls.put("yavin4_db", "1_136"); // Yavin 4: Docking Bay: transit from here free
        return new VirtualTableScenario(ls, new HashMap<>(), 10, 10, StartingSetup.DefaultLSGroundLocation,
            StartingSetup.DefaultDSGroundLocation, StartingSetup.NoLSStartingInterrupts, StartingSetup.NoDSStartingInterrupts,
            StartingSetup.NoLSShields, StartingSetup.NoDSShields, VirtualTableScenario.Open);
    }

    /** Result of one transit: the Gemp card selection's min / max, the moved troopers, the Force paid, the queries. */
    static final class Transit {
        int min, max, moved, forcePaid;
        String answer;
        List<Step> steps;
    }

    /**
     * Three Rebel Troopers at Docking Bay 94, LS move phase: transit to Yavin 4: Docking Bay; PolicyAi answers the card
     * selection with the script. visibleCancel: apply the training rule (min 0 -> 1 when every card is visible).
     */
    static Transit transit(BiFunction<Integer, StateEncoder.Encoded, Integer> choose, boolean visibleCancel) {
        VirtualTableScenario scn = transitScenario();
        PhysicalCardImpl[] troopers = {scn.GetLSFiller(1), scn.GetLSFiller(2), scn.GetLSFiller(3)};
        PhysicalCardImpl tat = scn.GetLSCard("tat_db"), yavin = scn.GetLSCard("yavin4_db");
        scn.StartGame();
        scn.MoveLocationToTable(tat);
        scn.MoveLocationToTable(yavin);
        scn.MoveCardsToLocation(tat, troopers);
        skipTo(scn, scn.LS, Phase.MOVE);
        int force = scn.GetLSForcePileCount();
        check(force >= 1, "Force to pay the transit");
        scn.LSUseCardAction(tat, "transit");
        check(scn.LSDecisionAvailable("transit to"), "destination question");
        scn.LSChooseCard(yavin);
        AwaitingDecision d = scn.LSGetDecision();
        check(d.getText().startsWith("Choose cards to docking bay transit to "), "card question: " + d.getText());
        if (visibleCancel) com.gempukku.swccgo.game.rules.VisibleCancel.restrict(d, scn.LS);
        Transit t = new Transit();
        t.min = Integer.parseInt(d.getDecisionParameters().get("min")[0]);
        t.max = Integer.parseInt(d.getDecisionParameters().get("max")[0]);
        PolicyAi ai = new PolicyAi("localhost", 0);
        Script script = new Script(choose);
        ai.ranker = script;
        ai.setGame(scn.game());
        t.answer = ai.decide(scn.LS, d, scn.gameState());
        t.steps = script.steps;
        scn.PlayerDecided(scn.LS, t.answer);
        for (int i = 0; i < 40 && !scn.AwaitingDSMovePhaseActions() && !scn.AwaitingLSMovePhaseActions(); i++) {
            if (scn.DSAnyDecisionsAvailable()) scn.DSPass();
            else if (scn.LSAnyDecisionsAvailable()) scn.LSPass();
        }
        for (PhysicalCard c : troopers)
            if (c.getAtLocation() == yavin) t.moved++;
        t.forcePaid = force - scn.GetLSForcePileCount();
        return t;
    }

    /** First trooper option not yet chosen (the troopers' card ids in option order), by step. */
    static int card(StateEncoder.Encoded e, int n) {
        int seen = 0;
        for (int i = 0; i < e.answers.size(); i++)
            if (!e.isNone.get(i) && !StateEncoder.DONE.equals(e.tensorOptions.get(i)[0]) && seen++ == n) return i;
        throw new AssertionError("no card " + n);
    }

    static void twoThenDone() {
        Transit t = transit((step, e) -> step < 2 ? card(e, step) : done(e), true);
        check(t.min == 1 && t.max == 3, "min 1 (training rule) / max 3: " + t.min + " / " + t.max);
        check(t.steps.size() == 3, "three queries: two cards, then Done: " + t.steps.size());
        check(t.steps.get(0).done < 0 && t.steps.get(0).excluded.isEmpty(), "first step: no Done, nothing excluded");
        check(t.steps.get(1).done == 3 && t.steps.get(1).excluded.equals(List.of(0)), "second step: Done offered, first card excluded");
        check(t.steps.get(2).done == 3 && t.steps.get(2).excluded.equals(List.of(0, 1)), "third step: both cards excluded");
        check(t.steps.get(0).flagged.isEmpty() && t.steps.get(1).flagged.equals(List.of(0))
            && t.steps.get(2).flagged.equals(List.of(0, 1)), "stage B: the selected cards are flagged in each step");
        for (Step s : t.steps) check(s.k == 1, "one card per query");
        check(t.answer.split(",").length == 2, "one answer with two cards: " + t.answer);
        check(t.moved == 2, "Gemp moved two troopers: " + t.moved);
        check(t.forcePaid == 1, "two cards cost 1 Force (per use): " + t.forcePaid);
    }

    static void doneAtMin() {
        Transit t = transit((step, e) -> step == 0 ? card(e, 0) : done(e), true);
        check(t.steps.size() == 2, "two queries");
        check(t.steps.get(1).done >= 0, "Done offered once the count reaches min 1");
        check(t.moved == 1 && t.forcePaid == 1, "one trooper for 1 Force: " + t.moved + " / " + t.forcePaid);
    }

    static void maxReached() {
        Transit t = transit((step, e) -> card(e, step), true);
        check(t.steps.size() == 3, "three queries, none after max: " + t.steps.size());
        check(t.steps.get(2).options - t.steps.get(2).excluded.size() == 2, "last query: the last card or Done");
        check(t.answer.split(",").length == 3, "max = 3 cards: " + t.answer);
        check(t.moved == 3, "Gemp moved three troopers: " + t.moved);
        check(t.forcePaid == 1, "three cards cost 1 Force (per use): " + t.forcePaid);
    }

    static void cancel() {
        Transit t = transit((step, e) -> none(e), false);
        check(t.min == 0, "Gemp's own decision allows cancel (min 0): " + t.min);
        check(t.steps.size() == 1 && t.steps.get(0).noneOffered && t.steps.get(0).done < 0, "cancel offered, Done not");
        check(t.answer.isEmpty(), "cancel = the empty answer");
        check(t.moved == 0 && t.forcePaid == 0, "nothing moved, nothing paid: " + t.moved + " / " + t.forcePaid);
    }

    static void minZeroCardThenDone() {
        Transit t = transit((step, e) -> step == 0 ? card(e, 0) : done(e), false);
        check(t.min == 0 && t.steps.size() == 2, "min 0, two queries");
        check(t.steps.get(0).noneOffered, "first step offers cancel");
        check(!t.steps.get(1).noneOffered && t.steps.get(1).done >= 0, "second step: Done, cancel excluded");
        check(t.moved == 1 && t.forcePaid == 1, "one trooper moved");
    }

    // ---------------------------------------------------------------- Gemp-native stepwise (I've Lost Artoo! (V))

    static final class Artoo {
        List<Step> steps = new ArrayList<>();
        int inHand, decisions;
        List<String> types = new ArrayList<>(), seen = new ArrayList<>();
    }

    /** chooseSecond: at the stepwise question, add the second card (true) or finish with Done (false). */
    static Artoo artoo(boolean chooseSecond) {
        HashMap<String, String> ds = new HashMap<>();
        ds.put("artoo", "601_157"); // I've Lost Artoo! (V)
        ds.put("comlink", "1_201"); // Comlink: a device that deploys on a character
        ds.put("promotion", "4_121"); // Field Promotion: an Effect with 'skill' in its game text
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>(), ds, 10, 10, StartingSetup.DefaultLSGroundLocation,
            StartingSetup.DefaultDSGroundLocation, StartingSetup.NoLSStartingInterrupts, StartingSetup.NoDSStartingInterrupts,
            StartingSetup.NoLSShields, StartingSetup.NoDSShields, VirtualTableScenario.Open);
        PhysicalCardImpl artoo = scn.GetDSCard("artoo"), comlink = scn.GetDSCard("comlink"), promotion = scn.GetDSCard("promotion");
        scn.StartGame();
        scn.MoveCardsToDSSideOfTable(artoo);
        skipTo(scn, scn.DS, Phase.DEPLOY);
        // after activation (which draws the top of the Reserve Deck into the Force Pile)
        scn.MoveCardsToTopOfDSReserveDeck(comlink, promotion);
        check(List.of(comlink.getZone(), promotion.getZone()).stream().allMatch(z -> z == Zone.RESERVE_DECK || z == Zone.TOP_OF_RESERVE_DECK), "both in the Reserve Deck: " + comlink.getZone() + " " + promotion.getZone());
        check(scn.DSCardActionAvailable(artoo, "Take cards"), "I've Lost Artoo! (V) offers its take action");
        scn.DSUseCardAction(artoo, "Take cards");
        Artoo a = new Artoo();
        PolicyAi ai = new PolicyAi("localhost", 0);
        Script script = new Script((step, e) -> {
            if (step == 0) return 0; // the first card of the combination
            return chooseSecond ? card(e, 1) : done(e);
        });
        ai.ranker = script;
        ai.setGame(scn.game());
        com.gempukku.swccgo.ai.models.BeginnerAi other = new com.gempukku.swccgo.ai.models.BeginnerAi();
        other.setGame(scn.game());
        for (int i = 0; i < 40; i++) {
            String who = scn.userFeedback().getAwaitingDecision(scn.DS) != null ? scn.DS : scn.LS;
            AwaitingDecision d = scn.userFeedback().getAwaitingDecision(who);
            if (d == null || d.getText() != null && d.getText().contains("action or Pass") && a.decisions > 0) break;
            if (d.getText() == null || !d.getText().startsWith("Choose a device")) {
                a.seen.add(who + ": " + d.getDecisionType() + " " + d.getText());
                String type = d.getDecisionType().name();
                scn.PlayerDecided(who, type.equals("CARD_ACTION_CHOICE") ? "" : other.decide(who, d, scn.gameState()));
                continue;
            }
            a.types.add(d.getDecisionType() + " min " + d.getDecisionParameters().get("min")[0] + " max "
                + d.getDecisionParameters().get("max")[0] + " preselected " + count(d.getDecisionParameters().get("preselected")));
            a.decisions++;
            scn.PlayerDecided(scn.DS, ai.decide(scn.DS, d, scn.gameState()));
        }
        a.steps = script.steps;
        for (PhysicalCard c : List.of(comlink, promotion)) if (c.getZone() == Zone.HAND) a.inHand++;
        return a;
    }

    static int count(String[] flags) {
        int n = 0;
        if (flags != null) for (String f : flags) if ("true".equals(f)) n++;
        return n;
    }

    static void gempStepwiseDone() {
        long offered = PolicyAi.GEMP_STEPWISE_DONE_OFFERED.get(), done = PolicyAi.GEMP_STEPWISE_DONE.get();
        Artoo a = artoo(false);
        check(a.decisions == 2, "Gemp asks twice (first card, then add or finish): " + a.types + " other " + a.seen);
        check(a.types.get(1).contains("preselected 1"), "the second question is stepwise: " + a.types);
        check(a.steps.size() == 2 && a.steps.get(0).done < 0 && a.steps.get(1).done >= 0, "Done offered on the stepwise question");
        check(a.steps.get(1).excluded.size() == 1, "the preselected card is excluded: " + a.steps.get(1).excluded);
        check(a.steps.get(0).flagged.isEmpty() && a.steps.get(1).flagged.equals(a.steps.get(1).excluded),
            "stage B: Gemp's preselected card is flagged: " + a.steps.get(1).flagged);
        check(a.inHand == 1, "Done resubmitted the preselection: one card taken, " + a.inHand);
        check(PolicyAi.GEMP_STEPWISE_DONE_OFFERED.get() == offered + 1 && PolicyAi.GEMP_STEPWISE_DONE.get() == done + 1, "counters");
        Artoo b = artoo(true);
        check(b.inHand == 2, "adding the second card takes both: " + b.inHand);
    }

    /** The old convention is gone: a preselected card ranked first is not a finish; Done ranked first is. */
    static void noPreselectedFinish() {
        StateEncoder.Encoded e = new StateEncoder.Encoded();
        e.type = "ARBITRARY_CARDS";
        e.min = 1;
        e.max = 2;
        for (String id : List.of("temp0", "temp1", "temp2")) {
            e.answers.add(id);
            e.isNone.add(false);
            e.preselected.add(id.equals("temp0"));
        }
        e.preselectedAnswers.add("temp0");
        check(StepwiseCandidates.offersDone(e), "finish allowed and a card may be added: Done offered");
        check(StepwiseCandidates.excluded(e).equals(List.of(0)), "the preselected option is excluded");
        check(StepwiseCandidates.candidates(e, List.of(0, 2, 1)).get(0).equals("temp0,temp2"),
            "a preselected card ranked first adds the best new card (no finish convention)");
        int done = 3;
        e.answers.add("");
        e.isNone.add(false);
        e.preselected.add(false);
        check(StepwiseCandidates.candidates(e, List.of(3, 2, 1, 0), done).get(0).equals("temp0"), "Done first = finish");
        check(StepwiseCandidates.candidates(e, List.of(1, 3, 2, 0), done).get(0).equals("temp0,temp1"), "a new card first = add");
    }


    // ---------------------------------------------------------------- Collision! (Amendment 1: take max)

    /**
     * Two Victory-class Star Destroyers at the Yavin 4 system, LS plays Collision! (Card1_074). Gemp's "Choose starships"
     * needs all of them (min = max = 2 inside the effect; sent with min 0, raised to 1 by VisibleCancel). PolicyAi takes max
     * (one query, both starships) and the play goes through (1 Force, Collision! to the Lost Pile); a one-starship answer
     * (what the count choice could pick) cancels the play for free.
     */
    static String[] collision(boolean policy) {
        HashMap<String, String> ls = new HashMap<>();
        ls.put("yavin", "1_135");
        ls.put("collision", "1_074");
        HashMap<String, String> ds = new HashMap<>();
        ds.put("vcsd1", "2_155");
        ds.put("vcsd2", "2_155");
        VirtualTableScenario scn = new VirtualTableScenario(ls, ds, 10, 10, StartingSetup.DefaultLSGroundLocation,
            StartingSetup.DefaultDSGroundLocation, StartingSetup.NoLSStartingInterrupts, StartingSetup.NoDSStartingInterrupts,
            StartingSetup.NoLSShields, StartingSetup.NoDSShields, VirtualTableScenario.Open);
        PhysicalCardImpl yavin = scn.GetLSCard("yavin"), collision = scn.GetLSCard("collision");
        scn.StartGame();
        scn.MoveLocationToTable(yavin);
        scn.MoveCardsToLocation(yavin, scn.GetDSCard("vcsd1"), scn.GetDSCard("vcsd2"));
        scn.MoveCardToZone(scn.LS, collision, Zone.HAND); // before the phase question lists the playable cards
        skipTo(scn, scn.LS, Phase.DEPLOY);
        check(collision.getZone() == Zone.HAND, "Collision! still in hand");
        int force = scn.GetLSForcePileCount();
        check(scn.LSCardActionAvailable(collision), "Collision! playable: " + yavin.getTitle() + " " + yavin.getBlueprint().getCardSubtype()
            + " vcsd " + scn.GetDSCard("vcsd1").getZone() + " at " + scn.GetDSCard("vcsd1").getAtLocation() + " force " + force
            + " phase " + scn.gameState().getCurrentPhase() + " actions " + scn.GetLSAvailableActions());
        scn.LSUseCardAction(collision);
        PolicyAi ai = new PolicyAi("localhost", 0);
        Script script = new Script((step, e) -> 0);
        ai.ranker = script;
        ai.setGame(scn.game());
        com.gempukku.swccgo.ai.models.BeginnerAi other = new com.gempukku.swccgo.ai.models.BeginnerAi();
        other.setGame(scn.game());
        String answer = null, params = null;
        for (int i = 0; i < 60; i++) {
            String who = scn.userFeedback().getAwaitingDecision(scn.LS) != null ? scn.LS : scn.DS;
            AwaitingDecision d = scn.userFeedback().getAwaitingDecision(who);
            if (d == null || answer != null && who.equals(scn.LS) && d.getText().contains("action or Pass")) break;
            if (who.equals(scn.LS) && d.getText().startsWith(PolicyAi.COLLISION)) {
                com.gempukku.swccgo.game.rules.VisibleCancel.restrict(d, scn.LS);
                params = d.getDecisionParameters().get("min")[0] + "/" + d.getDecisionParameters().get("max")[0];
                answer = policy ? ai.decide(scn.LS, d, scn.gameState()) : d.getDecisionParameters().get("cardId")[0];
                scn.PlayerDecided(scn.LS, answer);
                continue;
            }
            if (who.equals(scn.LS) && d.getText().startsWith("Choose location in order to")) {
                scn.PlayerDecided(scn.LS, String.valueOf(yavin.getCardId()));
                continue;
            }
            String type = d.getDecisionType().name();
            scn.PlayerDecided(who, type.equals("CARD_ACTION_CHOICE") ? "" : other.decide(who, d, scn.gameState()));
        }
        check(answer != null, "Gemp asked which starships");
        return new String[]{answer, params, String.valueOf(force - scn.GetLSForcePileCount()), String.valueOf(collision.getZone()),
            String.valueOf(script.steps.size())};
    }

    static void collisionTakesMax() {
        String[] r = collision(true);
        check(r[1].equals("1/2"), "min 1 (VisibleCancel) / max 2: " + r[1]);
        check(r[4].equals("1"), "one query (take max, not step by step): " + r[4]);
        check(r[0].split(",").length == 2, "both starships: " + r[0]);
        check(r[2].equals("1") && r[3].endsWith("LOST_PILE"), "the play went through: Force " + r[2] + ", Collision! in " + r[3]);
        String[] one = collision(false);
        check(one[2].equals("0") && one[3].equals("HAND"), "one starship cancels the play: Force " + one[2] + ", Collision! in " + one[3]);
    }

    public static void main(String[] args) {
        twoThenDone();
        doneAtMin();
        maxReached();
        cancel();
        minZeroCardThenDone();
        gempStepwiseDone();
        noPreselectedFinish();
        collisionTakesMax();
        System.out.println("{\"passed\":true,\"transit_two_then_done\":true,\"done_at_min\":true,\"max_reached\":true,"
            + "\"cancel_min0\":true,\"min0_card_then_done\":true,\"transit_force_per_use\":1,\"gemp_stepwise_done\":true,"
            + "\"no_preselected_finish\":true,\"selected_flags\":true,\"collision_take_max\":true}");
    }
}
