package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.ai.models.BeginnerAi;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.framework.StartingSetup;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 * Plan facedown-fixes §1.1: the set-up stack "deploy on table with up to 10 cards from outside your deck face-down under
 * here" (Fear Is My Ally, 13_69) with 12 Defensive Shields outside the deck. Gemp asks "Choose cards to stack under" with
 * min 1 / max 10. PolicyAi's pick count and candidate rule must answer 10 cards for any ranking (best-first and reversed),
 * and Gemp must stack all 10 face down. With 10 or fewer shields Gemp stacks them all without asking (no decision).
 * A "choose up to N" selection of another kind keeps max(min, 1).
 *
 *   SWCCG_V31_CLASSES=DIR bash scripts/efficient_java_check.sh com.gempukku.swccgo.bench.SetupStackPicksCheck
 */
public final class SetupStackPicksCheck {
    private static final List<String> SHIELDS = List.of("13_90", "13_81", "13_100", "13_61", "13_52", "13_66", "13_68",
        "13_95", "13_51", "13_84", "13_99", "13_98");

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static StartingSetup shields(int count) {
        return new StartingSetup() {
            @Override
            public HashMap<String, String> Cards() {
                HashMap<String, String> cards = new HashMap<>();
                for (int i = 0; i < count; i++) cards.put("shield-" + i, SHIELDS.get(i));
                return cards;
            }

            @Override
            public void Setup(VirtualTableScenario scn) {
            }
        };
    }

    /** The test framework names only deck cards; this table also names the cards outside the deck. */
    private static final class Table extends VirtualTableScenario {
        Table(HashMap<String, String> ds, StartingSetup shields) {
            super(new HashMap<>(), ds, 10, 10, StartingSetup.DefaultLSGroundLocation, StartingSetup.DefaultDSGroundLocation,
                StartingSetup.NoLSStartingInterrupts, StartingSetup.NoDSStartingInterrupts, StartingSetup.NoLSShields, shields,
                VirtualTableScenario.Open);
        }

        @Override
        public void ResetGameState(HashMap<String, String> ls, HashMap<String, String> ds) {
            HashMap<String, String> named = new HashMap<>(ds);
            int i = 0;
            for (PhysicalCard c : game().getGameState().getAllPermanentCards())
                if (c.getZone() == Zone.OUTSIDE_OF_DECK && c.getBlueprint().getSide() == com.gempukku.swccgo.common.Side.DARK)
                    named.put("outside-" + i++, c.getBlueprintId(true));
            super.ResetGameState(ls, named);
        }
    }

    private static VirtualTableScenario scenario(int shields) {
        HashMap<String, String> ds = new HashMap<>();
        ds.put("fear", "13_69");
        return new Table(ds, shields(shields));
    }

    /** Plays the game start with BeginnerAi until the stack decision (returned) or until Fear Is My Ally is on table. */
    private static AwaitingDecision untilStack(VirtualTableScenario scn) {
        BeginnerAi ai = new BeginnerAi();
        PhysicalCard fear = scn.GetDSCard("fear");
        for (int step = 0; step < 200; step++) {
            boolean answered = false;
            for (String player : new String[]{scn.DS, scn.LS}) {
                AwaitingDecision d = scn.userFeedback().getAwaitingDecision(player);
                if (d == null) continue;
                if (d.getText() != null && d.getText().startsWith(PolicyAi.STACK_UNDER)) return d;
                if (fear.getZone() != null && fear.getZone().isInPlay()) return null;
                scn.PlayerDecided(player, ai.decide(player, d, scn.gameState()));
                answered = true;
                break;
            }
            if (!answered) break;
        }
        return null;
    }

    private static int stackedUnder(PhysicalCard card, VirtualTableScenario scn) {
        int n = 0;
        for (PhysicalCard c : scn.gameState().getAllPermanentCards())
            if (c.getStackedOn() == card && c.getZone() == Zone.STACKED_FACE_DOWN) n++;
        return n;
    }

    private static void moreShieldsThanMax(boolean reversed) {
        VirtualTableScenario scn = scenario(12);
        AwaitingDecision d = untilStack(scn);
        check(d != null, "Gemp asks which shields to stack with 12 outside the deck");
        StateEncoder.Encoded e = StateEncoder.optionsOnly(d);
        check(e.min == 1 && e.max == 10, "min 1 / max 10: " + e.min + " / " + e.max);
        check(!StepwiseCandidates.applies(e), "not a stepwise selection");
        int k = PolicyAi.multiPicks(e, d.getText());
        check(k == 10, "picks min(N, available) = 10, got " + k);
        List<Integer> ranking = new ArrayList<>();
        for (int i = 0; i < e.answers.size(); i++) ranking.add(i);
        if (reversed) Collections.reverse(ranking);
        String answer = PolicyAi.candidates(e, ranking, k).get(0);
        check(answer.split(",").length == 10, "answer with 10 cards: " + answer);
        scn.PlayerDecided(scn.DS, answer);
        check(stackedUnder(scn.GetDSCard("fear"), scn) == 10, "Gemp stacked 10 shields: " + stackedUnder(scn.GetDSCard("fear"), scn));
    }

    private static void allFit() {
        VirtualTableScenario scn = scenario(8);
        check(untilStack(scn) == null, "no question when every shield fits");
        check(stackedUnder(scn.GetDSCard("fear"), scn) == 8, "Gemp stacked all 8 shields itself");
    }

    private static void otherUpToN() {
        StateEncoder.Encoded e = new StateEncoder.Encoded();
        e.type = "CARD_SELECTION";
        e.min = 1;
        e.max = 3;
        for (int i = 0; i < 4; i++) {
            e.answers.add(String.valueOf(i));
            e.isNone.add(false);
        }
        check(PolicyAi.multiPicks(e, "Choose cards to docking bay transit to <c>, or click 'Done' to cancel") == 1,
            "other templates keep max(min, 1)");
        check(PolicyAi.multiPicks(e, "Choose cards to replace, or click 'Done' to cancel") == 3,
            "squadron replacement answers its replacement count (max)");
        e.min = 2;
        check(PolicyAi.multiPicks(e, "Choose 2 destiny") == 2, "min 2 keeps 2");
    }

    public static void main(String[] args) {
        moreShieldsThanMax(false);
        moreShieldsThanMax(true);
        allFit();
        otherUpToN();
        System.out.println("{\"passed\":true,\"setup_stack_takes_10_of_12\":true,\"all_fit_without_question\":true,"
            + "\"other_templates_unchanged\":true}");
    }
}
