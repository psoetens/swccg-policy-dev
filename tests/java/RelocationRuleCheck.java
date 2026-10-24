import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.rules.FreeMoveRule;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.HashMap;

/**
 * Plan relocation-rule: a card's own top-level "Relocate ..." game-text action is offered until that card was relocated
 * by it in the current phase, then not again in that phase; the next phase offers it again. Stolen Data Tapes (203_14,
 * "If at Dune Sea, may relocate to your character there") on Luke at Dune Sea, with Leia there too. The scenario
 * framework does not call the rule's hooks (TrainGen and Brainiac do), so the check calls restrict / decided itself.
 *
 *   bash scripts/v31_engine.sh RelocationRuleCheck   (after scripts/efficient_java_check.sh)
 */
public final class RelocationRuleCheck {
    private static final String RELOCATE = "Relocate to your character";

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** The LS action menu after the rule filtered it; the action id of the tapes' relocation, or null. */
    private static String offered(VirtualTableScenario scn) {
        AwaitingDecision decision = scn.LSGetDecision();
        check(decision != null, "LS has a decision");
        FreeMoveRule.restrict(scn.game(), decision);
        return scn.GetCardActionId(scn.LS, scn.GetLSCard("tapes"), RELOCATE);
    }

    private static void passOptionalResponses(VirtualTableScenario scn) {
        for (int i = 0; i < 10; i++) {
            if (scn.LSDecisionAvailable("Optional responses")) scn.LSPass();
            else if (scn.DSDecisionAvailable("Optional responses")) scn.DSPass();
            else return;
        }
    }

    public static void main(String[] args) {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>() {{
            put("tapes", "203_014");
            put("dunesea", "1_130");
            put("luke", "1_019");
            put("leia", "1_017");
        }}, new HashMap<>());
        var tapes = scn.GetLSCard("tapes");
        var dunesea = scn.GetLSCard("dunesea");
        var luke = scn.GetLSCard("luke");
        var leia = scn.GetLSCard("leia");
        scn.StartGame();
        scn.MoveLocationToTable(dunesea);
        scn.MoveCardsToLocation(dunesea, luke, leia);
        scn.AttachCardsTo(luke, tapes);
        // DS's first turn (no turn skipping: the knowledge engine's activation rule asks for an amount first). The tapes
        // relocate "at any time", so LS's menus in DS's phases offer it.
        scn.DSDecided(0);
        passOptionalResponses(scn);
        scn.DSPass();
        passOptionalResponses(scn);
        check(scn.GetCurrentPhase() == Phase.ACTIVATE && scn.AwaitingLSActivatePhaseActions(), "LS's menu in DS's Activate phase");

        String id = offered(scn);
        check(id != null, "the relocation is offered before it was used this phase");
        AwaitingDecision decision = scn.LSGetDecision();
        FreeMoveRule.decided(scn.game(), decision, id);
        scn.LSDecided(id);
        scn.LSChooseCard(leia);
        passOptionalResponses(scn);
        check(tapes.getAttachedTo() == leia, "the tapes moved to Leia (on " + tapes.getAttachedTo() + ")");
        if (scn.AwaitingDSActivatePhaseActions()) {
            scn.DSPass();
            passOptionalResponses(scn);
        }
        check(scn.GetCurrentPhase() == Phase.ACTIVATE && scn.AwaitingLSActivatePhaseActions(), "LS's menu again, same phase");
        check(offered(scn) == null, "the relocation is not offered again in the same phase");

        scn.LSPass();
        passOptionalResponses(scn);
        if (scn.AwaitingDSActivatePhaseActions()) scn.DSPass();
        passOptionalResponses(scn);
        for (int i = 0; i < 5 && !(scn.GetCurrentPhase() == Phase.CONTROL && scn.LSGetDecision() != null); i++) {
            if (scn.DSGetDecision() != null) scn.DSPass(); else break;
            passOptionalResponses(scn);
        }
        check(scn.GetCurrentPhase() == Phase.CONTROL && scn.LSGetDecision() != null, "LS's menu in DS's Control phase (now "
                + scn.GetCurrentPhase() + ")");
        check(offered(scn) != null, "the next phase offers the relocation again");
        System.out.println("RelocationRuleCheck: OK");
    }
}
