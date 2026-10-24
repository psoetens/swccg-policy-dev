import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.rules.FreeMoveRule;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.HashMap;

/**
 * Plan capacity-slot-rule: Gemp offers the capacity-slot move in the owner's deploy phase too. Sergeant Irol (8_113)
 * piloting Tempest 1 (8_170) moves to a passenger slot in DS's deploy phase; the move back is then not offered again in
 * that deploy phase; the following move phase offers a slot move again. The scenario framework does not call the rule's
 * hooks (TrainGen and Brainiac do), so the check calls restrict / decided itself.
 *
 *   bash scripts/v31_engine.sh CapacitySlotRuleCheck   (after scripts/efficient_java_check.sh)
 */
public final class CapacitySlotRuleCheck {
    private static final String TO_PASSENGER = "Move to passenger capacity slot";
    private static final String TO_PILOT = "Move to pilot capacity slot";

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** DS's action menu after the rule filtered it; the action id of Irol's slot move with this text, or null. */
    private static String offered(VirtualTableScenario scn, String text) {
        AwaitingDecision decision = scn.DSGetDecision();
        check(decision != null, "DS has a decision");
        FreeMoveRule.restrict(scn.game(), decision);
        return scn.GetCardActionId(scn.DS, scn.GetDSCard("irol"), text);
    }

    private static void passOptionalResponses(VirtualTableScenario scn) {
        for (int i = 0; i < 10; i++) {
            if (scn.LSDecisionAvailable("Optional responses")) scn.LSPass();
            else if (scn.DSDecisionAvailable("Optional responses")) scn.DSPass();
            else return;
        }
    }

    /** Pass until DS has its own action menu in the given phase of its turn. */
    private static void passTo(VirtualTableScenario scn, Phase phase) {
        for (int i = 0; i < 30; i++) {
            passOptionalResponses(scn);
            if (scn.GetCurrentPhase() == phase && scn.DSGetDecision() != null) return;
            if (scn.DSGetDecision() != null) scn.DSPass();
            else if (scn.LSGetDecision() != null) scn.LSPass();
        }
        check(false, "DS's menu in its " + phase + " phase (now " + scn.GetCurrentPhase() + ")");
    }

    public static void main(String[] args) {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>() {{
            put("site", "1_130");  // a Light Side site: Gemp refuses it in the Dark Side deck
        }}, new HashMap<>() {{
            put("tempest", "8_170");
            put("irol", "8_113");
        }});
        var site = scn.GetLSCard("site");
        var tempest = scn.GetDSCard("tempest");
        var irol = scn.GetDSCard("irol");
        scn.StartGame();
        scn.MoveLocationToTable(site);
        scn.MoveCardsToLocation(site, tempest);
        scn.BoardAsPilot(tempest, irol);
        check(irol.isPilotOf(), "Irol pilots Tempest 1");
        // DS's first turn (no turn skipping: the knowledge engine's activation rule asks for an amount first).
        scn.DSDecided(0);
        passTo(scn, Phase.DEPLOY);

        String id = offered(scn, TO_PASSENGER);
        check(id != null, "the move to a passenger slot is offered in the deploy phase");
        AwaitingDecision decision = scn.DSGetDecision();
        FreeMoveRule.decided(scn.game(), decision, id);
        scn.DSDecided(id);
        passOptionalResponses(scn);
        check(irol.isPassengerOf(), "Irol is a passenger now");
        if (scn.GetCurrentPhase() == Phase.DEPLOY && scn.DSGetDecision() == null && scn.LSGetDecision() != null) {
            scn.LSPass();
            passOptionalResponses(scn);
        }
        check(scn.GetCurrentPhase() == Phase.DEPLOY && scn.DSGetDecision() != null, "DS's menu again, same deploy phase");
        check(offered(scn, TO_PILOT) == null, "the move back to the pilot slot is not offered again in the deploy phase");

        scn.DSPass();
        passTo(scn, Phase.MOVE);
        check(offered(scn, TO_PILOT) != null, "the move phase offers the move to the pilot slot again");
        System.out.println("CapacitySlotRuleCheck: OK");
    }
}
