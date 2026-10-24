import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.framework.VirtualTableScenario;

import java.util.HashMap;

/**
 * Plan critic-replay, E-1 check 1: the own Activate phase starts with "Choose amount of Force to activate" (0 … max);
 * the answer (0 included) is the turn's activation, "Activate Force" is not offered again, and Pass ends the phase
 * without the "You have not activated Force" check. The opponent's Activate-phase menu is unaffected.
 *
 *   bash scripts/v31_engine.sh ActivationRuleCheck   (after scripts/efficient_java_check.sh)
 */
public final class ActivationRuleCheck {
    private static final String AMOUNT = "Choose amount of Force to activate";
    private static final String CHECK = "You have not activated Force";

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static VirtualTableScenario scenario() {
        return new VirtualTableScenario(new HashMap<>(), new HashMap<>());
    }

    private static void passOptionalResponses(VirtualTableScenario scn) {
        for (int i = 0; i < 10; i++) {
            if (scn.LSDecisionAvailable("Optional responses")) scn.LSPass();
            else if (scn.DSDecisionAvailable("Optional responses")) scn.DSPass();
            else return;
        }
    }

    private static void zeroThenPass() {
        VirtualTableScenario scn = scenario();
        scn.StartGame();
        check(scn.GetCurrentPhase() == Phase.ACTIVATE && scn.GetCurrentPlayer().equals(scn.DS), "game starts in DS Activate");
        check(scn.DSDecisionAvailable(AMOUNT), "DS's Activate phase starts with the amount question");
        check(Integer.parseInt(scn.DSGetFirstADParam("min")) == 0, "the amount question allows 0");
        int force = scn.GetDSForcePileCount();
        scn.DSDecided(0);
        passOptionalResponses(scn);
        check(scn.GetDSForcePileCount() == force, "0 activates nothing");
        check(scn.AwaitingDSActivatePhaseActions(), "the Activate menu follows");
        check(!scn.DSActionAvailable("Activate Force"), "Activate Force is not offered again after 0");
        scn.DSPass();
        check(!scn.DSDecisionAvailable(CHECK), "no 'You have not activated Force' check after Pass");
        check(scn.AwaitingLSActivatePhaseActions(), "the opponent gets its normal Activate-phase menu");
        check(!scn.LSDecisionAvailable(AMOUNT), "the opponent is not asked to activate in DS's turn");
        scn.LSPass();
        check(scn.GetCurrentPhase() == Phase.CONTROL, "two passes end the Activate phase (now " + scn.GetCurrentPhase() + ")");
        // LS's turn: its own Activate phase starts with the amount question as well.
        scn.SkipToPhase(Phase.DRAW);
        scn.PassDrawActions();
        scn.PassResponses("RECIRCULATED");
        check(scn.GetCurrentPlayer().equals(scn.LS) && scn.GetCurrentPhase() == Phase.ACTIVATE, "LS's turn reached");
        check(scn.LSDecisionAvailable(AMOUNT), "LS's Activate phase starts with the amount question");
        int lsForce = scn.GetLSForcePileCount();
        int max = scn.LSGetChoiceMax();
        check(max > 0, "LS can activate");
        scn.LSDecided(max);
        passOptionalResponses(scn);
        check(scn.GetLSForcePileCount() == lsForce + max, "LS activated " + max + " (pile " + lsForce + " -> " + scn.GetLSForcePileCount() + ")");
        // As after any top-level action, the opponent may act first; then LS gets its menu again.
        if (scn.AwaitingDSActivatePhaseActions()) scn.DSPass();
        check(scn.AwaitingLSActivatePhaseActions() && !scn.LSActionAvailable("Activate Force"), "menu without Activate Force after activating");
        check(!scn.LSDecisionAvailable(AMOUNT), "the amount is asked once per turn");
        System.out.println("zero-then-pass: ok (LS activated " + max + ")");
    }

    private static void activateSome() {
        VirtualTableScenario scn = scenario();
        scn.StartGame();
        check(scn.DSDecisionAvailable(AMOUNT), "amount question first");
        int force = scn.GetDSForcePileCount();
        int max = scn.DSGetChoiceMax();
        int amount = Math.max(1, max / 2);
        scn.DSDecided(amount);
        passOptionalResponses(scn);
        check(scn.GetDSForcePileCount() == force + amount, "DS activated " + amount);
        if (scn.AwaitingLSActivatePhaseActions()) scn.LSPass();
        check(scn.AwaitingDSActivatePhaseActions() && !scn.DSActionAvailable("Activate Force"), "menu without Activate Force");
        scn.DSPass();
        check(!scn.DSDecisionAvailable(CHECK), "no check prompt");
        System.out.println("activate-some: ok (" + amount + " of max " + max + ")");
    }

    public static void main(String[] args) {
        zeroThenPass();
        activateSome();
        System.out.println("ActivationRuleCheck passed");
    }
}
