import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.rules.FreeMoveRule;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.HashMap;

/**
 * Plan mpg-retry-rule: Hoth: Main Power Generators (3_061) in LS's hand, no 4th / 5th / 6th Marker on table and no 4th
 * Marker in the Reserve Deck. In LS's deploy phase its deploy is offered; the attempt fails (the 4th Marker is not
 * found); the deploy is not offered again in that phase; in LS's next deploy phase it is offered again. The scenario
 * framework does not call the rule's hooks (TrainGen and Brainiac do), so the check calls restrict / decided itself.
 *
 *   bash scripts/v31_engine.sh MpgRetryRuleCheck   (after scripts/efficient_java_check.sh)
 */
public final class MpgRetryRuleCheck {
    private static final String AMOUNT = "Choose amount of Force to activate";

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** LS's action menu after the rule filtered it; the id of the Main Power Generators deploy, or null. */
    private static String offered(VirtualTableScenario scn) {
        AwaitingDecision d = scn.LSGetDecision();
        check(d != null, "LS has a decision");
        FreeMoveRule.restrict(scn.game(), d);
        return scn.GetCardActionId(scn.LS, scn.GetLSCard("mpg"), null);
    }

    /** One step: activate the maximum when asked for an amount, otherwise pass (whoever has a decision). */
    private static void step(VirtualTableScenario scn) {
        for (String player : new String[]{scn.DS, scn.LS}) {
            AwaitingDecision d = player.equals(scn.DS) ? scn.DSGetDecision() : scn.LSGetDecision();
            if (d == null) continue;
            FreeMoveRule.restrict(scn.game(), d);
            if (d.getText() != null && d.getText().startsWith(AMOUNT)) {
                String[] max = d.getDecisionParameters().get("max");
                scn.PlayerDecided(player, max == null ? "0" : max[0]);
            } else if (player.equals(scn.DS)) scn.DSPass();
            else scn.LSPass();
            return;
        }
    }

    private static int lsTurn(VirtualTableScenario scn) {
        return scn.game().getGameState().getPlayersLatestTurnNumber(scn.LS);
    }

    /** Step until LS has its deploy-phase menu in an LS turn numbered at least {@code turn}. */
    private static void toLsDeploy(VirtualTableScenario scn, int turn) {
        for (int i = 0; i < 80; i++) {
            if (scn.GetCurrentPhase() == Phase.DEPLOY && scn.LS.equals(scn.game().getGameState().getCurrentPlayerId())
                    && lsTurn(scn) >= turn && scn.AwaitingLSDeployPhaseActions())
                return;
            step(scn);
        }
        check(false, "LS's deploy-phase menu in LS turn " + turn + " (now " + scn.GetCurrentPhase() + ", LS turn " + lsTurn(scn) + ")");
    }

    public static void main(String[] args) {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>() {{
            put("mpg", "3_061");
        }}, new HashMap<>());
        var mpg = scn.GetLSCard("mpg");
        scn.StartGame();
        scn.MoveCardsToHand(mpg);
        toLsDeploy(scn, 1);
        int turn = lsTurn(scn);

        String id = offered(scn);
        check(id != null, "the Main Power Generators deploy is offered");
        AwaitingDecision menu = scn.LSGetDecision();
        FreeMoveRule.decided(scn.game(), menu, id);
        scn.LSDecided(id);
        for (int i = 0; i < 10 && !(scn.GetCurrentPhase() == Phase.DEPLOY && scn.AwaitingLSDeployPhaseActions()); i++) step(scn);
        check(scn.GetCurrentPhase() == Phase.DEPLOY && scn.AwaitingLSDeployPhaseActions(), "LS's deploy menu again, same phase");
        check(mpg.getZone() == com.gempukku.swccgo.common.Zone.HAND, "the attempt failed: Main Power Generators still in hand (" + mpg.getZone() + ")");
        check(offered(scn) == null, "the deploy is not offered again in the phase");

        scn.LSPass();
        toLsDeploy(scn, turn + 1);
        check(offered(scn) != null, "LS's next deploy phase offers the deploy again");
        System.out.println("MpgRetryRuleCheck: OK");
    }
}
