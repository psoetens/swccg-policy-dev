import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.rules.FreeMoveRule;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.HashMap;

/**
 * Plan churn-and-drain-cancel, item 1: (a) in the deploy phase (Gemp offers weapon / device transfers there, as a
 * deploy), a weapon transferred from one holder to another is not offered back to the first holder (Anakin's Lightsaber 3_071, Young Skywalker 222_029 -> Anakin Skywalker, Padawan
 * Learner 200_002, at Dune Sea); (b) in the move phase, two starships that ship-docked are not offered to ship-dock again (Mara Jade In
 * VT-49 Decimator 208_058 and Eli Vanto In Dreadnaught 223_012 at Hoth); (c) plan dock-churn-and-loop-watch-run: during
 * a ship-dock of Thunderflare 9_174 with Chimaera 9_154 at Endor, Captain Yorr 9_103 (pilot) and Onyx 1 9_160 (cargo)
 * transferred to Chimaera are not offered a transfer back to Thunderflare in the phase. The scenario framework does not call the rule's
 * hooks (TrainGen and Brainiac do), so the check calls restrict / decided itself.
 *
 *   bash scripts/v31_engine.sh TransferDockRuleCheck   (after scripts/efficient_java_check.sh)
 */
public final class TransferDockRuleCheck {
    private static final String AMOUNT = "Choose amount of Force to activate";

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static String menuText(AwaitingDecision d) {
        if (d == null) return "-";
        String[] texts = d.getDecisionParameters().get("actionText");
        return d.getText() + " " + (texts == null ? "[]" : java.util.Arrays.toString(texts));
    }

    @SuppressWarnings("unchecked")
    private static String actionClasses(AwaitingDecision d) {
        try {
            var f = com.gempukku.swccgo.logic.decisions.CardActionSelectionDecision.class.getDeclaredField("_actions");
            f.setAccessible(true);
            StringBuilder b = new StringBuilder();
            for (var a : (java.util.List<com.gempukku.swccgo.logic.timing.Action>) f.get(d))
                b.append(a.getClass().getName()).append(':').append(a.getText()).append(':')
                    .append(a.getActionSource() == null ? "-" : a.getActionSource().getTitle()).append("; ");
            return b.toString();
        } catch (ReflectiveOperationException e) {
            return e.toString();
        }
    }

    private static AwaitingDecision decision(VirtualTableScenario scn, String player) {
        return player.equals(scn.DS) ? scn.DSGetDecision() : scn.LSGetDecision();
    }

    /** The player's action menu after the rule filtered it; the id of the card's action with this text, or null. */
    private static String offered(VirtualTableScenario scn, String player, PhysicalCard card, String text) {
        AwaitingDecision d = decision(scn, player);
        check(d != null, player + " has a decision");
        FreeMoveRule.restrict(scn.game(), d);
        return scn.GetCardActionId(player, (com.gempukku.swccgo.game.PhysicalCardImpl) card, text);
    }

    /** One step towards the target: activate the maximum when asked for an amount, otherwise pass. */
    private static void step(VirtualTableScenario scn) {
        for (String player : new String[]{scn.DS, scn.LS}) {
            AwaitingDecision d = decision(scn, player);
            if (d == null) continue;
            if (d.getText() != null && d.getText().startsWith(AMOUNT)) {
                String[] max = d.getDecisionParameters().get("max");
                scn.PlayerDecided(player, max == null ? "0" : max[0]);
            } else if (player.equals(scn.DS)) scn.DSPass();
            else scn.LSPass();
            return;
        }
    }

    private static boolean awaitingMenu(VirtualTableScenario scn, String player, Phase phase) {
        if (phase == Phase.DEPLOY) return player.equals(scn.DS) ? scn.AwaitingDSDeployPhaseActions() : scn.AwaitingLSDeployPhaseActions();
        return player.equals(scn.DS) ? scn.AwaitingDSMovePhaseActions() : scn.AwaitingLSMovePhaseActions();
    }

    /** Step until the player has its own action menu in the given phase of its own turn. */
    private static void passTo(VirtualTableScenario scn, String player, Phase phase) {
        for (int i = 0; i < 60; i++) {
            if (scn.GetCurrentPhase() == phase && player.equals(scn.game().getGameState().getCurrentPlayerId())
                    && decision(scn, player) != null && awaitingMenu(scn, player, phase))
                return;
            step(scn);
        }
        check(false, player + "'s menu in its " + phase + " phase (now " + scn.GetCurrentPhase() + ")");
    }

    /** Answer follow-up decisions (cost choices, responses) until the player's menu in the phase is back. */
    private static void backToMenu(VirtualTableScenario scn, String player, Phase phase) {
        for (int i = 0; i < 20; i++) {
            if (scn.GetCurrentPhase() == phase && awaitingMenu(scn, player, phase)) return;
            step(scn);
        }
        check(false, player + "'s " + phase + " menu again (now " + scn.GetCurrentPhase() + ")");
    }

    private static void transfer() {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>() {{
            put("site", "1_130");
            put("young", "222_029");
            put("anakin", "200_002");
            put("saber", "3_071");
        }}, new HashMap<>());
        var site = scn.GetLSCard("site");
        var young = scn.GetLSCard("young");
        var anakin = scn.GetLSCard("anakin");
        var saber = scn.GetLSCard("saber");
        scn.StartGame();
        scn.MoveLocationToTable(site);
        scn.MoveCardsToLocation(site, young, anakin);
        scn.AttachCardsTo(young, saber);
        passTo(scn, scn.LS, Phase.DEPLOY);

        String id = offered(scn, scn.LS, saber, "Transfer");
        check(id != null, "the transfer is offered (menu: " + menuText(scn.LSGetDecision()) + ", Force pile "
                + scn.game().getGameState().getForcePile(scn.LS).size() + ")");
        AwaitingDecision menu = scn.LSGetDecision();
        FreeMoveRule.decided(scn.game(), menu, id);
        scn.LSDecided(id);
        AwaitingDecision target = scn.LSGetDecision();
        if (target != null && target.getText() != null && target.getText().startsWith("Choose where to transfer")) {
            FreeMoveRule.restrict(scn.game(), target);
            scn.LSChooseCard(anakin);
        }
        backToMenu(scn, scn.LS, Phase.DEPLOY);
        check(saber.getAttachedTo() == anakin, "the saber moved to Anakin (on " + saber.getAttachedTo() + ")");
        check(offered(scn, scn.LS, saber, "Transfer") == null, "the transfer back to Young Skywalker is not offered in the phase ("
                + actionClasses(scn.LSGetDecision()) + ")");
    }

    private static void dock() {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>(), new HashMap<>() {{
            put("hoth", "3_143");
            put("mara", "208_058");
            put("eli", "223_012");
        }});
        var hoth = scn.GetDSCard("hoth");
        var mara = scn.GetDSCard("mara");
        var eli = scn.GetDSCard("eli");
        scn.StartGame();
        scn.MoveLocationToTable(hoth);
        scn.MoveCardsToLocation(hoth, mara, eli);
        passTo(scn, scn.DS, Phase.MOVE);

        String id = offered(scn, scn.DS, mara, "Ship-dock");
        check(id != null, "the ship-dock is offered");
        AwaitingDecision menu = scn.DSGetDecision();
        FreeMoveRule.decided(scn.game(), menu, id);
        scn.DSDecided(id);
        AwaitingDecision target = scn.DSGetDecision();
        if (target != null && target.getText() != null && target.getText().startsWith("Choose starship to ship-dock")) {
            FreeMoveRule.restrict(scn.game(), target);
            FreeMoveRule.decided(scn.game(), target, String.valueOf(eli.getCardId()));
            scn.DSChooseCard(eli);
        }
        backToMenu(scn, scn.DS, Phase.MOVE);
        check(offered(scn, scn.DS, mara, "Ship-dock") == null, "Mara Jade's ship-dock is not offered again in the phase");
        check(offered(scn, scn.DS, eli, "Ship-dock") == null, "Eli Vanto's ship-dock is not offered again in the phase");
    }

    private static final String DOCKED_MENU = "Perform a ship-docked action";
    private static final String TRANSFER = "Transfer to other starship";

    /** Answer the follow-ups of a docked transfer (target, capacity slot) until the ship-docked menu is back. */
    private static void toDockedMenu(VirtualTableScenario scn, com.gempukku.swccgo.game.PhysicalCardImpl target, String slot) {
        for (int i = 0; i < 10; i++) {
            AwaitingDecision d = scn.DSGetDecision();
            if (d == null) { step(scn); continue; }
            if (d.getText() != null && d.getText().startsWith(DOCKED_MENU)) return;
            FreeMoveRule.restrict(scn.game(), d);
            String type = d.getDecisionType().name();
            if (type.equals("MULTIPLE_CHOICE")) {
                String[] results = d.getDecisionParameters().get("results");
                int at = java.util.Arrays.asList(results).indexOf(slot);
                scn.PlayerDecided(scn.DS, String.valueOf(Math.max(at, 0)));
            } else if (type.equals("CARD_SELECTION") || type.equals("ARBITRARY_CARDS")) {
                FreeMoveRule.decided(scn.game(), d, String.valueOf(target.getCardId()));
                scn.DSChooseCard(target);
            } else step(scn);
        }
        check(false, "the ship-docked menu again (now " + menuText(scn.DSGetDecision()) + ")");
    }

    private static void docked() {
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>(), new HashMap<>() {{
            put("endor", "8_157");
            put("chimaera", "9_154");
            put("thunderflare", "9_174");
            put("yorr", "9_103");
            put("onyx", "9_160");
        }});
        var endor = scn.GetDSCard("endor");
        var chimaera = scn.GetDSCard("chimaera");
        var thunderflare = scn.GetDSCard("thunderflare");
        var yorr = scn.GetDSCard("yorr");
        var onyx = scn.GetDSCard("onyx");
        scn.StartGame();
        scn.MoveLocationToTable(endor);
        scn.MoveCardsToLocation(endor, chimaera, thunderflare);
        scn.BoardAsPilot(thunderflare, yorr);
        scn.BoardAsVehicle(thunderflare, onyx);
        passTo(scn, scn.DS, Phase.MOVE);

        String id = offered(scn, scn.DS, thunderflare, "Ship-dock");
        check(id != null, "Thunderflare's ship-dock is offered (" + actionClasses(scn.DSGetDecision()) + ")");
        FreeMoveRule.decided(scn.game(), scn.DSGetDecision(), id);
        scn.DSDecided(id);
        toDockedMenu(scn, chimaera, null);

        for (var card : new com.gempukku.swccgo.game.PhysicalCardImpl[]{yorr, onyx}) {
            String move = offered(scn, scn.DS, card, TRANSFER);
            check(move != null, card.getTitle() + "'s transfer to Chimaera is offered (" + actionClasses(scn.DSGetDecision()) + ")");
            FreeMoveRule.decided(scn.game(), scn.DSGetDecision(), move);
            scn.DSDecided(move);
            toDockedMenu(scn, chimaera, card == yorr ? "Pilot" : "Starship");
            check(card.getAttachedTo() == chimaera, card.getTitle() + " is aboard Chimaera (" + card.getAttachedTo() + ")");
            check(offered(scn, scn.DS, card, TRANSFER) == null, card.getTitle() + "'s transfer back to Thunderflare is not offered in the phase ("
                    + actionClasses(scn.DSGetDecision()) + ")");
        }
    }

    /** Arguments: none (all), "transfer", "dock" or "docked". */
    public static void main(String[] args) {
        String only = args.length > 0 ? args[0] : "";
        if (only.isEmpty() || only.equals("transfer")) {
            transfer();
            System.out.println("TransferDockRuleCheck: transfer OK");
        }
        if (only.isEmpty() || only.equals("dock")) {
            dock();
            System.out.println("TransferDockRuleCheck: dock OK");
        }
        if (only.isEmpty() || only.equals("docked")) {
            docked();
            System.out.println("TransferDockRuleCheck: docked OK");
        }
        System.out.println("TransferDockRuleCheck: OK");
    }
}
