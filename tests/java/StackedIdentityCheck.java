import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.framework.VirtualTableScenario;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.PhysicalCardImpl;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import com.gempukku.swccgo.game.knowledge.PileKnowledge;
import com.gempukku.swccgo.game.state.GameState;

import java.util.Collections;
import java.util.HashMap;

/**
 * Plan facedown-fixes §1.2 / §1.3: the owner's knowledge of its own cards stacked face down (combat cards under a Jedi).
 * <ul>
 *   <li>identity: a card the owner held in hand and stacked face down shows its blueprint on the owner's token; the
 *       opponent's token stays masked and the opponent's view of the stack holds no identity;</li>
 *   <li>never seen: a card stacked face down straight from the shuffled Reserve Deck stays masked for its owner too;</li>
 *   <li>stack two, use one: putting one stacked card in the Used Pile (as "Use combat card" does) subtracts exactly that
 *       card from the owner's stack row; the other stays exact [1,1];</li>
 *   <li>an engine snapshot (GameKnowledge.copy) keeps the binding.</li>
 * </ul>
 * Runs with -Dpolicy.validateKnowledge=true (the knowledge oracle check after every step).
 *
 *   SWCCG_V31_CLASSES=DIR bash scripts/efficient_java_check.sh StackedIdentityCheck
 */
public final class StackedIdentityCheck {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static String token(SwccgGame game, String viewer, PhysicalCard card) {
        ObservationV2 o = new ObservationV2(game, viewer, false, 4);
        Integer index = o.byCardId.get(card.getCardId());
        return index == null ? null : o.tokens.get(index).bp;
    }

    private static PileKnowledge.Range stack(GameKnowledge k, int viewer, int owner, String bp) {
        PileKnowledge view = k.views[viewer][owner];
        view.tighten();
        return view.pile(PileKnowledge.STACK).range(bp);
    }

    private static void stackFaceDown(VirtualTableScenario scn, PhysicalCard card, PhysicalCard on) {
        scn.gameState().removeCardsFromZone(Collections.singleton(card));
        scn.gameState().stackCard(card, on, true, false, false);
    }

    public static void main(String[] args) {
        System.setProperty("policy.knowledge", "true");
        System.setProperty("policy.validateKnowledge", "true");
        VirtualTableScenario scn = new VirtualTableScenario(new HashMap<>(), new HashMap<>());
        scn.StartGame();
        GameState gs = scn.gameState();
        GameKnowledge k = gs.getKnowledge();
        check(k != null, "knowledge enabled");
        int ls = k.player(scn.LS), ds = k.player(scn.DS);
        PhysicalCard host = scn.GetLSCard("starting-location");
        PhysicalCardImpl first = scn.GetLSDestiny(3), second = scn.GetLSDestiny(5);
        String firstBp = GameKnowledge.blueprint(first), secondBp = GameKnowledge.blueprint(second);
        check(!firstBp.equals(secondBp), "two different blueprints");

        // Stack two cards from hand face down (the owner saw them in hand).
        scn.MoveCardsToLSHand(first, second);
        stackFaceDown(scn, first, host);
        stackFaceDown(scn, second, host);
        check(first.getZone() == Zone.STACKED_FACE_DOWN, "stacked face down");
        k.validate(gs);
        check(firstBp.equals(token(scn.game(), scn.LS, first)), "owner token shows the stacked card: " + token(scn.game(), scn.LS, first));
        check(secondBp.equals(token(scn.game(), scn.LS, second)), "owner token shows the second stacked card");
        String opponentToken = token(scn.game(), scn.DS, first);
        check(opponentToken != null && opponentToken.startsWith("-1_"), "opponent token masked: " + opponentToken);
        check(k.rememberedTableIdentity(first, ds) == null, "no opponent binding");
        PileKnowledge.Range r = stack(k, ls, ls, firstBp);
        check(r.lo == 1 && r.hi == 1, "owner stack row exact before the use: " + r.lo + ".." + r.hi);
        PileKnowledge.Range o = stack(k, ds, ls, firstBp);
        check(o.lo == 0, "opponent view of the stack holds no identity: " + o.lo + ".." + o.hi);
        check(firstBp.equals(k.copy().rememberedTableIdentity(first, ls)), "snapshot keeps the binding");

        // Use one: the combat card goes from under the Jedi to the Used Pile, unrevealed (PutOneStackedCardInCardPileEffect).
        gs.removeCardsFromZone(Collections.singleton(first));
        gs.addCardToTopOfZone(first, Zone.USED_PILE, scn.LS);
        k.validate(gs);
        PileKnowledge.Range used = stack(k, ls, ls, firstBp), kept = stack(k, ls, ls, secondBp);
        check(used.lo == 0 && used.hi == 0, "used card subtracted exactly: " + used.lo + ".." + used.hi);
        check(kept.lo == 1 && kept.hi == 1, "remaining card stays exact: " + kept.lo + ".." + kept.hi);
        check(secondBp.equals(token(scn.game(), scn.LS, second)), "remaining card still shown to its owner");
        check(k.rememberedTableIdentity(first, ls) == null, "binding dropped on removal");
        PileKnowledge.Range opponentKept = stack(k, ds, ls, secondBp);
        check(opponentKept.lo == 0, "opponent still knows nothing of the stack: " + opponentKept.lo + ".." + opponentKept.hi);

        // Never seen: shuffle the Reserve Deck, stack a card face down straight from it.
        scn.ShuffleLSReserveDeck();
        PhysicalCard unseen = gs.getReserveDeck(scn.LS).get(0);
        stackFaceDown(scn, unseen, host);
        k.validate(gs);
        String unseenToken = token(scn.game(), scn.LS, unseen);
        check(unseenToken != null && unseenToken.startsWith("-1_"), "never-seen stacked card stays masked: " + unseenToken);
        check(k.rememberedTableIdentity(unseen, ls) == null, "no binding for a never-seen card");
        System.out.println("{\"passed\":true,\"owner_identity\":true,\"opponent_masked\":true,\"stack_two_use_one\":true,"
            + "\"never_seen_masked\":true,\"snapshot\":true}");
    }
}
