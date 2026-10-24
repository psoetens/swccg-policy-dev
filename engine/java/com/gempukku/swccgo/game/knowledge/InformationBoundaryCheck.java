package com.gempukku.swccgo.game.knowledge;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.*;
import com.gempukku.swccgo.game.state.GameState;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Debug-only counterfactual: swap two unseen opponent cards between hand and
 * buried Reserve, preserving all observations, public counts and remembered facts.
 * Restores the simulator before returning; never used in training or inference. */
public final class InformationBoundaryCheck {
    public static final AtomicInteger checked=new AtomicInteger();
    static {
        Runtime.getRuntime().addShutdownHook(new Thread(()->System.err.println("knowledge counterfactual swaps checked: "+checked.get())));
    }
    @SuppressWarnings("unchecked")
    public static void check(SwccgGame game,String viewer) {
        if(checked.get()>=100) return;
        GameState gs=game.getGameState();String opponent=game.getOpponent(viewer);
        try {
            Field hf=GameState.class.getDeclaredField("_hands"),rf=GameState.class.getDeclaredField("_reserveDecks");
            hf.setAccessible(true);rf.setAccessible(true);
            List<PhysicalCard> hand=((Map<String,List<PhysicalCard>>)hf.get(gs)).get(opponent);
            List<PhysicalCard> reserve=((Map<String,List<PhysicalCard>>)rf.get(gs)).get(opponent);
            if(hand.isEmpty() || reserve.size()<2) return;
            PhysicalCard a=hand.get(0),b=null;int j=-1;
            for(int i=1;i<reserve.size();i++) if(!reserve.get(i).isInserted() &&
                !GameKnowledge.blueprint(a).equals(GameKnowledge.blueprint(reserve.get(i)))) {b=reserve.get(i);j=i;break;}
            if(b==null) return;
            String actor=new ObservationV2(game,viewer,false,Integer.getInteger("policy.schema",4)).json();
            String critic=new ObservationV2(game,viewer,true,Integer.getInteger("policy.schema",4)).json();
            Zone az=a.getZone(),bz=b.getZone();
            try {
                hand.set(0,b);reserve.set(j,a);a.setZone(bz);b.setZone(az);
                if(!actor.equals(new ObservationV2(game,viewer,false,Integer.getInteger("policy.schema",4)).json())) throw new AssertionError("actor leaked unseen hand/Reserve swap");
                if(critic.equals(new ObservationV2(game,viewer,true,Integer.getInteger("policy.schema",4)).json())) throw new AssertionError("critic missed hidden pile membership");
                checked.incrementAndGet();
            } finally {hand.set(0,a);reserve.set(j,b);a.setZone(az);b.setZone(bz);}
        } catch(ReflectiveOperationException e) {throw new IllegalStateException(e);}
    }
}
