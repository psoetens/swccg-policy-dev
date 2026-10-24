package com.gempukku.swccgo.game.knowledge;
import java.util.*;
import static com.gempukku.swccgo.game.knowledge.PileKnowledge.*;

/** CPU-only semantic regression checks, runnable without the game engine. */
public final class KnowledgeCheck {
    static void exact(PileKnowledge k,int p,String bp,int n) {
        k.tighten();
        Range r=k.pile(p).range(bp);
        if(r.lo!=n || r.hi!=n) throw new AssertionError(bp+" in "+p+": "+r.lo+".."+r.hi+" expected "+n);
    }
    public static void main(String[] args) {
        PileKnowledge k=new PileKnowledge();
        k.initialize(RESERVE,4,List.of("Luke","Han","Han","Leia"));
        k.initialize(SUPPLY,1,List.of("Shield"));
        k.rememberInventory(); k.shuffle(RESERVE);
        k.move(RESERVE,FORCE,null,0,0);k.move(RESERVE,FORCE,null,0,0);
        k.inspect(RESERVE,List.of("Han","Leia"));
        exact(k,FORCE,"Luke",1);exact(k,FORCE,"Han",1);exact(k,FORCE,"Leia",0);
        k.move(FORCE,USED,null,0,0);
        Range r=k.pile(FORCE).range("Luke");
        if(r.lo!=0 || r.hi!=1) throw new AssertionError("hidden spend leaked identity");
        k.inspect(USED,List.of("Luke"));exact(k,FORCE,"Han",1);exact(k,FORCE,"Luke",0);
        k.move(RESERVE,LOST,"Leia",-1,0);exact(k,LOST,"Leia",1);
        k.move(RESERVE,LOST,"Han",-1,0);exact(k,LOST,"Leia",1);
        if(!k.order.get(LOST).equals(List.of("Han","Leia"))) throw new AssertionError("lost order");
        k.inspect(LOST,List.of("Leia","Han"));
        if(!k.order.get(LOST).equals(List.of("Han","Leia"))) throw new AssertionError("inspection erased known order");
        PileKnowledge copy=k.copy(); copy.shuffle(LOST);exact(copy,LOST,"Leia",1);
        if(k.order.get(LOST).get(0)==null || copy.order.get(LOST).get(0)!=null) throw new AssertionError("snapshot/shuffle");
        k.move(LOST,USED,"Han",0,0);exact(k,LOST,"Han",0);
        exact(k,FORCE,"Shield",0); // supply isn't a missing main-deck card
        PileKnowledge opponent=new PileKnowledge(); opponent.initialize(RESERVE,2,List.of());
        opponent.initialize(HAND,2,List.of());opponent.rememberInventory();
        opponent.inspect(RESERVE,List.of("Vader","Vader"));
        exact(opponent,RESERVE,"Vader",2);
        if(opponent.pile(HAND).range("Vader").lo!=0) throw new AssertionError("inspection leaked opponent hand");
        PileKnowledge partial=new PileKnowledge();partial.initialize(HAND,4,List.of());partial.rememberInventory();
        partial.observe(HAND,"Luke",2);
        if(partial.pile(HAND).range("Luke").lo!=2) throw new AssertionError("duplicate partial reveal");
        PileKnowledge transfer=new PileKnowledge();transfer.initialize(RESERVE,2,List.of("Luke","Han"));transfer.rememberInventory();
        transfer.position(RESERVE,0,"Luke");
        String seen=transfer.move(RESERVE,TRANSIT,null,0,-1);
        transfer.move(TRANSIT,FORCE,seen,-1,0);exact(transfer,FORCE,"Luke",1);
        PileKnowledge oldOwner=new PileKnowledge(),newOwner=new PileKnowledge();
        oldOwner.initialize(TABLE,1,List.of("Crossed"));oldOwner.rememberInventory();newOwner.rememberInventory();
        oldOwner.changeInventory(TABLE,"Crossed",-1);newOwner.changeInventory(TABLE,"Crossed",1);
        newOwner.move(TABLE,LOST,"Crossed",-1,0);
        exact(oldOwner,TABLE,"Crossed",0);exact(newOwner,LOST,"Crossed",1);
        PileKnowledge temporary=new PileKnowledge();
        temporary.initialize(RESERVE,2,List.of("Drawn","Playing"));temporary.rememberInventory();
        temporary.move(RESERVE,VOID,"Drawn",0,0);
        temporary.move(RESERVE,VOID,"Playing",0,0); // another physical list's position zero
        temporary.move(VOID,HAND,"Drawn",0,-1);
        exact(temporary,HAND,"Drawn",1);exact(temporary,VOID,"Playing",1);
        // Every compartment must preserve multiplicity through an atomic transit
        // and a snapshot. This includes uncommon stack/insert/out-of-play zones.
        for(int source=0;source<N;source++) for(int target=0;target<N;target++) {
            if(source==target || source==TRANSIT || target==TRANSIT)continue;
            PileKnowledge allZones=new PileKnowledge();
            allZones.initialize(source,2,List.of("Twin","Twin"));allZones.rememberInventory();
            allZones.move(source,TRANSIT,"Twin",0,-1);
            PileKnowledge checkpoint=allZones.copy();
            allZones.move(TRANSIT,target,"Twin",-1,0);
            exact(allZones,source,"Twin",1);exact(allZones,target,"Twin",1);exact(allZones,TRANSIT,"Twin",0);
            exact(checkpoint,TRANSIT,"Twin",1);exact(checkpoint,target,"Twin",0);
        }
        PileKnowledge history=new PileKnowledge();
        history.initialize(RESERVE,2,List.of());history.rememberInventory();
        history.inspect(RESERVE,List.of("Luke","Han"));history.position(RESERVE,0,"Luke");
        exact(history,RESERVE,"Vader",0); // verified failed search in a complete inspection
        history.shuffle(RESERVE);
        exact(history,RESERVE,"Luke",1);
        if(history.order.get(RESERVE).get(0)!=null)throw new AssertionError("shuffle retained a known top card");
        history.move(RESERVE,USED,"Luke",-1,0);history.move(USED,RESERVE,"Luke",0,-1);
        exact(history,RESERVE,"Luke",1);exact(history,USED,"Luke",0);
        // A wrong remembered position is tolerated (order forgotten, counted) outside validation.
        PileKnowledge fuzzy=new PileKnowledge();fuzzy.initialize(LOST,2,List.of("Han","Leia"));fuzzy.rememberInventory();
        fuzzy.position(LOST,0,"Han");fuzzy.position(LOST,1,"Leia");
        String validate=System.getProperty("policy.validateKnowledge");
        System.clearProperty("policy.validateKnowledge");
        long before=ORDER_MISMATCHES.get();
        String moved=fuzzy.move(LOST,USED,"Leia",0,0);
        if(!moved.equals("Leia") || ORDER_MISMATCHES.get()!=before+1) throw new AssertionError("tolerated mismatch");
        if(fuzzy.order.get(LOST).get(0)!=null) throw new AssertionError("mismatched pile order not forgotten");
        PileKnowledge strict=new PileKnowledge();strict.initialize(LOST,1,List.of("Han"));strict.rememberInventory();
        strict.position(LOST,0,"Han");
        System.setProperty("policy.validateKnowledge","true");
        try { strict.move(LOST,USED,"Leia",0,0); throw new AssertionError("validation must stay strict"); }
        catch(IllegalStateException expected) { }
        if(validate==null) System.clearProperty("policy.validateKnowledge"); else System.setProperty("policy.validateKnowledge",validate);
        System.out.println("KnowledgeCheck: deduction, duplicates, hidden transfer, lost memory, order, snapshot, supply, opponent inspection, tolerated mismatch passed");
    }
}
