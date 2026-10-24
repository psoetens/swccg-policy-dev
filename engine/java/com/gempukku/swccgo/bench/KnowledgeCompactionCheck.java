package com.gempukku.swccgo.bench;
import com.gempukku.swccgo.game.knowledge.ObservationV2;
import java.util.*;
public final class KnowledgeCompactionCheck {
    static ObservationV2.Token fact(int scope,int lo,int hi) {
        var t=new ObservationV2.Token();t.kind=scope==1?6:2;t.bp="1_1";t.zone=32;t.mine=1;
        t.count=lo;t.b=hi;t.flags=scope<<12;return t;
    }
    public static void main(String[] args) {
        // Private exact 1 and public 0..2 are distinct. Opponent-private 0..2
        // may coalesce with PUBLIC, but not with the more precise private fact.
        var tokens=new ArrayList<ObservationV2.Token>();
        tokens.add(fact(0,1,1));tokens.add(fact(1,0,2));tokens.add(fact(2,0,2));
        var exclusion=fact(0,0,0);exclusion.bp="1_2";tokens.add(exclusion);
        var empty=new ObservationV2.Token();empty.kind=3;empty.zone=33;empty.count=0;empty.flags=1;tokens.add(empty);
        var instance=new ObservationV2.Token();instance.bp="1_1";tokens.add(instance);tokens.add(new ObservationV2.Token());
        ObservationV2.compactKnowledge(tokens);
        if(tokens.size()!=6)throw new AssertionError("incorrect merging");
        if(tokens.get(0).flags!=2 || tokens.get(1).flags!=12)throw new AssertionError("lost scope");
        if(tokens.get(0).count!=1 || tokens.get(1).b!=2)throw new AssertionError("lost bounds");
        if(tokens.get(2).b!=0 || tokens.get(2).count!=0 || !tokens.get(2).bp.equals("1_2"))throw new AssertionError("lost exclusion");
        if(tokens.get(3).flags!=3 || tokens.get(3).count!=0)throw new AssertionError("lost empty complete pile");
        if(tokens.get(4)!=instance)throw new AssertionError("lost instance identity");
        System.out.println("scope compaction: distinct bounds, exclusion, empty pile, instances preserved");
    }
}
