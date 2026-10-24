package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.state.GameState;
import java.util.*;
import static com.gempukku.swccgo.game.knowledge.PileKnowledge.*;

/** Engine adapter. The belief core receives only visible identities and counts.
 * Permanent IDs route atomic transfers; they are never a remembered hidden identity. */
public final class GameKnowledge {
    public final String[] players;
    public final PileKnowledge[][] views=new PileKnowledge[2][2];
    // Public/disclosed facts: neither side's private observations enter these.
    public final PileKnowledge[] disclosed=new PileKnowledge[2];
    private final Map<Integer,Integer> provenance=new HashMap<>(), cells=new HashMap<>();
    private final Map<Integer,String[]> pending=new HashMap<>();
    private final Map<Integer,Integer> removalPositions=new HashMap<>();
    public void expectRemoval(PhysicalCard card,int position) {
        if(shadow!=null) shadow.expectRemoval(card,position);
        removalPositions.put(card.getPermanentCardId(),position);
    }
    // Identity bindings survive only while a card remains on the public table,
    // or (plan facedown-fixes) stacked on a table card, for its owner only: a player remembers the cards it put under
    // its Jedi (seen in hand before stacking). They are discarded on every removal; the four life-force piles, hand and
    // the other hidden piles never retain ID bindings, and the opponent never gains a binding for a stacked card.
    @SuppressWarnings("unchecked")
    private final Map<Integer,String>[] tableIdentity=new Map[]{new HashMap<>(),new HashMap<>()};
    // Plan critic-noise stage 0: the legacy rule (before facedown-fixes: table only) gives the old actor input for
    // distillation. -Dpolicy.legacyOwnStacked=true sets it for the game's knowledge; -Dpolicy.dualLegacy=DIR keeps a
    // shadow with it (-Dpolicy.dualLegacyRule=current: the current rule, the shadow's self-check), which receives every
    // event of this one and gives the old view of the same decisions. Both off by default.
    private boolean legacy=Boolean.getBoolean("policy.legacyOwnStacked");
    public GameKnowledge shadow;
    private boolean isShadow;
    private void linkTwins() {
        for(int i=0;i<2;i++) { disclosed[i].twin=shadow.disclosed[i]; for(int j=0;j<2;j++) views[i][j].twin=shadow.views[i][j]; }
    }
    private boolean bindable(PhysicalCard card,int viewer) {
        int p=cell(card);
        return p==TABLE || !legacy && p==STACK && viewer==origin(card);
    }
    public String rememberedTableIdentity(PhysicalCard card,int viewer) {
        return bindable(card,viewer)?tableIdentity[viewer].get(card.getPermanentCardId()):null;
    }
    private void rememberTable(PhysicalCard card,int viewer,String bp) {
        if(bindable(card,viewer) && bp!=null) tableIdentity[viewer].put(card.getPermanentCardId(),bp);
        else tableIdentity[viewer].remove(card.getPermanentCardId());
    }
    private final Deque<String> debugEvents=new ArrayDeque<>();
    private void debug(String event) {
        if(!Boolean.getBoolean("policy.validateKnowledge")) return;
        if(debugEvents.size()==40) debugEvents.removeFirst();
        debugEvents.addLast(event);
    }
    public long sequence;

    public GameKnowledge(String dark,String light) {
        players=new String[]{dark,light};
        for(int i=0;i<2;i++) { disclosed[i]=new PileKnowledge(); for(int j=0;j<2;j++) views[i][j]=new PileKnowledge(); }
    }
    public GameKnowledge copy() {
        GameKnowledge k=new GameKnowledge(players[0],players[1]);
        for(int i=0;i<2;i++) { k.disclosed[i]=disclosed[i].copy(); for(int j=0;j<2;j++) k.views[i][j]=views[i][j].copy(); }
        for(int v=0;v<2;v++) k.tableIdentity[v].putAll(tableIdentity[v]);
        k.removalPositions.putAll(removalPositions);k.provenance.putAll(provenance);k.cells.putAll(cells);pending.forEach((id,bps)->k.pending.put(id,bps.clone()));k.sequence=sequence;
        k.legacy=legacy;k.isShadow=isShadow;k.shadow=shadow==null?null:shadow.copy();
        if(k.shadow!=null) k.linkTwins();
        return k;
    }
    public int player(String name) { return players[0].equals(name) ? 0 : 1; }
    public int origin(PhysicalCard c) { return provenance.getOrDefault(c.getPermanentCardId(),player(c.getOwner())); }
    /** The player whose pile holds the card. The four life-force piles belong to their zone owner:
     * some effects (e.g. Podrace swaps) put a card into the opponent's Lost Pile without changing
     * its owner. Every other compartment follows the card owner. */
    public int holder(PhysicalCard c) {
        String zoneOwner=c.getZoneOwner();
        return cell(c)<=LOST && zoneOwner!=null ? player(zoneOwner) : player(c.getOwner());
    }
    public static int cell(Zone z) {
        if(z==null) return TRANSIT;
        switch(z) {
            case RESERVE_DECK: case TOP_OF_RESERVE_DECK: return RESERVE;
            case FORCE_PILE: case TOP_OF_FORCE_PILE: return FORCE;
            case USED_PILE: case TOP_OF_USED_PILE: return USED;
            case LOST_PILE: case TOP_OF_LOST_PILE: return LOST;
            case HAND: return HAND;
            case OUT_OF_PLAY: return OUT;
            case STACKED: case STACKED_FACE_DOWN: return STACK;
            case OUTSIDE_OF_DECK: return SUPPLY;
            case VOID: case UNRESOLVED_DESTINY_DRAW: case TOP_OF_UNRESOLVED_DESTINY_DRAW: return VOID;
            case SABACC_HAND: case REVEALED_SABACC_HAND: return SABACC;
            default: return TABLE;
        }
    }
    public static int cell(PhysicalCard c) { return c.isInserted() && cell(c.getZone())==RESERVE ? INSERT : cell(c.getZone()); }
    public static String blueprint(PhysicalCard c) {
        return c.getBlueprintId(true).replaceFirst("[\\*\\^]+$", "");
    }
    public void initialize(GameState gs) {
        if(System.getProperty("policy.dualLegacy")!=null && !isShadow) {
            shadow=new GameKnowledge(players[0],players[1]);
            shadow.isShadow=true;
            shadow.legacy=!"current".equals(System.getProperty("policy.dualLegacyRule"));
            shadow.initialize(gs);
            linkTwins();
        }
        @SuppressWarnings("unchecked") List<String>[][] inventory=new List[2][N];
        for(int o=0;o<2;o++) for(int p=0;p<N;p++) inventory[o][p]=new ArrayList<>();
        for(PhysicalCard c:gs.getAllPermanentCards()) {
            int o=holder(c),p=cell(c);
            provenance.put(c.getPermanentCardId(),o);cells.put(c.getPermanentCardId(),p);
            inventory[o][p].add(blueprint(c));
        }
        for(int o=0;o<2;o++) {
            for(int v=0;v<2;v++) {
                for(int p=0;p<N;p++) views[v][o].initialize(p,inventory[o][p].size(),v==o?inventory[o][p]:List.of());
                views[v][o].rememberInventory();
            }
            for(int p=0;p<N;p++) disclosed[o].initialize(p,inventory[o][p].size(),List.of());
            disclosed[o].rememberInventory();
        }
    }
    private static boolean publicIdentity(PhysicalCard c,GameState gs) {
        Zone z=c.getZone(); if(z==null) return false;
        String shown=c.getBlueprintId(gs,false);
        return !c.isNotShownOnUserInterface() && !shown.startsWith("-1_") &&
            (z.isPublic() || z==Zone.LOST_PILE);
    }
    private String observed(PhysicalCard c,GameState gs,int viewer) {
        Zone z=c.getZone();
        if(publicIdentity(c,gs) || (player(c.getOwner())==viewer && z!=null &&
            (z.isVisibleByOwner() && !z.isFaceDown() || z==Zone.OUTSIDE_OF_DECK ||
             c.isNotShownOnUserInterface() && c.getStackedOn()!=null))) return blueprint(c);
        return rememberedTableIdentity(c,viewer);
    }
    public void changeOwner(GameState gs,PhysicalCard c) {
        if(shadow!=null) shadow.changeOwner(gs,c);
        changeOwnerHere(gs,c);
    }
    private void changeOwnerHere(GameState gs,PhysicalCard c) {
        int id=c.getPermanentCardId(),old=origin(c),now=holder(c);
        if(old==now) return;
        int p=cells.getOrDefault(id,cell(c));
        String[] known=pending.getOrDefault(id,new String[3]);
        for(int v=0;v<2;v++) {
            String seen=observed(c,gs,v),bp=seen!=null?seen:known[v];
            views[v][old].changeInventory(p,bp,-1);
            views[v][now].changeInventory(p,bp,1);
        }
        String bp=publicIdentity(c,gs)?blueprint(c):known[2];
        disclosed[old].changeInventory(p,bp,-1);disclosed[now].changeInventory(p,bp,1);
        provenance.put(id,now);sequence++;
    }
    public void remove(GameState gs,PhysicalCard c) {
        if(shadow!=null) shadow.remove(gs,c);
        changeOwnerHere(gs,c);
        int id=c.getPermanentCardId(),o=origin(c),from=cells.getOrDefault(id,cell(c));
        if(pending.containsKey(id)) throw new IllegalStateException("double knowledge removal");
        if(from==TRANSIT) throw new IllegalStateException("unregistered knowledge card");
        int index=c.getZone()!=null && c.getZone().name().startsWith("TOP_OF_") ? 0 : -1;
        debug("remove "+c.getOwner()+" "+blueprint(c)+" from="+from+" zone="+c.getZone()+" index="+index);
        String[] known=new String[3];
        if(removalPositions.containsKey(id)) index=removalPositions.remove(id);
        for(int v=0;v<2;v++) known[v]=views[v][o].move(from,TRANSIT,observed(c,gs,v),index,-1);
        known[2]=disclosed[o].move(from,TRANSIT,publicIdentity(c,gs)?blueprint(c):null,index,-1);
        for(int v=0;v<2;v++) tableIdentity[v].remove(id);
        cells.put(id,TRANSIT);pending.put(id,known);sequence++;
    }
    public void add(GameState gs,PhysicalCard c,int index) {
        // Several effects change ownership directly between remove and add
        // (e.g. a card from hand placed in the opponent's Used Pile).
        if(shadow!=null) shadow.add(gs,c,index);
        changeOwnerHere(gs,c);
        int id=c.getPermanentCardId(),o=origin(c),to=cell(c);
        Integer from=cells.get(id);
        if(from==null) throw new IllegalStateException("new physical card requires provenance event: "+id);
        if(from==to) return;
        debug("add "+c.getOwner()+" "+blueprint(c)+" to="+to+" index="+index);
        String[] known=pending.getOrDefault(id,new String[3]);
        for(int v=0;v<2;v++) {
            String visible=observed(c,gs,v);
            String identity=visible!=null?visible:known[v];
            views[v][o].move(from,to,identity,-1,index);
            rememberTable(c,v,identity);
        }
        disclosed[o].move(from,to,publicIdentity(c,gs)?blueprint(c):known[2],-1,index);
        cells.put(id,to);pending.remove(id);sequence++;
    }
    public void reveal(GameState gs,PhysicalCard card,boolean front) {
        if(card==null || (!front && !publicIdentity(card,gs))) return;
        if(shadow!=null) shadow.reveal(gs,card,front);
        changeOwnerHere(gs,card);
        int owner=origin(card),p=cells.getOrDefault(card.getPermanentCardId(),cell(card));
        String bp=blueprint(card);
        for(int v=0;v<2;v++) rememberTable(card,v,bp);
        if(pending.containsKey(card.getPermanentCardId())) Arrays.fill(pending.get(card.getPermanentCardId()),bp);
        for(PileKnowledge k:new PileKnowledge[]{views[0][owner],views[1][owner],disclosed[owner]}) {
            if(card.getZone()!=null && card.getZone().name().startsWith("TOP_OF_") && p<=LOST)
                k.position(p,0,bp);
            else k.observe(p,bp);
        }
        sequence++;
    }
    /** Reconcile public ownership metadata at a stable decision boundary.
     * Some on-table effects call setOwner directly without moving the card.
     * No hidden identities, pile composition, modifiers or stats are queried. */
    public void synchronizeOwners(GameState gs) {
        if(shadow!=null) shadow.synchronizeOwners(gs);
        for(PhysicalCard card:gs.getAllPermanentCards())
            if(origin(card)!=player(card.getOwner())) changeOwnerHere(gs,card);
    }
    public void shuffle(String owner,Zone zone) {
        if(shadow!=null) shadow.shuffle(owner,zone);
        debug("shuffle/reverse "+owner+" "+zone);
        int o=player(owner),p=cell(zone);
        for(int v=0;v<2;v++) views[v][o].shuffle(p);
        disclosed[o].shuffle(p);sequence++;
    }
    public void reverse(String owner,Zone zone) {
        if(shadow!=null) shadow.reverse(owner,zone);
        debug("shuffle/reverse "+owner+" "+zone);
        int o=player(owner),p=cell(zone);
        for(int v=0;v<2;v++) views[v][o].reverse(p);
        disclosed[o].reverse(p);sequence++;
    }
    /** Full shown collection, including non-selectable cards. Completeness follows
     * from shown membership counts, not text matching or the search's success. */
    public void inspect(String viewer,Collection<PhysicalCard> shown,boolean publicView) {
        if(shadow!=null) shadow.inspect(viewer,shown,publicView);
        int v=player(viewer);
        Map<Integer,List<String>> groups=new TreeMap<>();
        for(PhysicalCard c:shown) {
            int o=origin(c),p=cells.getOrDefault(c.getPermanentCardId(),cell(c));
            groups.computeIfAbsent(o*N+p,k->new ArrayList<>()).add(blueprint(c));
        }
        for(Map.Entry<Integer,List<String>> e:groups.entrySet()) {
            int o=e.getKey()/N,p=e.getKey()%N;
            observeGroup(views[v][o],p,e.getValue());
            if(publicView) {
                observeGroup(views[1-v][o],p,e.getValue());observeGroup(disclosed[o],p,e.getValue());
            }
        }
        sequence++;
    }
    /** Debug-only oracle assertion. Never contributes a feature or a belief update. */
    public void validate(GameState gs) {
        for(int owner=0;owner<2;owner++) {
            List<Map<String,Integer>> truth=new ArrayList<>();
            for(int p=0;p<N;p++) truth.add(new HashMap<>());
            for(PhysicalCard c:gs.getAllPermanentCards()) if(origin(c)==owner)
                truth.get(cell(c)).merge(blueprint(c),1,Integer::sum);
            for(PileKnowledge k:new PileKnowledge[]{views[0][owner],views[1][owner],disclosed[owner]}) {
                k.tighten();
                for(Map.Entry<Integer,PileKnowledge.Bag> e:k.groups.entrySet()) {
                    Map<String,Integer> counts=new HashMap<>();
                    for(int p=0;p<N;p++) if((e.getKey()&(1<<p))!=0)
                        truth.get(p).forEach((bp,n)->counts.merge(bp,n,Integer::sum));
                    int size=counts.values().stream().mapToInt(Integer::intValue).sum();
                    if(size!=e.getValue().size) throw new IllegalStateException("knowledge size mismatch mask="+e.getKey()+" actual="+size+" belief="+e.getValue().size);
                    Set<String> names=new HashSet<>(counts.keySet());names.addAll(e.getValue().cards.keySet());
                    for(String bp:names) {
                        PileKnowledge.Range r=e.getValue().range(bp);int n=counts.getOrDefault(bp,0);
                        if(r.lo>n || r.hi<n) throw new IllegalStateException("unsound knowledge mask="+e.getKey()+" bp="+bp+" actual="+n+" range="+r.lo+".."+r.hi);
                    }
                }
                for(int p=0;p<4;p++) {
                    List<PhysicalCard> cards=gs.getCardPile(players[owner],ObservationV2.pileZone(p));
                    if(cards.size()!=k.order.get(p).size()) throw new IllegalStateException("order size mismatch owner="+owner+" pile="+p+" actual="+cards.size()+" belief="+k.order.get(p).size()+" members="+debugMembers(gs,owner,p)+" actualCards="+cards.stream().map(c->blueprint(c)+":"+c.getPermanentCardId()+":"+c.getZone()+":owner="+c.getOwner()+":zoneOwner="+c.getZoneOwner()).toList()+" events="+debugEvents);
                    for(int i=0;i<cards.size();i++) {
                        String known=k.order.get(p).get(i);
                        if(known!=null && !known.equals(blueprint(cards.get(i)))) throw new IllegalStateException("unsound known order owner="+owner+" pile="+p+" index="+i+" remembered="+known+" actual="+blueprint(cards.get(i))+" events="+debugEvents);
                    }
                }
            }
        }
    }
    private String debugMembers(GameState gs,int owner,int p) {
        List<String> out=new ArrayList<>();
        for(PhysicalCard c:gs.getAllPermanentCards()) if(origin(c)==owner && cell(c)==p)
            out.add(blueprint(c)+":"+c.getZone()+":owner="+c.getOwner()+":zoneOwner="+c.getZoneOwner());
        return out.toString();
    }
    private static void observeGroup(PileKnowledge k,int p,List<String> shown) {
        if(shown.size()==k.pile(p).size) k.inspect(p,shown);
        else {
            Map<String,Integer> counts=new HashMap<>();
            for(String bp:shown) counts.merge(bp,1,Integer::sum);
            counts.forEach((bp,n)->k.observe(p,bp,n));
        }
    }
}
