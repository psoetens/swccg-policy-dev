package com.gempukku.swccgo.game.knowledge;

import java.util.*;

/** Identity-free, count-based knowledge. This class never receives simulator cards. */
public final class PileKnowledge {
    public static final int RESERVE=0, FORCE=1, USED=2, LOST=3, HAND=4,
        OUT=5, TABLE=6, STACK=7, SUPPLY=8, VOID=9, SABACC=10, TRANSIT=11, INSERT=12, N=13;
    public static final int ALL=(1 << N)-1;
    public static final class Range {
        public int lo, hi;
        public Range(int lo, int hi) { this.lo=lo; this.hi=hi; }
    }
    public static final class Bag {
        public int size, unknown;
        public final Map<String,Range> cards=new TreeMap<>();
        Bag copy() {
            Bag b=new Bag(); b.size=size; b.unknown=unknown;
            cards.forEach((k,v)->b.cards.put(k,new Range(v.lo,v.hi))); return b;
        }
        public Range range(String bp) {
            Range r=cards.get(bp);
            if(r!=null) return new Range(r.lo,Math.min(size,r.hi));
            int accounted=0; for(Range x:cards.values()) accounted+=x.lo;
            return new Range(0,Math.min(Math.max(0,size-accounted),unknown));
        }
        public boolean complete() {
            int sum=0; for(Range r:cards.values()) { if(r.lo!=r.hi) return false; sum+=r.lo; }
            return sum==size;
        }
        void change(String bp, int delta) {
            if(delta<0 && size==0) throw new IllegalStateException("remove from empty knowledge pile");
            if(bp==null) {
                for(Range r:cards.values()) { if(delta<0) r.lo=Math.max(0,r.lo-1); else r.hi++; }
                if(delta>0) unknown++;
            } else {
                Range r=range(bp);
                if(delta<0 && r.hi==0) throw new IllegalStateException("known removal contradicts pile: "+bp);
                cards.put(bp,new Range(Math.max(0,r.lo+delta),Math.max(0,r.hi+delta)));
            }
            size+=delta; unknown=Math.min(size,unknown);
            for(Range r:cards.values()) { r.hi=Math.min(size,r.hi); if(r.lo>r.hi) throw new IllegalStateException("inconsistent bounds"); }
        }
    }
    // Constraints concern sets of compartments; a transfer internal to a set
    // preserves its multiset even when neither endpoint has known composition.
    public final Map<Integer,Bag> groups=new TreeMap<>();
    public final List<List<String>> order=new ArrayList<>();
    public long sequence;
    /** Known-order mismatches tolerated outside validation (diagnostics; JVM-wide). */
    public static final java.util.concurrent.atomic.AtomicLong ORDER_MISMATCHES=new java.util.concurrent.atomic.AtomicLong();
    private static final int LOGGED_MISMATCHES=20;
    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if(ORDER_MISMATCHES.get()>0) System.err.println("knowledge tolerated mismatches total="+ORDER_MISMATCHES.get());
        }));
    }
    /** Beliefs are fuzzy: a wrong remembered position is forgotten, not fatal. Validation stays strict. */
    private static void orderMismatch(List<String> pile,int from,int fromIndex,String remembered,String observed) {
        String message="known order mismatch pile="+from+" index="+fromIndex+" remembered="+remembered+" observed="+observed;
        if(Boolean.getBoolean("policy.validateKnowledge")) throw new IllegalStateException(message);
        long count=ORDER_MISMATCHES.incrementAndGet();
        if(count<=LOGGED_MISMATCHES) System.err.println("knowledge tolerated #"+count+": "+message+" (pile order forgotten)");
        Collections.fill(pile,null);
    }
    private boolean dirty=true;
    public PileKnowledge() {
        for(int p=0;p<N;p++) { groups.put(1<<p,new Bag()); order.add(new ArrayList<>()); }
    }
    public PileKnowledge copy() {
        PileKnowledge k=new PileKnowledge(); k.groups.clear();
        groups.forEach((m,b)->k.groups.put(m,b.copy()));
        for(int p=0;p<N;p++) k.order.set(p,new ArrayList<>(order.get(p)));
        k.sequence=sequence; k.dirty=dirty; return k;
    }
    public Bag pile(int p) { return groups.get(1<<p); }
    public void initialize(int p,int size,List<String> known) {
        Bag b=new Bag(); b.size=size; b.unknown=size-known.size();
        if(b.unknown<0) throw new IllegalArgumentException("too many known cards");
        for(String bp:known) b.cards.computeIfAbsent(bp,x->new Range(0,0)).lo++;
        for(Range r:b.cards.values()) r.hi=r.lo+b.unknown;
        dirty=true; groups.put(1<<p,b); order.set(p,new ArrayList<>(Collections.nCopies(size,null)));
    }
    public void rememberInventory() {
        Bag b=new Bag(); for(int p=0;p<N;p++) b=sum(b,pile(p)); groups.put(ALL,b);
        for(int mask:new int[]{3,5,6,7}) {
            Bag union=new Bag();
            for(int p=0;p<N;p++) if((mask&(1<<p))!=0) union=sum(union,pile(p));
            groups.put(mask|(1<<TRANSIT),union);
        }
        dirty=true;
    }
    private static Bag sum(Bag a,Bag b) {
        Bag c=new Bag(); c.size=a.size+b.size; c.unknown=a.unknown+b.unknown;
        Set<String> names=new TreeSet<>(a.cards.keySet()); names.addAll(b.cards.keySet());
        for(String s:names) { Range x=a.range(s),y=b.range(s); c.cards.put(s,new Range(x.lo+y.lo,x.hi+y.hi)); }
        return c;
    }
    public void inspect(int p,List<String> identities) {
        if(identities.size()!=pile(p).size) throw new IllegalArgumentException("inspection count mismatch");
        List<String> remembered=new ArrayList<>(order.get(p));
        initialize(p,identities.size(),identities); order.set(p,remembered); sequence++; dirty=true;
    }
    public void observe(int p,String bp) {
        observe(p,bp,1);
    }
    public void observe(int p,String bp,int multiplicity) {
        Bag b=pile(p); Range r=b.range(bp);
        if(r.hi<multiplicity) throw new IllegalStateException("observation contradicts knowledge: "+bp);
        b.cards.put(bp,new Range(Math.max(multiplicity,r.lo),r.hi)); sequence++; dirty=true;
    }
    /** Public ownership change transfers inventory across the two players. */
    public void changeInventory(int p,String observed,int delta) {
        if(delta!=1 && delta!=-1) throw new IllegalArgumentException("inventory delta");
        for(Map.Entry<Integer,Bag> e:groups.entrySet())
            if((e.getKey()&(1<<p))!=0) e.getValue().change(observed,delta);
        order.set(p,new ArrayList<>(Collections.nCopies(pile(p).size,null)));
        sequence++;dirty=true;
    }
    public void shuffle(int p) { Collections.fill(order.get(p),null); sequence++; }
    public void position(int p,int offset,String bp) {
        if(offset<0 || offset>=pile(p).size) throw new IllegalArgumentException("position");
        order.get(p).set(offset,bp); observe(p,bp);
    }
    public void reverse(int p) { Collections.reverse(order.get(p)); sequence++; }
    /** A semantic transfer; null means the identity was not observed. fromIndex
     * and toIndex are observable positions only, -1 means unspecified. */
    public String move(int from,int to,String observed,int fromIndex,int toIndex) {
        // Only the four life-force piles have a shared, meaningful order.
        // Other compartments may combine several physical engine lists (Void
        // and unresolved destiny), whose local indices are not interchangeable.
        if(from>LOST) fromIndex=-1;
        if(to>LOST) toIndex=-1;
        if(from==to) { shuffle(from); return observed; }
        int union=(1<<from)|(1<<to);
        groups.computeIfAbsent(union,mask -> sum(pile(from),pile(to)));
        String bp=observed;
        List<String> src=order.get(from),dst=order.get(to);
        if(fromIndex>=0 && fromIndex<src.size()) {
            String remembered=src.remove(fromIndex);
            if(bp==null) bp=remembered;
            else if(remembered!=null && !bp.equals(remembered)) orderMismatch(src,from,fromIndex,remembered,bp);
        } else { src.clear(); src.addAll(Collections.nCopies(pile(from).size-1,null)); }
        for(Map.Entry<Integer,Bag> e:groups.entrySet()) {
            boolean f=(e.getKey()&(1<<from))!=0,t=(e.getKey()&(1<<to))!=0;
            if(f!=t) e.getValue().change(bp,t ? 1 : -1);
        }
        if(toIndex>=0 && toIndex<=dst.size()) dst.add(toIndex,bp);
        else { dst.clear(); dst.addAll(Collections.nCopies(pile(to).size,null)); }
        sequence++; dirty=true; return bp;
    }
    /** Bounded propagation: sound interval arithmetic, at most three passes.
     * No possible-deck enumeration and no hidden identity/instance matching. */
    /** Plan critic-noise stage 0: the same pile in GameKnowledge.shadow. The propagation is bounded (not a fixed point),
     * so the result depends on when it runs: the twin is tightened whenever this pile is, as the engine would have. */
    PileKnowledge twin;
    public void tighten() {
        if(twin!=null) twin.tighten();
        if(!dirty) return;
        Set<String> names=new TreeSet<>(); for(Bag b:groups.values()) names.addAll(b.cards.keySet());
        // Cache lower sums only for this propagation call. Every bound write
        // updates the cache, preserving the sequential interval arithmetic.
        Map<Bag,int[]> lower=new IdentityHashMap<>();
        for(Bag b:groups.values()) {
            int sum=0;for(Range r:b.cards.values()) sum+=r.lo;
            lower.put(b,new int[]{sum});
        }
        Bag[] piles=new Bag[N];for(int p=0;p<N;p++) piles[p]=pile(p);
        for(int pass=0;pass<3;pass++) {
            boolean changed=false;
            for(Map.Entry<Integer,Bag> e:groups.entrySet()) {
                int mask=e.getKey(); if(Integer.bitCount(mask)<2) continue;
                for(String bp:names) {
                    Range group=cachedRange(e.getValue(),bp,lower); int lo=0,hi=0;
                    for(int p=0;p<N;p++) if((mask&(1<<p))!=0) { Range r=cachedRange(piles[p],bp,lower);lo+=r.lo;hi+=r.hi; }
                    int gl=Math.max(group.lo,lo),gh=Math.min(group.hi,hi);
                    if(gl>gh) throw new IllegalStateException("group contradiction: "+bp);
                    if(gl!=group.lo || gh!=group.hi) { setRange(e.getValue(),bp,gl,gh,lower);changed=true; }
                    for(int p=0;p<N;p++) if((mask&(1<<p))!=0) {
                        Bag b=piles[p];Range r=cachedRange(b,bp,lower);
                        int l=Math.max(r.lo,gl-(hi-r.hi)),h=Math.min(r.hi,gh-(lo-r.lo));
                        if(l>h) throw new IllegalStateException("pile contradiction: "+bp);
                        if(l!=r.lo || h!=r.hi) { setRange(b,bp,l,h,lower);changed=true; }
                    }
                }
            }
            if(!changed) break;
        }
        dirty=false;
    }
    private static Range cachedRange(Bag b,String bp,Map<Bag,int[]> lower) {
        Range r=b.cards.get(bp);
        return r!=null?new Range(r.lo,Math.min(b.size,r.hi)):
            new Range(0,Math.min(Math.max(0,b.size-lower.get(b)[0]),b.unknown));
    }
    private static void setRange(Bag b,String bp,int lo,int hi,Map<Bag,int[]> lower) {
        Range old=b.cards.put(bp,new Range(lo,hi));
        lower.get(b)[0]+=lo-(old==null?0:old.lo);
    }
}
