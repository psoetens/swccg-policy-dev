package com.gempukku.swccgo.game.knowledge;

import com.gempukku.swccgo.common.Icon;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.game.state.BattleState;
import com.gempukku.swccgo.game.state.ForceDrainState;
import java.util.*;
import static com.gempukku.swccgo.game.knowledge.PileKnowledge.*;

/** Cheap structural tokens. No ModifiersQuerying, legality or combat-stat calls. */
public final class ObservationV2 {
    // Stable explicit zone vocabulary; never serialize an enum ordinal.
    public static final List<String> ZONES=List.of("AT_LOCATION","SIDE_OF_TABLE","LOCATIONS","ATTACHED",
        "STACKED","STACKED_FACE_DOWN","OUT_OF_PLAY","TOP_OF_RESERVE_DECK","TOP_OF_FORCE_PILE","TOP_OF_USED_PILE",
        "TOP_OF_LOST_PILE","TOP_OF_UNRESOLVED_DESTINY_DRAW","REVEALED_SABACC_HAND","SIDE_OF_TABLE_NOT_IN_PLAY",
        "SIDE_OF_TABLE_FACE_DOWN_NOT_IN_PLAY","HAND","SABACC_HAND","VOID","RESERVE_DECK","FORCE_PILE",
        "USED_PILE","LOST_PILE","UNRESOLVED_DESTINY_DRAW","CONVERTED_LOCATIONS","OUTSIDE_OF_DECK");
    public static final class Token {
        public int kind,zone,mine,slot=-1,parent=-1,count=1,flags;
        public String bp="<pad>";
        public String contextKey;
        public float a,b,c,d;
        /** BL-20(b), schema-4 location tokens only: packed power, packed ability, drain us, drain them (tensor ABI 5). */
        public float[] state;
        /** Lazy source of {@link #state}: computed on first use, while the game still stands at this decision. */
        private SwccgGame stateGame;
        private PhysicalCard stateLocation;
        private String statePlayer;
        public float[] state() {
            if (state == null && stateGame != null) {
                state = locationState(stateGame, stateLocation, statePlayer);
                stateGame = null;
                stateLocation = null;
            }
            return state;
        }
        public String json() { return "["+kind+",\""+bp+"\","+zone+","+mine+","+(contextKey==null?Integer.toString(slot):TemplateCoverage.quote(contextKey))+","+parent+","+count+","+flags+","+a+","+b+","+c+","+d
            +(state()==null?"":","+state[0]+","+state[1]+","+state[2]+","+state[3])+"]"; }
    }
    public final List<Token> tokens=new ArrayList<>();
    /** Plan context-overhaul, checks only: overrides policy.context for observations built on this thread. */
    static final ThreadLocal<String> CONTEXT = new ThreadLocal<>();
    public final List<int[]> kgrid = new ArrayList<>();
    // Debug-only provenance for the A1 context-preservation audit; never serialized.
    public final List<Token> contextOmissions = new ArrayList<>();
    public record ContextDefault(Class<?> type, String field, Token owner, Token omitted) {}
    public final List<ContextDefault> contextDefaults = new ArrayList<>();
    public record ContextDuplicate(Token removed, Token twin) {}
    public final List<ContextDuplicate> contextDuplicates = new ArrayList<>();
    final IdentityHashMap<Token, Object> contextCounters = new IdentityHashMap<>();
    public record ContextMapKey(String before, String after, int[] indices, int source) {}
    public final List<ContextMapKey> contextMapKeys = new ArrayList<>();
    public final Map<Integer,Integer> byCardId=new HashMap<>();
    private final Map<Integer,Integer> byPermanent=new HashMap<>();
    public final int schema;
    private final GameState gs;
    private final GameKnowledge knowledge;
    private final int viewer;
    private final String player;
    private Set<String> titleBlueprints;

    public ObservationV2(SwccgGame game,String player,boolean privileged) {
        this(game,player,privileged,2);
    }
    public ObservationV2(SwccgGame game, String player, boolean privileged, int schema) {
        this(game, player, privileged, schema, game.getGameState().getKnowledge());
    }
    /** Plan critic-noise stage 0: the view of an explicit knowledge state (GameKnowledge.shadow, the legacy rule). */
    public ObservationV2(SwccgGame game, String player, boolean privileged, int schema, GameKnowledge knowledge) {
        if (schema != 4) {  // plan remove-old-generations: schema 2 / 3 removed
            throw new IllegalArgumentException("observation schema");
        }
        this.schema = schema;
        this.gs = game.getGameState();
        this.knowledge = knowledge;
        this.player = player;
        if (knowledge == null) {
            throw new IllegalStateException("knowledge not initialized");
        }
        knowledge.synchronizeOwners(gs);
        this.viewer = knowledge.player(player);
        List<PhysicalCard> cards = new ArrayList<>();
        for (PhysicalCard card : gs.getAllPermanentCards()) {
            Zone zone = card.getZone();
            if (zone == null) {
                continue;
            }
            boolean own = player.equals(card.getOwner());
            boolean visible = !card.isNotShownOnUserInterface()
                && (zone.isPublic() || own && zone.isVisibleByOwner());
            boolean supply = own && (zone == Zone.OUTSIDE_OF_DECK
                || card.isNotShownOnUserInterface() && card.getStackedOn() != null);
            if (privileged || visible || supply || knowledge.rememberedTableIdentity(card, viewer) != null) {
                cards.add(card);
            }
        }
        // IDs are transport lookup keys, never learned instance identities.
        cards.sort(Comparator.comparingInt(PhysicalCard::getPermanentCardId));
        for (PhysicalCard card : cards) {
            Token token = new Token();
            token.kind = card.getZone() == Zone.LOCATIONS || card.getZone() == Zone.CONVERTED_LOCATIONS ? 1 : 0;
            token.bp = privileged || player.equals(card.getOwner()) && card.isNotShownOnUserInterface()
                || card.getZone() == Zone.OUTSIDE_OF_DECK
                ? GameKnowledge.blueprint(card) : strip(card.getBlueprintId(gs, false));
            if (privileged && card.isFlipped()) {
                token.bp = strip(card.getBlueprintId(gs, false));
            }
            // A masked card the viewer has seen: a face-down table card, or (plan facedown-fixes) the viewer's own
            // card stacked face down whose identity it saw before stacking (e.g. a combat card under its Jedi).
            if (!privileged && token.bp.startsWith("-1_")) {
                String remembered = knowledge.rememberedTableIdentity(card, viewer);
                if (remembered != null) {
                    token.bp = remembered;
                }
            }
            token.zone = ZONES.indexOf(card.getZone().name());
            if (token.zone < 0) {
                throw new IllegalArgumentException("unsupported zone " + card.getZone());
            }
            token.mine = player.equals(card.getOwner()) ? 1 : 0;
            token.slot = token.kind == 1 ? card.getLocationZoneIndex() : -1;
            token.flags = (card.isDamaged() ? 1 : 0) | (card.isSuspended() ? 2 : 0)
                | (card.isCaptive() ? 4 : 0) | (card.isFrozen() ? 8 : 0) | (card.isMissing() ? 16 : 0)
                | (card.isUndercover() ? 32 : 0) | (card.isCollapsed() ? 64 : 0) | (card.isHit() ? 128 : 0)
                | (card.isCrossedOver() ? 256 : 0) | (card.isBinaryOff() ? 512 : 0)
                | (card.getZone().isFaceDown() ? 1024 : 0) | (card.isGameTextCanceled() ? 2048 : 0)
                | (card.isCrashed() ? 4096 : 0) | (card.isInverted() ? 8192 : 0)
                | (card.isBlownAway() ? 16384 : 0);
            token.c = player.equals(card.getZoneOwner()) ? 1 : -1;
            token.d = card.isInserted() ? 1 : 0;
            if (schema == 4 && token.kind == 1) {
                token.a = forceIcons(game, card);
                token.stateGame = game;
                token.stateLocation = card;
                token.statePlayer = player;
            }
            if (privileged) {
                int pile = GameKnowledge.cell(card.getZone());
                if (pile <= LOST) {
                    token.b = gs.getCardPile(card.getZoneOwner(), pileZone(pile), false).indexOf(card) + 1;
                }
            }
            byCardId.put(card.getCardId(), tokens.size());
            byPermanent.put(card.getPermanentCardId(), tokens.size());
            tokens.add(token);
        }
        for (PhysicalCard card : cards) {
            edge(card, card.getAtLocation(), 0);
            int role = card.isPilotOf() ? 2 : card.isPassengerOf() ? 3 : card.isInCargoHoldAsVehicle() ? 4
                : card.isInCargoHoldAsStarfighterOrTIE() ? 5 : card.isInCargoHoldAsCapitalStarship() ? 6 : 1;
            edge(card, card.getAttachedTo(), role);
            edge(card, card.getStackedOn(), 7);
            edge(card, card.getRelatedStarshipOrVehicle(), 8);
            if (card.getZone() == Zone.CONVERTED_LOCATIONS) {
                for (PhysicalCard other : cards) {
                    if (other.getZone() == Zone.LOCATIONS
                            && other.getLocationZoneIndex() == card.getLocationZoneIndex()) {
                        edge(card, other, 13);
                    }
                }
            }
            if (card.getZone() == Zone.LOCATIONS) {
                edge(card, gs.getLocationToLeftOf(card), 9);
                edge(card, gs.getLocationToRightOf(card), 10);
                String system = card.getPartOfSystem();
                if (system == null) {
                    system = card.getSystemOrbited();
                }
                if (system != null) {
                    for (PhysicalCard other : cards) {
                        if (other.getZone() == Zone.LOCATIONS && system.equals(other.getBlueprint().getTitle())) {
                            edge(card, other, 11);
                        }
                    }
                }
            }
        }
        publicContext();
        // Plan context-overhaul: policy.context selects the context rows of schema 4: "4" (default) the stored-field
        // reflection and usage counters, "5" the typed rows of StoredContextV5, "both" both (a match between models
        // of different context versions; each tensor builder keeps its own vocabulary's rows).
        String context = schema == 4 ? CONTEXT.get() != null ? CONTEXT.get() : System.getProperty("policy.context", "4")
            : "4";
        if (!Set.of("4", "5", "both").contains(context)) {
            throw new IllegalArgumentException("policy.context must be 4, 5 or both");
        }
        boolean contextV4 = !context.equals("5");
        if (contextV4) {
            StoredContextV4.appendPublicCounters(this, game, player, cards);
        }
        if (privileged && contextV4) {
            StoredContextV4.append(this, game, cards);
        }
        if (schema == 4 && !context.equals("4")) {
            StoredContextV5.append(this, game, player, privileged);
        }
        if (schema == 4 && System.getProperty("policy.titleTable") != null) {
            ContextTitles.normalizeAll(this);
        }
        if (schema == 4 && privileged) {
            StoredContextV4.removeDuplicateCounters(this);
        }
        for (int owner = 0; owner < 2; owner++) {
            addKnowledge(knowledge.views[viewer][owner], owner, 0);
        }
        for (int owner = 0; owner < 2; owner++) {
            addKnowledge(knowledge.disclosed[owner], owner, 1);
        }
        if (privileged) {
            for (int owner = 0; owner < 2; owner++) {
                addKnowledge(knowledge.views[1 - viewer][owner], owner, 2);
            }
        }
        compactKnowledge(tokens);
        KnowledgeGrid.append(this, knowledge, viewer, privileged);
    }
    public static int fieldId(String key) {
        int h=0x811c9dc5;
        for(byte b:key.getBytes(java.nio.charset.StandardCharsets.UTF_8)) h=(h^(b&255))*16777619;
        return (int)(Integer.toUnsignedLong(h)%8191);
    }
    /**
     * Plan objective-dojo BL-20(a): a location's printed and current Light/Dark Force icons, packed into the location
     * token's otherwise unused {@code a} field (schema 4): printedLight + 16 printedDark + 256 currentLight +
     * 4096 currentDark. Current counts include modifiers (gemp ModifiersQuerying.getIconCount). Exact in float32.
     */
    public static float forceIcons(SwccgGame game, PhysicalCard location) {
        int printedLight = location.getBlueprint().getIconCount(Icon.LIGHT_FORCE);
        int printedDark = location.getBlueprint().getIconCount(Icon.DARK_FORCE);
        int currentLight = game.getModifiersQuerying().getIconCount(game.getGameState(), location, Icon.LIGHT_FORCE);
        int currentDark = game.getModifiersQuerying().getIconCount(game.getGameState(), location, Icon.DARK_FORCE);
        for (int count : new int[]{printedLight, printedDark, currentLight, currentDark}) {
            if (count < 0 || count > 15) {
                throw new IllegalStateException("Force icon count outside 0..15 at " + location.getBlueprint().getTitle());
            }
        }
        return printedLight + 16 * printedDark + 256 * currentLight + 4096 * currentDark;
    }

    /**
     * Plan objective-dojo Amendment 2 (BL-20b): the engine's own totals at a location, for the deciding player ("us")
     * and the opponent ("them"): total power and total ability present (gemp ModifiersQuerying), and the Force drain
     * amount where that side occupies the location (0 otherwise). Power and ability are packed as
     * h(us) + 512 h(them) with h(x) = round(2 clamp(x, 0, 255.5)); exact in float32.
     */
    public static float[] locationState(SwccgGame game, PhysicalCard location, String player) {
        GameState gs = game.getGameState();
        String opponent = game.getOpponent(player);
        float[] power = new float[2], ability = new float[2], drain = new float[2];
        String[] sides = {player, opponent};
        for (int side = 0; side < 2; side++) {
            power[side] = game.getModifiersQuerying().getTotalPowerAtLocation(gs, location, sides[side], false, false);
            ability[side] = game.getModifiersQuerying().getTotalAbilityAtLocation(gs, sides[side], location);
            drain[side] = com.gempukku.swccgo.cards.GameConditions.occupies(game, sides[side], location)
                ? game.getModifiersQuerying().getForceDrainAmount(gs, location, sides[side]) : 0;
        }
        return new float[]{half(power[0]) + 512 * half(power[1]), half(ability[0]) + 512 * half(ability[1]),
            Math.max(0, drain[0]), Math.max(0, drain[1])};
    }

    private static int half(float value) {
        return Math.round(2 * Math.max(0, Math.min(255.5f, value)));
    }

    public void context(String key, String owner, PhysicalCard source, float value) {
        contextV4(key, owner, source, value, new int[0]);
    }
    /** Schema-4 context: source links and ordered list positions live on the row. */
    public Token contextV4(String key, String owner, PhysicalCard source, float value, int[] indices) {
        if (schema != 4 || indices.length > 4) {
            throw new IllegalArgumentException("schema-4 context requires at most four list indices");
        }
        Token token = new Token();
        token.kind = 7;
        token.contextKey = key;
        token.mine = player.equals(owner) ? 1 : 0;
        token.a = indices.length > 0 ? indices[0] : -1;
        token.c = indices.length > 1 ? indices[1] : -1;
        token.d = indices.length > 2 ? indices[2] : -1;
        token.count = indices.length > 3 ? indices[3] : -1;
        if (!Float.isFinite(value)) {
            token.flags = 1;
        } else if (value == Float.MAX_VALUE || value == (float) Integer.MAX_VALUE) {
            token.flags = 2;
        } else if (value == -Float.MAX_VALUE || value == (float) Integer.MIN_VALUE) {
            token.flags = 4;
        } else {
            token.b = value;
        }
        if (source != null) {
            token.parent = byPermanent.getOrDefault(source.getPermanentCardId(), -1);
        }
        tokens.add(token);
        return token;
    }
    /** Only emitted entities and this viewer's knowledge; never scans hidden instances. */
    public Set<String> titleBlueprints() {
        if (titleBlueprints == null) {
            Set<String> allowed = new TreeSet<>();
            for (Token token : tokens) {
                if (token.kind <= 1 && !token.bp.startsWith("-1_")) {
                    allowed.add(token.bp);
                }
            }
            for (PileKnowledge view : knowledge.views[viewer]) {
                for (PileKnowledge.Bag bag : view.groups.values()) {
                    allowed.addAll(bag.cards.keySet());
                }
            }
            titleBlueprints = Collections.unmodifiableSet(allowed);
        }
        return titleBlueprints;
    }
    private void publicContext() {
        BattleState b=gs.getBattleState();
        if(b!=null) {
            PhysicalCard location=b.getBattleLocation();
            context("battle.active",b.getPlayerInitiatedBattle(),location,1);
            context("battle.started",null,location,b.isBattleStarted()?1:0);
            context("battle.powerSegment",null,location,b.isReachedPowerSegment()?1:0);
            context("battle.damageSegment",null,location,b.isReachedDamageSegment()?1:0);
            context("battle.canceled",null,location,b.isCanceled()?1:0);
            for(String side:knowledge.players) {
                for(PhysicalCard card:b.getCardsParticipating(side)) context("battle.participant",side,card,1);
                context("battle.powerDestinyDrawn",side,location,b.getNumDestinyToTotalPowerDrawn(side));
                context("battle.powerDestiny",side,location,b.getTotalDestinyToPowerOnly(side));
                context("battle.battleDestinyDrawn",side,location,b.getNumBattleDestinyDrawn(side));
                context("battle.damageSatisfied",side,location,b.getBattleDamageSatisfied(side));
                context("battle.attritionSatisfied",side,location,b.getAttritionSatisfied(side));
                context("battle.forceLost",side,location,b.getForceLostToBattleDamage(side));
            }
        }
        ForceDrainState drain=gs.getForceDrainState();
        if(drain!=null) {
            context("drain.total",drain.getPlayerId(),drain.getLocation(),drain.getForceTotal());
            context("drain.remaining",drain.getPlayerId(),drain.getLocation(),drain.getForceRemaining());
            context("drain.paid",drain.getPlayerId(),drain.getLocation(),drain.getForcePaid());
        }
    }
    public static String strip(String b) { return b.replaceFirst("[\\*\\^]+$", ""); }
    public static Zone pileZone(int p) {
        return new Zone[]{Zone.RESERVE_DECK,Zone.FORCE_PILE,Zone.USED_PILE,Zone.LOST_PILE}[p];
    }
    private void edge(PhysicalCard source,PhysicalCard target,int type) {
        if(target==null) return;
        Integer s=byPermanent.get(source.getPermanentCardId()),d=byPermanent.get(target.getPermanentCardId());
        if(s==null || d==null) return;
        Token t=new Token();t.kind=8;t.parent=s;t.a=d+1;t.slot=type;tokens.add(t);
    }
    private void addKnowledge(PileKnowledge k,int owner,int scope) {
        k.tighten();
        for(int p=0;p<N;p++) {
            PileKnowledge.Bag bag=k.pile(p);
            Token summary=new Token();summary.kind=3;summary.mine=owner==viewer?1:0;summary.zone=32+p;
            summary.count=bag.size;summary.flags=(bag.complete()?1:0)|(scope<<12);tokens.add(summary);
            if(p==TABLE) continue; // identities already represented by board entities
            for(Map.Entry<String,PileKnowledge.Range> e:bag.cards.entrySet()) {
                PileKnowledge.Range r=bag.range(e.getKey());if(r.hi==0 && bag.complete()) continue;
                Token t=new Token();t.kind=scope==1?6:2;t.bp=e.getKey();t.zone=32+p;t.mine=summary.mine;
                t.count=r.lo;t.b=r.hi;t.flags=(r.lo==r.hi?1:0)|(scope<<12);tokens.add(t);
            }
            for(int pos=0;pos<k.order.get(p).size();pos++) {
                String bp=k.order.get(p).get(pos);if(bp==null) continue;
                Token t=new Token();t.kind=5;t.bp=bp;t.zone=32+p;t.mine=summary.mine;t.slot=pos;t.flags=scope<<12;tokens.add(t);
            }
        }
        for(Map.Entry<Integer,PileKnowledge.Bag> e:k.groups.entrySet()) {
            int mask=e.getKey()&~(1<<TRANSIT); // only informative relationships among the hidden life-force piles
            if((mask&~7)!=0 || Integer.bitCount(mask)<2) continue;
            for(Map.Entry<String,PileKnowledge.Range> c:e.getValue().cards.entrySet()) {
                PileKnowledge.Range r=c.getValue();
                int lower=0,upper=0;for(int p=0;p<3;p++) if((mask&(1<<p))!=0) { PileKnowledge.Range a=k.pile(p).range(c.getKey());lower+=a.lo;upper+=a.hi; }
                if(r.lo<=lower && r.hi>=upper) continue;
                Token t=new Token();t.kind=4;t.bp=c.getKey();t.zone=63;t.mine=owner==viewer?1:0;t.count=r.lo;t.b=r.hi;t.slot=mask;t.flags=(scope<<12)|(mask<<4);tokens.add(t);
            }
        }
    }
    /** Only knowledge rows coalesce: instances, edges and repeated references never do.
     * Bounds/positions/owners/pile masks all participate in the equality key.
     * Knowledge rows are appended after all instance/context edges, so no edge
     * endpoint can be invalidated by compacting this suffix. */
    public static void compactKnowledge(List<Token> tokens) {
        Map<String,Token> facts=new LinkedHashMap<>();
        List<Token> compact=new ArrayList<>();
        for(Token t:tokens) {
            if(t.kind<2 || t.kind>6) {compact.add(t);continue;}
            int scope=1<<(t.flags>>12);
            t.flags=t.flags&1;
            if(t.kind==6)t.kind=2;
            String key=t.json();
            Token previous=facts.get(key);
            if(previous==null) {facts.put(key,t);compact.add(t);t.flags|=scope<<1;}
            else previous.flags|=scope<<1;
        }
        tokens.clear();tokens.addAll(compact);
    }
    public void reference(String blueprint, int option, int ordinal) {
        Token token = new Token();
        token.kind = option > 0 ? 10 : 9;
        token.bp = strip(blueprint);
        token.a = option;
        token.slot = ordinal;
        tokens.add(token);
    }
    public void reference(String blueprint, int option, int ordinal, int flags) {
        reference(blueprint, option, ordinal);
        if (schema == 4) {
            tokens.get(tokens.size() - 1).flags = flags;
        }
    }
    public void reference(List<String> references, int option, int ordinal) {
        reference(references.get(ordinal), option, ordinal, TitleMatcher.flag(references, ordinal));
        if (schema == 4) {
            tokens.get(tokens.size() - 1).b = TitleMatcher.value(references, ordinal);
        }
    }
    public String json() {
        StringJoiner j=new StringJoiner(",","[","]");for(Token t:tokens) j.add(t.json());return j.toString();
    }
}
