package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying;

import java.lang.reflect.Field;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Encodes a decision point as one compact JSON line for the policy: the deciding player's view of the game (tokens), the
 * legal options, and (for training) the chosen option indices. Only information the deciding player can know is included:
 * a card is visible if its zone is public, or if it is the player's own card in a zone visible to its owner (the same rule
 * GameCommunicationChannel uses for what a client is sent). The player's own not-visible cards are summarised as a multiset
 * of blueprints ("remaining own deck": the player knows their decklist, not the order or location).
 *
 * Line layout (JSON array, version 1):
 * [1, gameId, decisionIndex, side, decisionType, questionTemplate, [questionBlueprints], [global...],
 *  [tokens...], [options...], [chosen option indices]]
 * token  = [kind, blueprint, zone, mine, locationSlot, attachedToToken, count, flags, power, ability, forfeit, deploy]
 *          kind: 0 card, 1 location, 2 own-unseen multiset entry
 * option = [template, [blueprints in text], sourceToken, blueprint, answer, value]
 */
public final class StateEncoder {
    public static final int VERSION = 1;
    public static final String[] ZONES;
    static {
        Zone[] z = Zone.values();
        ZONES = new String[z.length + 1];
        for (int i = 0; i < z.length; i++) ZONES[i] = z[i].name();
        ZONES[z.length] = "OWN_UNSEEN";
    }
    public static final int ZONE_OWN_UNSEEN = Zone.values().length;
    private static final Pattern HINT = Pattern.compile("<div class='cardHint' value='([^']*)'>.*?</div>");
    private static final Pattern TAG = Pattern.compile("<[^>]+>");
    private static final Pattern NUM = Pattern.compile("\\d+");

    /** An encoded decision: the JSON line (without chosen indices) and the answer string of each option. */
    public static final class Encoded {
        public final StringBuilder prefix = new StringBuilder(4096);
        public final List<String> answers = new ArrayList<>();
        public final List<Boolean> isNone = new ArrayList<>();
        /**
         * Step-by-step card selections (ARBITRARY_CARDS with a "preselected" parameter): per option whether its card is already
         * selected, and the answers of all preselected cards (selectable or not). Not model input.
         */
        public final List<Boolean> preselected = new ArrayList<>();
        public final List<String> preselectedAnswers = new ArrayList<>();
        /**
         * Plan multiselect-count stage B: per option whether its card is already selected in a card selection answered
         * step by step (PolicyAi's count choice, or Gemp's own preselection). Model input: the option row's reserved column
         * (opt[1]) and, in the record line, a seventh option field 1 (absent = 0). Set with {@link #setSelected}.
         */
        public final List<Boolean> selected = new ArrayList<>();
        /** The record JSON of each option and the length of the option array that ends the prefix. */
        final List<String> optionJson = new ArrayList<>();
        int optionsLength;
        public int min = 1, max = 1;
        public String type;

        public String line(String gameId, int decisionIndex, List<Integer> chosen) {
            StringBuilder sb = new StringBuilder(prefix.length() + 64);
            sb.append('[').append(VERSION).append(',').append(q(gameId)).append(',').append(decisionIndex).append(',');
            sb.append(prefix).append(",[");
            for (int i = 0; i < chosen.size(); i++) sb.append(i == 0 ? "" : ",").append(chosen.get(i));
            return sb.append("]]").toString();
        }
    }

    private StateEncoder() {
    }

    /** Text template of a prompt/action: card hints become "<c>", other HTML removed, numbers kept. */
    static String template(String text, List<String> blueprints) {
        if (text == null) return "";
        Matcher m = HINT.matcher(text);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            blueprints.add(m.group(1));
            m.appendReplacement(sb, "<c>");
        }
        m.appendTail(sb);
        return TAG.matcher(sb.toString()).replaceAll("").replaceAll("\\s+", " ").trim();
    }

    static boolean visibleTo(PhysicalCard c, String player) {
        Zone z = c.getZone();
        if (z == null || c.isNotShownOnUserInterface()) return false;
        return z.isPublic() || (z.isVisibleByOwner() && player.equals(c.getOwner()));
    }

    // -Dpolicy.encoderCache=false: stat lookups without the engine's query cache (the reference for the equality check)
    static final boolean QUERY_CACHE = !"false".equals(System.getProperty("policy.encoderCache"));

    /**
     * The decision's options only (answers, none flags, min/max, type; source tokens unresolved), without the game state. Same
     * answers as {@link #encode}, at a fraction of the cost: used to skip the state encoding of decisions with fewer than two
     * options, which never reach the policy.
     */
    public static Encoded optionsOnly(AwaitingDecision d) {
        Encoded e = new Encoded();
        e.type = d.getDecisionType().name();
        encodeOptions(e, d, Map.of());
        return e;
    }

    /**
     * Encodes the decision. The state is read inside the engine's query cache (read-only: the power / ability / forfeit /
     * deploy-cost lookups of all tokens share their modifier evaluations), which gives the same values as uncached lookups.
     */
    public static Encoded encode(SwccgGame game, String player, AwaitingDecision d) {
        if (!QUERY_CACHE) return encodeImpl(game, player, d);
        return game.getModifiersQuerying().withQueryCache(() -> encodeImpl(game, player, d));
    }

    static Encoded encodeImpl(SwccgGame game, String player, AwaitingDecision d) {
        GameState gs = game.getGameState();
        ModifiersQuerying mq = game.getModifiersQuerying();
        String opp = game.getOpponent(player);
        Encoded e = new Encoded();
        e.type = d.getDecisionType().name();
        StringBuilder p = e.prefix;
        p.append(q(game.getDarkPlayer().equals(player) ? "DS" : "LS")).append(',').append(q(e.type)).append(',');
        List<String> qbps = new ArrayList<>();
        String qt = template(d.getText(), qbps);
        p.append(q(qt)).append(',');
        appendStrings(p, qbps);
        p.append(',');

        // Global token (public counts and game stage).
        int[] g = new int[]{
                gs.getPlayersLatestTurnNumber(player), gs.getPlayersLatestTurnNumber(opp),
                gs.getCurrentPhase() == null ? -1 : gs.getCurrentPhase().ordinal(),
                player.equals(gs.getCurrentPlayerId()) ? 1 : 0,
                safe(() -> gs.getPlayerLifeForce(player)), safe(() -> gs.getPlayerLifeForce(opp)),
                safe(() -> gs.getReserveDeckSize(player)), safe(() -> gs.getReserveDeckSize(opp)),
                safe(() -> gs.getForcePileSize(player)), safe(() -> gs.getForcePileSize(opp)),
                gs.getUsedPile(player).size(), gs.getUsedPile(opp).size(),
                gs.getLostPile(player).size(), gs.getLostPile(opp).size(),
                gs.getHand(player).size(), gs.getHand(opp).size(),
                gs.isDuringBattle() ? 1 : 0, gs.getForceDrainState() != null ? 1 : 0
        };
        p.append('[');
        for (int i = 0; i < g.length; i++) p.append(i == 0 ? "" : ",").append(g[i]);
        p.append("],");

        // Tokens: visible cards, then the own-unseen multiset.
        List<PhysicalCard> visible = new ArrayList<>();
        Map<String, Integer> unseen = new TreeMap<>();
        for (PhysicalCard c : gs.getAllPermanentCards()) {
            boolean vis = visibleTo(c, player);
            if (vis) visible.add(c);
            // Own cards whose identity the player does not see (hidden zones, or face down like the top of the Reserve
            // Deck, which appears as a card back token) belong to the remaining-deck multiset.
            if (player.equals(c.getOwner()) && (!vis || c.getZone().isFaceDown()))
                unseen.merge(stripBp(c.getBlueprintId(true)), 1, Integer::sum);
        }
        Map<Integer, Integer> tokenOfCardId = new HashMap<>();
        for (int i = 0; i < visible.size(); i++) tokenOfCardId.put(visible.get(i).getCardId(), i);
        p.append('[');
        int ti = 0;
        for (PhysicalCard c : visible) {
            Zone z = c.getZone();
            boolean isLocation = z == Zone.LOCATIONS;
            int slot = -1;
            if (isLocation) slot = c.getLocationZoneIndex();
            else if (c.getAtLocation() != null) slot = c.getAtLocation().getLocationZoneIndex();
            PhysicalCard parent = c.getAttachedTo() != null ? c.getAttachedTo() : c.getStackedOn();
            int attach = parent == null ? -1 : tokenOfCardId.getOrDefault(parent.getCardId(), -1);
            int flags = (c.isDamaged() ? 1 : 0) | (c.isSuspended() ? 2 : 0) | (c.isCaptive() ? 4 : 0) | (c.isFrozen() ? 8 : 0)
                    | (c.isMissing() ? 16 : 0) | (c.isUndercover() ? 32 : 0) | (c.isCollapsed() ? 64 : 0) | (c.isHit() ? 128 : 0)
                    | (c.isCrossedOver() ? 256 : 0) | (c.isBinaryOff() ? 512 : 0) | (z.isFaceDown() ? 1024 : 0)
                    | (c.getStackedOn() != null ? 2048 : 0);
            float power = Float.NaN, ability = Float.NaN, forfeit = Float.NaN, deploy = Float.NaN;
            if (z.isInPlay()) {
                if (c.getBlueprint().hasPowerAttribute()) power = safeF(() -> mq.getPower(gs, c));
                if (c.getBlueprint().hasAbilityAttribute()) ability = safeF(() -> mq.getAbility(gs, c));
                if (c.getBlueprint().hasForfeitAttribute()) forfeit = safeF(() -> mq.getForfeit(gs, c));
            } else if (z == Zone.HAND) {
                // Blueprints without a deploy cost throw from getDeployCost(); safeF maps that to "no value".
                deploy = safeF(() -> c.getBlueprint().getDeployCost() == null ? Float.NaN : mq.getDeployCost(gs, c));
            }
            if (ti++ > 0) p.append(',');
            p.append('[').append(isLocation ? 1 : 0).append(',').append(q(stripBp(c.getBlueprintId(gs, false)))).append(',')
                    .append(z.ordinal()).append(',').append(player.equals(c.getOwner()) ? 1 : 0).append(',').append(slot).append(',')
                    .append(attach).append(",1,").append(flags).append(',').append(num(power)).append(',').append(num(ability))
                    .append(',').append(num(forfeit)).append(',').append(num(deploy)).append(']');
        }
        for (Map.Entry<String, Integer> u : unseen.entrySet()) {
            if (ti++ > 0) p.append(',');
            p.append("[2,").append(q(u.getKey())).append(',').append(ZONE_OWN_UNSEEN).append(",1,-1,-1,").append(u.getValue())
                    .append(",0,null,null,null,null]");
        }
        p.append("],");

        // Options.
        encodeOptions(e, d, tokenOfCardId);
        return e;
    }

    private static void encodeOptions(Encoded e, AwaitingDecision d, Map<Integer, Integer> tokenOfCardId) {
        Map<String, String[]> pr = d.getDecisionParameters();
        StringBuilder p = e.prefix;
        List<String[]> opts = new ArrayList<>(); // template, bps-json, sourceToken, blueprint, value
        switch (e.type) {
            case "CARD_ACTION_CHOICE":
            case "ACTION_CHOICE": {
                String[] ids = pr.getOrDefault("actionId", new String[0]);
                String[] texts = pr.getOrDefault("actionText", new String[ids.length]);
                String[] cards = pr.get("cardId");
                String[] bps = pr.get("blueprintId");
                for (int i = 0; i < ids.length; i++) {
                    List<String> tb = new ArrayList<>();
                    String t = template(texts[i], tb);
                    int src = -1;
                    if (cards != null && i < cards.length) {
                        try {
                            src = tokenOfCardId.getOrDefault(Integer.parseInt(cards[i]), -1);
                        } catch (NumberFormatException ignored) {
                        }
                    }
                    String bp = bps != null && i < bps.length ? stripBp(bps[i]) : null;
                    addOption(e, opts, t, tb, src, bp, ids[i], null, false);
                }
                boolean noPass = "true".equals(first(pr, "noPass"));
                if (e.type.equals("CARD_ACTION_CHOICE") && !noPass)
                    addOption(e, opts, "<pass>", List.of(), -1, null, "", null, true);
                break;
            }
            case "CARD_SELECTION": {
                String[] ids = pr.getOrDefault("cardId", new String[0]);
                e.min = parseInt(first(pr, "min"), 1);
                e.max = parseInt(first(pr, "max"), 1);
                for (String id : ids) {
                    int src = -1;
                    try {
                        src = tokenOfCardId.getOrDefault(Integer.parseInt(id), -1);
                    } catch (NumberFormatException ignored) {
                    }
                    addOption(e, opts, "<card>", List.of(), src, null, id, null, false);
                }
                if (e.min == 0) addOption(e, opts, "<none>", List.of(), -1, null, "", null, true);
                break;
            }
            case "ARBITRARY_CARDS": {
                String[] ids = pr.getOrDefault("cardId", new String[0]);
                String[] bps = pr.getOrDefault("blueprintId", new String[ids.length]);
                String[] sel = pr.get("selectable");
                String[] pre = pr.get("preselected");
                e.min = parseInt(first(pr, "min"), 1);
                e.max = parseInt(first(pr, "max"), 1);
                List<Integer> real = arbitraryCardIds(d);
                for (int i = 0; i < ids.length; i++) {
                    boolean isPre = pre != null && i < pre.length && "true".equals(pre[i]);
                    if (isPre) e.preselectedAnswers.add(ids[i]);
                    if (sel != null && i < sel.length && !"true".equals(sel[i])) continue;
                    int src = real != null && i < real.size() ? tokenOfCardId.getOrDefault(real.get(i), -1) : -1;
                    addOption(e, opts, "<card>", List.of(), src, stripBp(bps[i]), ids[i], null, false);
                    if (isPre) e.preselected.set(e.preselected.size() - 1, true);
                }
                if (e.min == 0) addOption(e, opts, "<none>", List.of(), -1, null, "", null, true);
                break;
            }
            case "MULTIPLE_CHOICE": {
                String[] res = pr.getOrDefault("results", new String[0]);
                for (int i = 0; i < res.length; i++) {
                    List<String> tb = new ArrayList<>();
                    addOption(e, opts, "<mc>" + template(res[i], tb), tb, -1, null, String.valueOf(i), null, false);
                }
                break;
            }
            case "INTEGER": {
                int min = parseInt(first(pr, "min"), 0), max = parseInt(first(pr, "max"), min);
                int hi = Math.min(max, min + 30);
                for (int v = min; v <= hi; v++)
                    addOption(e, opts, "<int>", List.of(), -1, null, String.valueOf(v), v, false);
                break;
            }
            default:
                break;
        }
        for (int i = 0; i < opts.size(); i++) e.optionJson.add(optionJson(opts.get(i), e.answers.get(i)));
        int start = p.length();
        appendOptions(e, p);
        e.optionsLength = p.length() - start;
    }

    private static String optionJson(String[] o, String answer) {
        StringBuilder p = new StringBuilder();
        p.append('[').append(q(o[0])).append(',').append(o[1]).append(',').append(o[2]).append(',')
                .append(o[3] == null ? "null" : q(o[3])).append(',').append(q(answer)).append(',')
                .append(o[4] == null ? "null" : o[4]).append(']');
        return p.toString();
    }

    private static void appendOptions(Encoded e, StringBuilder p) {
        p.append('[');
        for (int i = 0; i < e.optionJson.size(); i++) {
            String o = e.optionJson.get(i);
            if (i > 0) p.append(',');
            // Plan multiselect-count stage B: a selected option carries a seventh field 1.
            if (e.selected.get(i)) p.append(o, 0, o.length() - 1).append(",1]");
            else p.append(o);
        }
        p.append(']');
    }

    /** Rewrites the option array that ends the encoding (after an option or a selection flag changed). */
    private static void rewriteOptions(Encoded e) {
        int end = e.prefix.length(), start = end - e.optionsLength;
        if (start < 0 || e.prefix.charAt(start) != '[' || e.prefix.charAt(end - 1) != ']')
            throw new IllegalStateException("the options must end the encoding");
        e.prefix.setLength(start);
        appendOptions(e, e.prefix);
        e.optionsLength = e.prefix.length() - start;
    }

    /**
     * Plan multiselect-count stage B: marks exactly the given options as already selected (model input, see
     * {@link Encoded#selected}); the others are unmarked.
     */
    public static void setSelected(Encoded e, java.util.Collection<Integer> options) {
        boolean changed = false;
        for (int i = 0; i < e.selected.size(); i++) {
            boolean flag = options.contains(i);
            if (e.selected.get(i) != flag) {
                e.selected.set(i, flag);
                changed = true;
            }
        }
        if (changed) rewriteOptions(e);
    }

    /**
     * Plan multiselect-count: the option template of the harness-made "Done" option of a card selection answered step by
     * step (PolicyAi): stop with the cards selected so far. Not a Gemp option and not a "none" option (that stays the empty
     * answer, cancel); its own row of the option vocabulary.
     */
    public static final String DONE = "<done>Finish selection";

    /**
     * Appends the Done option as the last option of an encoded card selection (the options are the end of the encoding)
     * and returns its index. Its answer string is empty: PolicyAi translates a Done pick into the selection itself.
     */
    public static int appendDone(Encoded e) {
        List<String[]> opts = new ArrayList<>();
        addOption(e, opts, DONE, List.of(), -1, null, "", null, false);
        e.optionJson.add(optionJson(opts.get(0), ""));
        rewriteOptions(e);
        return e.answers.size() - 1;
    }

    private static void addOption(Encoded e, List<String[]> opts, String template, List<String> textBps, int src, String bp,
                                  String answer, Integer value, boolean none) {
        StringBuilder b = new StringBuilder();
        appendStrings(b, textBps);
        opts.add(new String[]{template, b.toString(), String.valueOf(src), bp, value == null ? null : String.valueOf(value)});
        e.answers.add(answer == null ? "" : answer);
        e.isNone.add(none);
        e.preselected.add(false);
        e.selected.add(false);
    }

    /** Maps an AI answer string to option indices (comma-separated multi-answers for card selections). */
    public static List<Integer> chosen(Encoded e, String answer) {
        List<Integer> out = new ArrayList<>();
        if (answer == null) return out;
        String a = answer.trim();
        if (a.isEmpty()) {
            for (int i = 0; i < e.answers.size(); i++) if (e.isNone.get(i)) out.add(i);
            return out;
        }
        if (e.type.equals("INTEGER")) {
            int best = -1;
            for (int i = 0; i < e.answers.size(); i++) if (e.answers.get(i).equals(a)) best = i;
            if (best < 0 && !e.answers.isEmpty()) best = e.answers.size() - 1; // clamped range
            if (best >= 0) out.add(best);
            return out;
        }
        for (String tok : a.split(",")) {
            for (int i = 0; i < e.answers.size(); i++) {
                if (!e.isNone.get(i) && e.answers.get(i).equals(tok.trim())) {
                    out.add(i);
                    break;
                }
            }
        }
        return out;
    }

    static List<Integer> arbitraryCardIds(AwaitingDecision d) {
        try {
            Class<?> c = d.getClass();
            while (c != null && !c.getSimpleName().equals("ArbitraryCardsSelectionDecision")) c = c.getSuperclass();
            if (c == null) return null;
            Field f = c.getDeclaredField("_physicalCards");
            f.setAccessible(true);
            List<Integer> ids = new ArrayList<>();
            for (Object o : (Collection<?>) f.get(d)) ids.add(((PhysicalCard) o).getCardId());
            return ids;
        } catch (ReflectiveOperationException ex) {
            return null;
        }
    }

    static String stripBp(String bp) {
        if (bp == null) return null;
        int end = bp.length();
        while (end > 0 && (bp.charAt(end - 1) == '*' || bp.charAt(end - 1) == '^')) end--;
        return bp.substring(0, end);
    }

    private static String first(Map<String, String[]> pr, String k) {
        String[] v = pr.get(k);
        return v == null || v.length == 0 ? null : v[0];
    }

    private static int parseInt(String s, int def) {
        try {
            return s == null ? def : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    interface IntSupplier { int get(); }
    interface FloatSupplier { float get(); }

    private static int safe(IntSupplier s) {
        try {
            return s.get();
        } catch (RuntimeException e) {
            return -1;
        }
    }

    private static float safeF(FloatSupplier s) {
        try {
            return s.get();
        } catch (RuntimeException e) {
            return Float.NaN;
        }
    }

    private static String num(float f) {
        if (Float.isNaN(f) || Float.isInfinite(f)) return "null";
        if (f == (int) f) return String.valueOf((int) f);
        return String.valueOf(f);
    }

    private static void appendStrings(StringBuilder sb, List<String> xs) {
        sb.append('[');
        for (int i = 0; i < xs.size(); i++) sb.append(i == 0 ? "" : ",").append(q(xs.get(i)));
        sb.append(']');
    }

    static String q(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
