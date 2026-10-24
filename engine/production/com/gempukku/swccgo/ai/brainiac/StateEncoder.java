package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying;
import com.gempukku.swccgo.game.knowledge.TitleMatcher;

import java.lang.reflect.Field;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Encodes a decision point as one compact JSON line for the policy (record layout of swccg-policy's schema 4, built by
 * {@link KnowledgeStateEncoder}): the deciding player's view of the game, the legal options, and the human-readable
 * labels Brainiac's GUI shows. This class holds the option encoding and the line format; plan brainiac-v6 removed the
 * schema-1 encoder.
 */
public final class StateEncoder {
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
        /** Human-readable option texts and the player's hand ([blueprint, title]), for display only (not model input). */
        public final List<String> labels = new ArrayList<>();
        public final List<String[]> hand = new ArrayList<>();
        /**
         * Step-by-step card selections (ARBITRARY_CARDS with a "preselected" parameter): per option whether its card is already
         * selected, and the answers of all preselected cards (selectable or not). Not model input.
         */
        public final List<Boolean> preselected = new ArrayList<>();
        public final List<String> preselectedAnswers = new ArrayList<>();
        /** Plan brainiac-v6: action ids left out of the options (ActivationAdapter). Not model input. */
        public final Set<String> hiddenActions = new HashSet<>();
        public int min = 1, max = 1;
        public String type;
        public int version=1;
        public com.gempukku.swccgo.game.knowledge.ObservationV2 observation;

        public String line(String gameId, int decisionIndex, List<Integer> chosen) {
            StringBuilder sb = new StringBuilder(prefix.length() + 64);
            sb.append('[').append(version).append(',').append(q(gameId)).append(',').append(decisionIndex).append(',');
            sb.append(prefix).append(",[");
            for (int i = 0; i < chosen.size(); i++) sb.append(i == 0 ? "" : ",").append(chosen.get(i));
            sb.append("]");
            if (version == 4) {
                sb.append(",").append(com.gempukku.swccgo.game.knowledge.KnowledgeGrid.json(observation.kgrid));
            }
            return sb.append("]").toString();
        }
    }

    private StateEncoder() {
    }

    /** Text template of a prompt/action: card hints become "<c>", other HTML removed, numbers kept. */
    public static String template(String text, List<String> blueprints) {
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

    public static boolean visibleTo(PhysicalCard c, String player) {
        Zone z = c.getZone();
        if (z == null || c.isNotShownOnUserInterface()) return false;
        return z.isPublic() || (z.isVisibleByOwner() && player.equals(c.getOwner()));
    }

    /**
     * Plan brainiac-v6: the one production encoding, the schema-4 observation with context version 5
     * ({@link KnowledgeStateEncoder}); hidden action ids (see {@link ActivationAdapter}) are left out of the options.
     */
    public static Encoded encode(SwccgGame game, String player, AwaitingDecision decision, Set<String> hiddenActions) {
        return KnowledgeStateEncoder.encode(game, player, decision, hiddenActions);
    }

    /** The training harness's parity checks pass a schema version; only schema 4 remains. */
    public static Encoded encode(SwccgGame game, String player, AwaitingDecision decision, int version) {
        if (version != 4) {
            throw new IllegalArgumentException("Brainiac encodes schema 4 only, not " + version);
        }
        return encode(game, player, decision, Set.of());
    }

    static void encodeOptions(Encoded encoded, AwaitingDecision decision,
            Map<Integer, Integer> tokenOfCardId, GameState state) {
        Map<String, String[]> parameters = decision.getDecisionParameters();
        StringBuilder output = encoded.prefix;
        List<String[]> options = new ArrayList<>(); // template, bps-json, sourceToken, blueprint, value
        switch (encoded.type) {
            case "CARD_ACTION_CHOICE":
            case "ACTION_CHOICE": {
                String[] ids = parameters.getOrDefault("actionId", new String[0]);
                String[] texts = parameters.getOrDefault("actionText", new String[ids.length]);
                String[] cards = parameters.get("cardId");
                String[] blueprints = parameters.get("blueprintId");
                for (int index = 0; index < ids.length; index++) {
                    if (encoded.hiddenActions.contains(ids[index])) {
                        continue;
                    }
                    List<String> references = encoded.version == 4 ? new TitleMatcher.References() : new ArrayList<>();
                    String template = templateFor(encoded, texts[index], references);
                    int sourceToken = -1;
                    if (cards != null && index < cards.length) {
                        try {
                            sourceToken = tokenOfCardId.getOrDefault(Integer.parseInt(cards[index]), -1);
                        } catch (NumberFormatException ignored) {
                        }
                    }
                    String blueprint = blueprints != null && index < blueprints.length
                        ? stripBp(blueprints[index]) : null;
                    addOption(encoded, options, template, references, sourceToken, blueprint, ids[index],
                        null, false, plain(texts[index]));
                }
                boolean noPass = "true".equals(first(parameters, "noPass"));
                if (encoded.type.equals("CARD_ACTION_CHOICE") && !noPass) {
                    addOption(encoded, options, "<pass>", List.of(), -1, null, "", null, true, "Pass");
                }
                break;
            }
            case "CARD_SELECTION": {
                String[] ids = parameters.getOrDefault("cardId", new String[0]);
                encoded.min = parseInt(first(parameters, "min"), 1);
                encoded.max = parseInt(first(parameters, "max"), 1);
                for (String id : ids) {
                    int sourceToken = -1;
                    try {
                        sourceToken = tokenOfCardId.getOrDefault(Integer.parseInt(id), -1);
                    } catch (NumberFormatException ignored) {
                    }
                    addOption(encoded, options, "<card>", List.of(), sourceToken, null, id,
                        null, false, title(state, id));
                }
                if (encoded.min == 0) {
                    addOption(encoded, options, "<none>", List.of(), -1, null, "", null, true, "None");
                }
                break;
            }
            case "ARBITRARY_CARDS": {
                String[] ids = parameters.getOrDefault("cardId", new String[0]);
                String[] blueprints = parameters.getOrDefault("blueprintId", new String[ids.length]);
                String[] selectable = parameters.get("selectable");
                String[] preselected = parameters.get("preselected");
                encoded.min = parseInt(first(parameters, "min"), 1);
                encoded.max = parseInt(first(parameters, "max"), 1);
                List<Integer> realCardIds = arbitraryCardIds(decision);
                for (int index = 0; index < ids.length; index++) {
                    boolean isPre = preselected != null && index < preselected.length
                        && "true".equals(preselected[index]);
                    if (isPre) {
                        encoded.preselectedAnswers.add(ids[index]);
                    }
                    if (selectable != null && index < selectable.length && !"true".equals(selectable[index])) {
                        continue;
                    }
                    int sourceToken = realCardIds != null && index < realCardIds.size()
                        ? tokenOfCardId.getOrDefault(realCardIds.get(index), -1) : -1;
                    String label = realCardIds != null && index < realCardIds.size()
                        ? title(state, String.valueOf(realCardIds.get(index))) : blueprints[index];
                    addOption(encoded, options, "<card>", List.of(), sourceToken, stripBp(blueprints[index]),
                        ids[index], null, false, label);
                    if (isPre) {
                        encoded.preselected.set(encoded.preselected.size() - 1, true);
                    }
                }
                if (encoded.min == 0) {
                    addOption(encoded, options, "<none>", List.of(), -1, null, "", null, true, "None");
                }
                break;
            }
            case "MULTIPLE_CHOICE": {
                String[] results = parameters.getOrDefault("results", new String[0]);
                for (int index = 0; index < results.length; index++) {
                    List<String> references = encoded.version == 4 ? new TitleMatcher.References() : new ArrayList<>();
                    addOption(encoded, options, "<mc>" + templateFor(encoded, results[index], references),
                        references, -1, null, String.valueOf(index), null, false, plain(results[index]));
                }
                break;
            }
            case "INTEGER": {
                int min = parseInt(first(parameters, "min"), 0);
                int max = parseInt(first(parameters, "max"), min);
                if (encoded.version >= 2 && (long) max - min > 4096) {
                    throw new IllegalArgumentException("INTEGER action range exceeds supported enumeration");
                }
                int upper = encoded.version >= 2 ? max : Math.min(max, min + 30);
                for (int value = min; value <= upper; value++) {
                    addOption(encoded, options, "<int>", List.of(), -1, null,
                        String.valueOf(value), value, false, String.valueOf(value));
                }
                break;
            }
            default:
                break;
        }
        output.append('[');
        for (int index = 0; index < options.size(); index++) {
            String[] option = options.get(index);
            output.append(index == 0 ? "" : ",").append('[').append(q(option[0])).append(',')
                .append(option[1]).append(',').append(option[2]).append(',')
                .append(option[3] == null ? "null" : q(option[3])).append(',')
                .append(q(encoded.answers.get(index))).append(',')
                .append(option[4] == null ? "null" : option[4]).append(']');
        }
        output.append(']');
    }

    private static void addOption(Encoded encoded, List<String[]> options, String template,
            List<String> textBlueprints, int sourceToken, String blueprint, String answer,
            Integer value, boolean none, String label) {
        StringBuilder referenceJson = new StringBuilder();
        appendStrings(referenceJson, textBlueprints);
        options.add(new String[]{template, referenceJson.toString(), String.valueOf(sourceToken), blueprint,
            value == null ? null : String.valueOf(value)});
        if (encoded.observation != null) {
            for (int index = 0; index < textBlueprints.size(); index++) {
                if (encoded.version == 4) {
                    encoded.observation.reference(textBlueprints, encoded.answers.size() + 1, index);
                } else {
                    encoded.observation.reference(textBlueprints.get(index), encoded.answers.size() + 1, index);
                }
            }
        }
        encoded.answers.add(answer == null ? "" : answer);
        encoded.isNone.add(none);
        encoded.labels.add(label == null ? "" : label);
        encoded.preselected.add(false);
    }

    /** Versioned title normalization with ordered blueprint references. */
    private static String templateFor(Encoded encoded, String text, List<String> references) {
        if (encoded.version == 4) {
            return TitleMatcher.shared().template(text, (TitleMatcher.References) references,
                TitleMatcher.permitted(encoded.observation));
        }
        return template(text, references);
    }

    /** Text without HTML (a card hint keeps its title). */
    private static String plain(String html) {
        return html == null ? "" : TAG.matcher(html).replaceAll("").replaceAll("\\s+", " ").trim();
    }

    private static String title(GameState gs, String cardId) {
        try {
            PhysicalCard c = gs.findCardById(Integer.parseInt(cardId));
            return c == null ? cardId : c.getTitle();
        } catch (RuntimeException ex) {
            return cardId;
        }
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

    public static String stripBp(String bp) {
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

    static void appendStrings(StringBuilder sb, List<String> xs) {
        sb.append('[');
        for (int i = 0; i < xs.size(); i++) sb.append(i == 0 ? "" : ",").append(q(xs.get(i)));
        sb.append(']');
    }

    public static String q(String s) {
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
