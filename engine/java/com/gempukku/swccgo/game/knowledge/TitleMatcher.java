package com.gempukku.swccgo.game.knowledge;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Static card-table titles plus an explicitly supplied permitted blueprint set. */
public final class TitleMatcher {
    public static final String NORMALIZER_VERSION = "knowledge-v3.1-amendment3-v1";
    /**
     * Plan titles-and-learner-ablation: v2 = v1 plus the engine's marked card names (NameMarks): a marked name becomes
     * {@code <c>} with its exact blueprint. Marks reach the matcher only inside NameMarks.raw, for a v2 bundle.
     */
    public static final String NORMALIZER_V2 = "knowledge-v3.1-amendment3-v2";

    public static java.util.Set<String> normalizers() {
        return java.util.Set.of(NORMALIZER_VERSION, NORMALIZER_V2);
    }
    public static final int CANONICAL = 1;
    public static final int NUMERIC = 128;
    private static final Pattern HINT = Pattern.compile(
        "<div\\s+class=['\"]cardHint['\"]\\s+value=['\"]([^'\"]*)['\"][^>]*>.*?</div>", Pattern.DOTALL
    );
    private static final Pattern TAG = Pattern.compile("<(?!>|/?(?:mc|c|n)>)[^>]+>");
    private static final Pattern NUMBER = Pattern.compile(
        "(?<![\\w-])\\d+(?![\\w-])", Pattern.UNICODE_CHARACTER_CLASS
    );
    private static volatile TitleMatcher shared;

    public static final class References extends ArrayList<String> {
        private final List<Integer> flags = new ArrayList<>();
        private final List<Float> values = new ArrayList<>();

        void reference(String blueprint, int flag) {
            add(ObservationV2.strip(blueprint));
            flags.add(flag);
            values.add(0f);
        }

        void number(String literal) {
            java.math.BigInteger integer = new java.math.BigInteger(literal);
            float value = integer.floatValue();
            if (!Float.isFinite(value)
                    || !new java.math.BigDecimal((double) value).toBigIntegerExact().equals(integer)) {
                throw new IllegalArgumentException("Numeric reference is not exact in float32: " + literal);
            }
            add("<pad>");
            flags.add(NUMERIC);
            values.add(value);
        }
    }

    public record Resolved(String blueprint, int flags) {}
    public record Counter(String key, Resolved reference) {}

    private final Map<String, List<String>> blueprints;
    private final Map<Character, List<String>> candidates = new HashMap<>();

    public TitleMatcher(Map<String, ? extends Collection<String>> table) {
        Map<String, List<String>> names = new HashMap<>();
        table.forEach((title, ids) -> {
            String normalized = stripPrefix(title);
            if (!normalized.isBlank() && !normalized.startsWith("<")) {
                TreeSet<String> merged = new TreeSet<>(names.getOrDefault(normalized, List.of()));
                merged.addAll(ids);
                names.put(normalized, List.copyOf(merged));
            }
        });
        blueprints = Map.copyOf(names);
        for (String title : blueprints.keySet()) {
            candidates.computeIfAbsent(title.charAt(0), ignored -> new ArrayList<>()).add(title);
        }
        candidates.values().forEach(titles -> titles.sort(
            Comparator.comparingInt(String::length).reversed().thenComparing(Comparator.naturalOrder())
        ));
    }

    public static TitleMatcher shared() {
        TitleMatcher result = shared;
        if (result == null) {
            synchronized (TitleMatcher.class) {
                result = shared;
                if (result == null) {
                    String path = System.getProperty("policy.titleTable");
                    if (path == null) {
                        throw new IllegalStateException("schema 4 requires a frozen policy.titleTable");
                    }
                    Map<String, List<String>> table = new HashMap<>();
                    try {
                        for (String line : Files.readAllLines(Path.of(path))) {
                            String[] fields = line.split("\t", -1);
                            if (fields.length != 2 || fields[0].isBlank() || fields[1].isBlank()) {
                                throw new IllegalArgumentException("Invalid title-table row");
                            }
                            table.computeIfAbsent(fields[1], ignored -> new ArrayList<>()).add(fields[0]);
                        }
                    } catch (IOException exception) {
                        throw new java.io.UncheckedIOException(exception);
                    }
                    result = new TitleMatcher(table);
                    shared = result;
                }
            }
        }
        return result;
    }

    public static int flag(List<String> references, int ordinal) {
        return references instanceof References typed ? typed.flags.get(ordinal) : 0;
    }

    public static float value(List<String> references, int ordinal) {
        return references instanceof References typed ? typed.values.get(ordinal) : 0;
    }

    public static Set<String> permitted(ObservationV2 observation) {
        if (observation.schema == 4 && !observation.kgrid.isEmpty()) {
            return observation.titleBlueprints();
        }
        Set<String> result = new HashSet<>();
        for (ObservationV2.Token token : observation.tokens) {
            if (token.kind <= 1 || token.kind >= 2 && token.kind <= 5 && (token.flags & 2) != 0) {
                if (!token.bp.equals("<pad>") && !token.bp.startsWith("-1_")) {
                    result.add(token.bp);
                }
            }
        }
        return Set.copyOf(result);
    }

    private static String stripPrefix(String title) {
        return title.replace("&lt;&gt;", "<>").replaceFirst("^(?:•+|<>|◇)+", "");
    }

    public Resolved resolve(String title, Set<String> permitted) {
        List<String> choices = blueprints.get(stripPrefix(title));
        if (choices == null) {
            return null;
        }
        String selected = null;
        int known = 0;
        for (String blueprint : choices) {
            if (permitted.contains(blueprint)) {
                selected = blueprint;
                known++;
            }
        }
        return known == 1 ? new Resolved(selected, 0) : new Resolved(choices.get(0), CANONICAL);
    }

    public Counter counter(String key, Set<String> permitted) {
        int family = key.indexOf("LimitCounters");
        if (family < 0) {
            return null;
        }
        int end = key.indexOf('|', family);
        if (end < 0) {
            end = key.length();
        }
        for (int start = key.indexOf('.', family); start >= 0 && start < end;
                start = key.indexOf('.', start + 1)) {
            Resolved reference = resolve(key.substring(start + 1, end), permitted);
            if (reference != null) {
                return new Counter(key.substring(0, start) + key.substring(end), reference);
            }
        }
        return null;
    }

    public String template(String text, References references, Set<String> permitted) {
        return template(text, references, permitted, true);
    }

    public String contextTemplate(String text, References references, Set<String> permitted) {
        return template(text, references, permitted, false);
    }

    /** Offline key seeding only: references were already extracted by an older encoder. */
    public String seedKey(String family, String text) {
        if (family.equals("o") && Set.of("<pass>", "<card>", "<none>", "<int>").contains(text)) {
            return text;
        }
        String key = text.replaceAll("(<c>)(?: \\(AI\\))?(?: \\(V\\))", "$1")
            .replace("<c> (AI)", "<c>");
        References references = new References();
        if (family.equals("context")) {
            if (key.contains(".type=") || key.contains(".opaque=")) {
                return key;
            }
            Counter counter = counter(key, Set.of());
            return counter == null ? contextTemplate(key, references, Set.of()) : counter.key();
        }
        if (!family.equals("q") && !family.equals("o")) {
            throw new IllegalArgumentException("Unknown vocabulary family: " + family);
        }
        return template(key, references, Set.of());
    }

    private String template(String text, References references, Set<String> permitted, boolean numbers) {
        if (text == null) {
            return "";
        }
        if (text.indexOf(NameMarks.START) >= 0) {
            return marked(text, references, permitted, numbers);
        }
        Matcher hints = HINT.matcher(text);
        StringBuilder result = new StringBuilder();
        int start = 0;
        while (hints.find()) {
            appendPlain(result, text.substring(start, hints.start()), references, permitted, numbers);
            result.append("<c>");
            references.reference(hints.group(1), 0);
            start = hints.end();
        }
        appendPlain(result, text.substring(start), references, permitted, numbers);
        return result.toString().replaceAll("\\s+", " ").trim();
    }

    /**
     * Marked names become cardHint markup, which the hint rule below turns into {@code <c>} with that exact blueprint;
     * a mark without a blueprint uses its exact title's resolution, and an unknown title stays text.
     */
    private String marked(String text, References references, Set<String> permitted, boolean numbers) {
        Matcher marks = NameMarks.MARK.matcher(text);
        StringBuilder result = new StringBuilder(text.length() + 32);
        while (marks.find()) {
            String blueprint = marks.group(1);
            if (blueprint.isEmpty()) {
                Resolved resolved = resolve(marks.group(2), permitted);
                blueprint = resolved == null ? "" : resolved.blueprint();
            }
            String replacement = blueprint.isEmpty() ? marks.group(2)
                : "<div class='cardHint' value='" + blueprint + "'>" + marks.group(2) + "</div>";
            marks.appendReplacement(result, Matcher.quoteReplacement(replacement));
        }
        marks.appendTail(result);
        // A malformed mark (never produced by NameMarks) loses its marker characters, so this cannot recurse again.
        return template(result.toString().replaceAll("[\uE000\uE001\uE002]", ""), references, permitted, numbers);
    }

    private static boolean word(char character) {
        return Character.isLetterOrDigit(character) || character == '_';
    }

    private static boolean quote(char character) {
        return character == '\'' || character == '"' || character == '“' || character == '”';
    }

    private void appendPlain(StringBuilder output, String html, References references,
            Set<String> permitted, boolean numbers) {
        String text = TAG.matcher(html).replaceAll("").replace("&lt;&gt;", "<>")
            .replace("&amp;", "&").replace("&apos;", "'").replace("&#39;", "'")
            .replace("&quot;", "\"").replace("&nbsp;", " ");
        boolean unknownMarkedTitle = false;
        for (int start = 0; start < text.length();) {
            int titleStart = start;
            while (titleStart < text.length()) {
                char marker = text.charAt(titleStart);
                if (marker == '•' || marker == '◇') {
                    titleStart++;
                } else if (text.startsWith("<>", titleStart)) {
                    titleStart += 2;
                } else {
                    break;
                }
            }
            boolean prefixed = titleStart != start;
            if (prefixed) {
                unknownMarkedTitle = false;
            }
            String match = null;
            if (!unknownMarkedTitle && titleStart < text.length()
                    && (prefixed || start == 0 || !word(text.charAt(start - 1)))) {
                for (String title : candidates.getOrDefault(text.charAt(titleStart), List.of())) {
                    int end = titleStart + title.length();
                    if (!text.startsWith(title, titleStart) || end < text.length() && word(text.charAt(end))) {
                        continue;
                    }
                    if (prefixed && end < text.length() && text.charAt(end) == ':') {
                        continue;
                    }
                    boolean quoted = start > 0 && end < text.length()
                        && quote(text.charAt(start - 1)) && quote(text.charAt(end));
                    if (prefixed || quoted || title.trim().split("\\s+").length >= 2) {
                        match = title;
                        break;
                    }
                }
            }
            if (match == null) {
                if (prefixed && !(start > 0 && text.charAt(start - 1) == '('
                        && titleStart < text.length() && text.charAt(titleStart) == ')')) {
                    unknownMarkedTitle = true;
                }
                if (numbers && !unknownMarkedTitle && Character.isDigit(text.charAt(start))) {
                    Matcher number = NUMBER.matcher(text).region(start, text.length()).useTransparentBounds(true);
                    if (number.lookingAt()) {
                        references.number(number.group());
                        output.append("<n>");
                        start = number.end();
                        continue;
                    }
                }
                output.append(text.charAt(start++));
            } else {
                Resolved resolved = resolve(match, permitted);
                references.reference(resolved.blueprint(), resolved.flags());
                output.append("<c>");
                start = titleStart + match.length();
            }
        }
    }
}
