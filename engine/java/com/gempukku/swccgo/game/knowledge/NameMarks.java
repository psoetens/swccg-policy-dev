package com.gempukku.swccgo.game.knowledge;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Plan titles-and-learner-ablation, part A: exact card names for the encoder, invisible to everyone else.
 *
 * <p>{@code GameUtils.getFullName} always wraps a card's plain title in private-use
 * markers carrying its blueprint id: {@code U+E000 blueprint U+E001 title U+E002} (the blueprint is empty when only a
 * blueprint object was at hand). Every reader of decision and action texts gets them stripped back to today's text
 * (clients, the serializer, the built-in AIs, duplicate-action checks, snapshots, messages). Only an encoder that runs
 * inside {@link #raw} for a normalizer-v2 table bundle sees the marks; {@link TitleMatcher} turns them into exact card
 * references. Every other reader sees exactly the text it saw before the marks existed (proved
 * byte-identical with an on/off switch that was removed after the check; reports/titles-and-learner-ablation.md A1).
 */
public final class NameMarks {
    public static final char START = '', TITLE = '', END = '';
    static final Pattern MARK = Pattern.compile("([^]*)([^]*)");
    private static final ThreadLocal<Boolean> RAW = ThreadLocal.withInitial(() -> Boolean.FALSE);

    private NameMarks() {
    }

    /** The name as shown today, marked for the encoder. */
    public static String mark(String blueprintId, String name) {
        if (name == null) {
            return name;
        }
        return START + (blueprintId == null ? "" : blueprintId) + TITLE + name + END;
    }

    /** Today's text: every mark replaced by its plain name. */
    public static String strip(String text) {
        if (text == null || text.indexOf(START) < 0) {
            return text;
        }
        Matcher matcher = MARK.matcher(text);
        StringBuilder result = new StringBuilder(text.length());
        while (matcher.find()) {
            matcher.appendReplacement(result, Matcher.quoteReplacement(matcher.group(2)));
        }
        matcher.appendTail(result);
        return result.toString();
    }

    /** A decision's text for the current reader: raw inside {@link #raw}, stripped otherwise. */
    public static String read(String text) {
        return RAW.get() ? text : strip(text);
    }

    /** Decision parameters for the current reader; a stripped copy only when a value carries a mark. */
    public static Map<String, String[]> read(Map<String, String[]> parameters) {
        if (RAW.get() || !marked(parameters)) {
            return parameters;
        }
        // Same iteration order as the original: serialized output stays byte-identical.
        Map<String, String[]> result = new LinkedHashMap<>(parameters.size() * 2);
        for (Map.Entry<String, String[]> entry : parameters.entrySet()) {
            String[] values = entry.getValue();
            if (values == null) {
                result.put(entry.getKey(), null);
                continue;
            }
            String[] copy = new String[values.length];
            for (int i = 0; i < values.length; i++) {
                copy[i] = strip(values[i]);
            }
            result.put(entry.getKey(), copy);
        }
        return result;
    }

    private static boolean marked(Map<String, String[]> parameters) {
        for (String[] values : parameters.values()) {
            if (values != null) {
                for (String value : values) {
                    if (value != null && value.indexOf(START) >= 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /** Runs an encoding with the marks visible when {@code marked} (the seat's bundle uses normalizer v2). */
    public static <T> T raw(boolean marked, Supplier<T> encoding) {
        if (!marked) {
            return encoding.get();
        }
        boolean previous = RAW.get();
        RAW.set(Boolean.TRUE);
        try {
            return encoding.get();
        } finally {
            RAW.set(previous);
        }
    }
}
