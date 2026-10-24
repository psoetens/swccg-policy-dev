package com.gempukku.swccgo.game.knowledge;

/** Blueprint references for title-bearing stored strings and counters. */
public final class ContextTitles {
    public static final int CANONICAL = 32;
    public static final int REFERENCE = 64;

    public static void normalizeAll(ObservationV2 observation) {
        int length = observation.tokens.size();
        for (int index = 0; index < length; index++) {
            ObservationV2.Token token = observation.tokens.get(index);
            if (token.kind == 7) {
                normalize(observation, token, index);
            }
        }
    }

    private static void normalize(ObservationV2 observation, ObservationV2.Token token, int source) {
        String key = token.contextKey;
        if (key.contains(".type=") || key.contains(".opaque=") || StoredContextV5.owns(key)) {
            return;
        }
        TitleMatcher matcher = TitleMatcher.shared();
        TitleMatcher.Counter counter = matcher.counter(key, observation.titleBlueprints());
        if (counter != null) {
            token.contextKey = counter.key();
            token.bp = counter.reference().blueprint();
            token.flags |= counter.reference().flags() == 0 ? 0 : CANONICAL;
            return;
        }
        TitleMatcher.References references = new TitleMatcher.References();
        String normalized = matcher.contextTemplate(key, references, observation.titleBlueprints());
        if (references.isEmpty()) {
            return;
        }
        token.contextKey = normalized;
        token.bp = references.get(0);
        token.flags |= TitleMatcher.flag(references, 0) == 0 ? 0 : CANONICAL;
        for (int ordinal = 1; ordinal < references.size(); ordinal++) {
            ObservationV2.Token reference = new ObservationV2.Token();
            reference.kind = 7;
            reference.contextKey = "context.titleReference";
            reference.bp = references.get(ordinal);
            reference.parent = source;
            reference.b = ordinal;
            reference.a = -1;
            reference.c = -1;
            reference.d = -1;
            reference.count = -1;
            reference.flags = REFERENCE | (TitleMatcher.flag(references, ordinal) == 0 ? 0 : CANONICAL);
            observation.tokens.add(reference);
        }
    }
}
