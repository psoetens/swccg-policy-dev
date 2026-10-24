package com.gempukku.swccgo.bench;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Random;

/** Corpus-only per-game shuffle seeds, independent of scheduling across game threads. */
public final class CorpusRandom extends Random {
    private static final CorpusRandom INSTANCE = new CorpusRandom();
    private final ThreadLocal<Random> games = new ThreadLocal<>();

    public static synchronized void seed(long seed) {
        try {
            Field random = Collections.class.getDeclaredField("r");
            random.setAccessible(true);
            random.set(null, INSTANCE);
            INSTANCE.games.set(new Random(seed));
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(exception);
        }
    }

    @Override
    protected int next(int bits) {
        Random random = games.get();
        if (random == null) {
            throw new IllegalStateException("Corpus shuffle on an unseeded game thread");
        }
        return random.nextInt() >>> (32 - bits);
    }
}
