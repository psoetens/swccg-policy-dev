import com.gempukku.swccgo.bench.CorpusRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

public final class CorpusRandomCheck {
    public static void main(String[] args) throws Exception {
        try (var workers = Executors.newFixedThreadPool(8)) {
            List<Future<?>> jobs = new ArrayList<>();
            for (int game = 0; game < 32; game++) {
                final int seed = 885001 + game;
                jobs.add(workers.submit(() -> {
                    CorpusRandom.seed(seed);
                    Random reference = new Random(seed);
                    for (int repeat = 0; repeat < 20; repeat++) {
                        List<Integer> expected = new ArrayList<>(IntStream.range(0, 60).boxed().toList());
                        List<Integer> actual = new ArrayList<>(expected);
                        Collections.shuffle(expected, reference);
                        Collections.shuffle(actual);
                        if (!expected.equals(actual)) {
                            throw new AssertionError("Parallel shuffle stream changed: " + seed);
                        }
                    }
                }));
            }
            for (Future<?> job : jobs) {
                job.get();
            }
        }
        System.out.println("{\"passed\":true,\"games\":32,\"shuffles_per_game\":20}");
    }
}
