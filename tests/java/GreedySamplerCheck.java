import com.gempukku.swccgo.bench.PolicySampler;

import java.util.List;
import java.util.Random;

/**
 * Plan nvf-b-local A4: temperature 0 plays the highest score (multi-pick: the top k in order; exact ties at random);
 * temperature 1 still samples.
 *
 *   javac -cp <server test-classes> -d <out> tests/java/GreedySamplerCheck.java; java -cp <out>:<classes> GreedySamplerCheck
 */
public final class GreedySamplerCheck {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        float[] logits = {1.0f, 3.0f, 2.9f, -1.0f};
        boolean[] none = {true, false, false, false};
        Random rng = new Random(1);
        for (int i = 0; i < 1000; i++)
            check(PolicySampler.sample(logits, 4, none, 1, 0.0, null, rng).seq.equals(List.of(1)), "greedy picks the top");
        check(PolicySampler.sample(logits, 4, none, 2, 0.0, null, rng).seq.equals(List.of(1, 2)), "greedy top k in order");
        check(PolicySampler.sample(logits, 4, none, 1, 0.0, new int[]{1}, rng).seq.equals(List.of(2)), "greedy skips excluded");
        float[] tied = {5.0f, 5.0f, 0.0f};
        int first = 0;
        for (int i = 0; i < 2000; i++)
            if (PolicySampler.sample(tied, 3, new boolean[3], 1, 0.0, null, rng).seq.get(0) == 0) first++;
        check(first > 800 && first < 1200, "exact ties broken at random: " + first);
        int other = 0;
        for (int i = 0; i < 2000; i++)
            if (PolicySampler.sample(logits, 4, none, 1, 1.0, null, rng).seq.get(0) != 1) other++;
        check(other > 500, "temperature 1 still samples: " + other);
        System.out.println("GreedySamplerCheck ok (ties " + first + "/2000, sampled non-top " + other + "/2000)");
    }
}
