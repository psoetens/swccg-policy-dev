import com.gempukku.swccgo.bench.TrainGen;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Plan links-round2 (balanced matchups): a pairing listed in -Dtraingen.pairWeights at weight 0.25 is drawn at a quarter
 * of the rate of the unlisted pairings; the draw is deterministic per (pairSeed, game).
 *
 *   printf 'a.xml\tx.xml\t0.25\n' > W; efficient_java_check.sh -Dtraingen.pairWeights=W PairWeightsCheck
 */
public final class PairWeightsCheck {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        check(TrainGen.PAIR_WEIGHTS.size() == 1, "one weighted pairing loaded: " + TrainGen.PAIR_WEIGHTS);
        List<String> ds = List.of("a.xml", "b.xml"), ls = List.of("x.xml", "y.xml");
        Map<String, Integer> counts = new HashMap<>();
        int games = 400_000;
        for (int g = 0; g < games; g++) {
            java.util.Random rnd = new java.util.Random(990001L * 1_000_003L + g);
            String[] p = TrainGen.weightedPairing(ds.get(rnd.nextInt(2)), ls.get(rnd.nextInt(2)), ds, ls, g % 2 == 0, 990001L, g);
            counts.merge(p[0] + "/" + p[1], 1, Integer::sum);
        }
        double other = (counts.get("a.xml/y.xml") + counts.get("b.xml/x.xml") + counts.get("b.xml/y.xml")) / 3.0;
        double ratio = counts.get("a.xml/x.xml") / other;
        System.out.printf("pair counts %s, weighted / unweighted rate %.4f%n", counts, ratio);
        check(Math.abs(ratio - 0.25) < 0.01, "weighted pairing drawn at 1/4: " + ratio);
        check(String.join("/", TrainGen.weightedPairing("a.xml", "x.xml", ds, ls, true, 7L, 11))
                  .equals(String.join("/", TrainGen.weightedPairing("a.xml", "x.xml", ds, ls, true, 7L, 11))), "deterministic per game");
        System.out.println("PairWeightsCheck OK");
    }
}
