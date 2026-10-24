import com.gempukku.swccgo.bench.TrainGen;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Plan links-round2 W-bal: -Dtraingen.deckWeights reweights each seat's deck draw by its factor (unlisted: "*").
 * With "* 0.5", "a.xml 1" and "x.xml 0.125": DS a is drawn 2x as often as b, LS x at 1/4 the rate of y.
 *
 *   printf '*\t0.5\na.xml\t1\nx.xml\t0.125\n' > W; efficient_java_check.sh -Dtraingen.deckWeights=W DeckWeightsCheck
 */
public final class DeckWeightsCheck {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        check(TrainGen.DECK_WEIGHTS.size() == 3, "three entries: " + TrainGen.DECK_WEIGHTS);
        List<String> ds = List.of("a.xml", "b.xml"), ls = List.of("x.xml", "y.xml");
        Map<String, Integer> counts = new HashMap<>();
        for (int g = 0; g < 400_000; g++) {
            java.util.SplittableRandom first = new java.util.SplittableRandom(g * 7919L + 13);
            String[] p = TrainGen.weightedPairing(ds.get(first.nextInt(2)), ls.get(first.nextInt(2)), ds, ls, g % 2 == 0, 990001L, g);
            counts.merge(p[0], 1, Integer::sum);
            counts.merge(p[1], 1, Integer::sum);
        }
        double dsRatio = counts.get("a.xml") / (double) counts.get("b.xml");
        double lsRatio = counts.get("x.xml") / (double) counts.get("y.xml");
        System.out.printf("counts %s: DS a/b %.4f (2), LS x/y %.4f (0.25)%n", counts, dsRatio, lsRatio);
        check(Math.abs(dsRatio - 2.0) < 0.05, "DS factor 2x: " + dsRatio);
        check(Math.abs(lsRatio - 0.25) < 0.01, "LS factor 1/4: " + lsRatio);
        System.out.println("DeckWeightsCheck OK");
    }
}
