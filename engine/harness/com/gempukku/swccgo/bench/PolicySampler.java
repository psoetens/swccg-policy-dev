package com.gempukku.swccgo.bench;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Samples the option(s) PolicyAi answers with, and their behaviour log-probability: the Java port of swccg-policy's
 * rl.sample_sequence (eps = 0), used when the policy runs inside the JVM.
 *
 * A "none"/"pass" option picked first means answering nothing; otherwise up to k picks. First pick over all legal options
 * (minus the excluded ones), later picks over the remaining non-none options (Plackett-Luce). Per pick p = softmax(logits / T)
 * over the allowed options. Computed in double (the Python code uses float32 numpy, so log-probabilities agree to ~1e-6, not
 * bit for bit).
 */
public final class PolicySampler {
    /** A sampled pick sequence (option indices) and its log-probability. */
    public static final class Sample {
        public final List<Integer> seq;
        public final double logp;

        Sample(List<Integer> seq, double logp) {
            this.seq = seq;
            this.logp = logp;
        }
    }

    private PolicySampler() {
    }

    /**
     * @param logits   option logits (the first n entries are the options)
     * @param isNone   per option: "none"/"pass"
     * @param excluded options that may not be picked (may be null)
     */
    public static Sample sample(float[] logits, int n, boolean[] isNone, int k, double temperature, int[] excluded,
                                RandomGenerator rng) {
        return run(logits, n, isNone, k, temperature, excluded, rng, null);
    }

    /** Log-probability of a given pick sequence under the same rules (-Infinity if a pick is not allowed). */
    public static double logProb(float[] logits, int n, boolean[] isNone, int k, double temperature, int[] excluded, List<Integer> seq) {
        return run(logits, n, isNone, k, temperature, excluded, null, seq).logp;
    }

    private static Sample run(float[] logits, int n, boolean[] isNone, int k, double temperature, int[] excluded,
                              RandomGenerator rng, List<Integer> given) {
        boolean[] avail = new boolean[n];
        Arrays.fill(avail, true);
        if (excluded != null)
            for (int i : excluded)
                if (i >= 0 && i < n) avail[i] = false;
        double t = Math.max(temperature, 1e-6);
        List<Integer> seq = new ArrayList<>();
        double logp = 0.0;
        double[] p = new double[n];
        int[] idx = new int[n];
        for (int j = 0; j < Math.max(1, k); j++) {
            int m = 0;
            double zmax = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (!avail[i] || (j > 0 && isNone[i])) continue;
                idx[m] = i;
                p[m] = logits[i] / t;
                zmax = Math.max(zmax, p[m]);
                m++;
            }
            if (m == 0) break;
            double sum = 0;
            for (int c = 0; c < m; c++) {
                p[c] = Math.exp(p[c] - zmax);
                sum += p[c];
            }
            int c = -1;
            if (given != null) {
                if (j >= given.size()) break;
                for (int q = 0; q < m; q++) if (idx[q] == given.get(j)) c = q;
                if (c < 0) return new Sample(seq, Double.NEGATIVE_INFINITY);
            } else if (temperature <= 0) {
                // Plan nvf-b-local A4: greedy (temperature 0) plays the highest score; exact ties are broken at random.
                int ties = 0;
                for (int q = 0; q < m; q++) if (p[q] == 1.0 && rng.nextInt(++ties) == 0) c = q;
            } else {
                double x = rng.nextDouble() * sum, acc = 0;
                c = m - 1;
                for (int q = 0; q < m; q++) {
                    acc += p[q];
                    if (x < acc) {
                        c = q;
                        break;
                    }
                }
            }
            int choice = idx[c];
            seq.add(choice);
            logp += Math.log(Math.max(p[c] / sum, 1e-12));
            avail[choice] = false;
            if (j == 0 && isNone[choice]) break;
        }
        return new Sample(seq, logp);
    }
}
