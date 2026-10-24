package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Plan brainiac-v6 (-Dpolicy.checkBrainiacEncoder=summary.json, schema 4): every decision's record line from production
 * Brainiac's encoder (ai.brainiac.StateEncoder, nothing hidden) must equal the training harness's, byte for byte. Both
 * feed the shared ActorTensors, so equal lines mean equal tensors. Throws on the first difference; the summary is
 * written at exit.
 */
final class BrainiacEncoderCheck {
    private static long checked;
    private static final Path SUMMARY;

    static {
        String path = System.getProperty("policy.checkBrainiacEncoder");
        SUMMARY = path == null ? null : Path.of(path);
        if (SUMMARY != null) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    Files.writeString(SUMMARY, "{\"decisions_checked\": " + checked + ", \"mismatches\": 0}\n");
                } catch (IOException exception) {
                    throw new UncheckedIOException(exception);
                }
            }));
        }
    }

    private BrainiacEncoderCheck() {
    }

    static boolean enabled() {
        return SUMMARY != null;
    }

    static synchronized void check(StateEncoder.Encoded encoded, SwccgGame game, String player,
            AwaitingDecision decision) {
        String bench = encoded.line("parity", 0, List.of());
        String production = com.gempukku.swccgo.ai.brainiac.StateEncoder.encode(game, player, decision, Set.of())
            .line("parity", 0, List.of());
        if (!bench.equals(production)) {
            int index = 0;
            while (index < Math.min(bench.length(), production.length())
                    && bench.charAt(index) == production.charAt(index)) {
                index++;
            }
            throw new IllegalStateException("Brainiac/training encoder mismatch at character " + index + ": bench "
                + bench.substring(Math.max(0, index - 80), Math.min(bench.length(), index + 80)) + " production "
                + production.substring(Math.max(0, index - 80), Math.min(production.length(), index + 80)));
        }
        checked++;
    }
}
