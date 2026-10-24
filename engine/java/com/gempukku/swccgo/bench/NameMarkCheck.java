package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Plan titles-and-learner-ablation A1 (-Dpolicy.checkNameMarks=FILE): every decision as its non-encoder readers see it
 * (clients, the serializer, the built-in AIs): text, parameters in their iteration order, and the answer taken. It
 * proved the marks invisible (two seeded runs, marks off and on, byte-identical) while the marks still had a switch;
 * any two seeded runs with a pinned identity hash still must write byte-identical files.
 */
final class NameMarkCheck {
    private static final PrintWriter OUT;

    static {
        String path = System.getProperty("policy.checkNameMarks");
        try {
            OUT = path == null ? null : new PrintWriter(path, StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
        if (OUT != null) {
            Runtime.getRuntime().addShutdownHook(new Thread(OUT::close));
        }
    }

    private NameMarkCheck() {
    }

    static void decision(String game, int index, String player, AwaitingDecision decision, String answer) {
        if (OUT == null) {
            return;
        }
        StringBuilder line = new StringBuilder();
        line.append(game).append('\t').append(index).append('\t').append(player).append('\t')
            .append(decision.getDecisionType()).append('\t').append(decision.getText());
        for (Map.Entry<String, String[]> entry : decision.getDecisionParameters().entrySet()) {
            line.append('\t').append(entry.getKey()).append('=');
            String[] values = entry.getValue();
            if (values != null) {
                line.append(String.join("\u001f", values));
            }
        }
        line.append('\t').append(answer);
        synchronized (OUT) {
            OUT.println(line);
        }
    }
}
