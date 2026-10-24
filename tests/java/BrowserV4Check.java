package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.ai.brainiac.BrainiacRelease;
import com.gempukku.swccgo.game.knowledge.ActorTensors;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Production browser packing must exactly match the native actor tensors. Plan brainiac-v6: the release is v6-only
 * (encoder 5, tensor ABI 5); run with -Dpolicy.knowledge=true -Dpolicy.context=5, as the Brainiac server does.
 * Arguments: records.jsonl, release directory, output JSON (base64 browser payloads).
 */
public final class BrowserV4Check {
    public static void main(String[] args) throws Exception {
        BrainiacRelease release = BrainiacRelease.load(Path.of(args[1]));
        PolicyTensors.Maps maps = PolicyTensors.Maps.load(Path.of(args[1]));
        List<Map<String, Object>> results = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            JsonArray full = JsonParser.parseString(line).getAsJsonArray();
            if (full.size() == 15) {
                try {
                    release.tensors(line);
                    throw new AssertionError("Browser admitted a privileged envelope");
                } catch (IllegalArgumentException expected) {
                    // Production actor entry point must refuse a training envelope.
                }
            }
            JsonArray actor = new JsonArray();
            for (int index = 0; index < 12; index++) {
                actor.add(full.get(index));
            }
            ActorTensors.Tensors browser = release.tensors(actor.toString());
            PolicyTensors nativeTensor = PolicyTensorsV4.fromRecord(actor, maps, false);
            if (browser.floatWidth != 8 || !Arrays.equals(browser.tok32, nativeTensor.tok32)
                    || !Arrays.equals(browser.tokf, nativeTensor.tokf)
                    || !Arrays.equals(browser.opt, nativeTensor.opt)
                    || !Arrays.equals(browser.head, nativeTensor.head)
                    || !Arrays.equals(browser.kgrid, nativeTensor.kgrid)) {
                throw new AssertionError("Production browser/native tensor mismatch");
            }
            results.add(Map.of("payload", Base64.getEncoder().encodeToString(browser.browserBytes()),
                "tokens", browser.tokens, "options", browser.options, "grids", browser.gridRows));
        }
        Files.writeString(Path.of(args[2]), new Gson().toJson(results));
        System.out.println("Browser ABI-5 exact tensor records: " + results.size());
    }
}
