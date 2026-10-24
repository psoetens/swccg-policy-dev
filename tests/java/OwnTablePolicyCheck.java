package com.gempukku.swccgo.bench;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Historical ABI-4 opponents must resolve templates with their own explicit bundle. */
public final class OwnTablePolicyCheck {
    public static void main(String[] args) throws Exception {
        PolicyEngine engine = PolicyEngine.get();
        JsonArray results = new JsonArray();
        int index = 0;
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            JsonObject result = new JsonObject();
            for (String key : new String[]{"learner", "ck:" + args[1]}) {
                StateEncoder.Encoded encoded = new StateEncoder.Encoded();
                List<Integer> ranking = engine.query(key + "@greedy", "own-table-" + index, "DS",
                    line, line, encoded, 1, 0, false, null);
                result.addProperty(key.equals("learner") ? "grown" : "old", ranking.get(0));
            }
            results.add(result);
            index++;
        }
        Files.writeString(Path.of(args[2]), results.toString());
        engine.close();
    }
}
