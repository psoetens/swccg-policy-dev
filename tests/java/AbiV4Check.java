package com.gempukku.swccgo.bench;

import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Actual Java conversion/writer/JNI with the same records consumed by Python. */
public final class AbiV4Check {
    public static void main(String[] args) throws Exception {
        Path records = Path.of(args[0]);
        PolicyTensors.Maps maps = PolicyTensors.Maps.load(Path.of(args[1]));
        Path output = Path.of(args[2]);
        Files.createDirectories(output);
        List<PolicyTensors> actors = new ArrayList<>();
        List<PolicyTensors> critics = new ArrayList<>();
        List<Long> sequences = new ArrayList<>();
        for (String line : Files.readAllLines(records)) {
            JsonArray record = JsonParser.parseString(line).getAsJsonArray();
            actors.add(PolicyTensorsV4.fromRecord(record, maps, false));
            critics.add(PolicyTensorsV4.fromRecord(record, maps, true));
            sequences.add(record.get(13).getAsLong());
        }
        for (int level : new int[]{0, 1}) {
            System.setProperty("policy.trajectoryCompression", Integer.toString(level));
            BinaryTrajectoryWriter writer = new BinaryTrajectoryWriter(output.resolve("gzip" + level), 1);
            for (int index = 0; index < actors.size(); index++) {
                writer.step("fixture", "ds", actors.get(index), critics.get(index), sequences.get(index),
                    List.of(0), -0.25, 0, 1, null, "learner:7", maps);
            }
            writer.finish("fixture", "ds", "finished", "ds", new int[]{20, 0}, "fixture", "learner:7");
            writer.close();
        }
        try {
            PolicyTensors.shape(List.of(critics.get(0)));
            throw new AssertionError("Privileged tensors admitted to actor batch");
        } catch (IllegalArgumentException expected) {
            // Actor batching must reject an explicitly privileged payload.
        }
        if (args.length == 3) {
            return;
        }
        NativePolicy.ensureLoaded();
        NativePolicy.setTf32(false);
        for (boolean graphs : new boolean[]{false, true}) {
            long slot = NativePolicy.load(args[3], graphs);
            if (slot == 0 || NativePolicyV4.checkTable(slot, maps.tableSha) != 0) {
                throw new IllegalStateException(NativePolicy.lastError());
            }
            try {
                if (NativePolicyV4.checkTable(slot, "0".repeat(64)) == 0) {
                    throw new AssertionError("Native model accepted unrelated table SHA");
                }
                for (int batch : new int[]{1, 8}) {
                    List<String> logits = new ArrayList<>();
                    for (int begin = 0; begin < actors.size();) {
                        int end = begin;
                        while (end < actors.size() && end - begin < batch && actors.get(end).options <= 64) {
                            end++;
                        }
                        if (end == begin) {
                            end++;
                        }
                        List<PolicyTensors> items = actors.subList(begin, end);
                        int[] shape = PolicyTensors.shape(items);
                        ByteBuffer input = ByteBuffer.allocateDirect(PolicyTensors.packedBytes(shape));
                        PolicyTensors.pack(items, shape, input);
                        FloatBuffer result = ByteBuffer.allocateDirect(shape[0] * shape[2] * 4)
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                        if (NativePolicyV4.launch(slot, input, shape[0], shape[1], shape[2], shape[3]) != 0
                                || NativePolicy.collect(slot, result) != 0) {
                            throw new IllegalStateException(NativePolicy.lastError());
                        }
                        for (int index = 0; index < items.size(); index++) {
                            JsonArray row = new JsonArray();
                            for (int option = 0; option < items.get(index).options; option++) {
                                row.add(result.get(index * shape[2] + option));
                            }
                            logits.add(row.toString());
                        }
                        begin = end;
                    }
                    Files.write(output.resolve("logits-b" + batch + "-graphs" + graphs + ".jsonl"), logits);
                }
            } finally {
                NativePolicy.free(slot);
            }
        }
    }
}
