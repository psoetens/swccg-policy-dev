package com.gempukku.swccgo.ai.brainiac;

import com.gempukku.swccgo.game.knowledge.ActorTensors;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A Brainiac model release (plan brainiac-v6: one format, the v6 policy), a directory written by
 * swccg-policy's scripts/brainiac_release.py:
 * <ul>
 *   <li>manifest.json: name, model file, its SHA-256 and size, runtime, encoder 5, tensor ABI 5, table SHA, and the
 *       checkpoint it was exported from</li>
 *   <li>the table bundle: vocab.json, cards.json, bundle.json (validated by {@link ActorTensors.Tables#load})</li>
 *   <li>titles.tsv: the frozen title table training encoded with (the engine's policy.titleTable)</li>
 *   <li>the model file itself (ONNX), which only the browser uses</li>
 * </ul>
 * Encoder 5 = the schema-4 observation with context version 5, tensor ABI 5 (eight floats per token). Any other
 * release is refused at load. The browser runtime (onnxruntime-web) lives next to the releases, in
 * runtime/{runtime}/.
 */
public final class BrainiacRelease {
    public static final int ENCODER = 5;
    public final Path dir;
    public final String name;
    public final String modelFile;
    public final String modelSha256;
    public final long modelBytes;
    public final String runtime;
    public final ActorTensors.Tables tables;

    private BrainiacRelease(Path dir, JsonObject manifest) throws IOException {
        this.dir = dir;
        name = manifest.get("name").getAsString();
        modelFile = manifest.get("model").getAsString();
        modelSha256 = manifest.get("sha256").getAsString();
        modelBytes = manifest.get("bytes").getAsLong();
        runtime = manifest.get("runtime").getAsString();
        if (!runtime.matches("[A-Za-z0-9._-]+"))
            throw new IllegalArgumentException("invalid runtime name " + runtime);
        int encoder = manifest.has("encoder") ? manifest.get("encoder").getAsInt() : 1;
        int abi = manifest.has("tensor_abi") ? manifest.get("tensor_abi").getAsInt() : 0;
        if (encoder != ENCODER || abi != 5)
            throw new IllegalArgumentException("release " + name + " has encoder " + encoder + " / tensor ABI " + abi
                    + "; this server serves only encoder " + ENCODER + " / tensor ABI 5 (v6 policies)");
        // The engine must track knowledge from game start and emit context version 5 (see brainiac_server.sh).
        if (!Boolean.getBoolean("policy.knowledge") || !"5".equals(System.getProperty("policy.context")))
            throw new IllegalStateException("Brainiac needs -Dpolicy.knowledge=true -Dpolicy.context=5");
        Path titles = dir.resolve(manifest.get("titles").getAsString()).normalize();
        String configured = System.getProperty("policy.titleTable");
        if (configured != null && !Path.of(configured).toAbsolutePath().normalize().equals(titles))
            throw new IllegalStateException("policy.titleTable is " + configured + ", the release needs " + titles);
        if (!Files.isRegularFile(titles))
            throw new IllegalArgumentException("release " + name + " lacks its title table " + titles);
        System.setProperty("policy.titleTable", titles.toString());
        tables = ActorTensors.Tables.load(dir);
        if (tables.tensorAbi != 5 || tables.contextVersion != 5)
            throw new IllegalArgumentException("release " + name + " tables are not tensor ABI 5 / context 5");
        if (!tables.tableSha.equals(manifest.get("table_sha").getAsString()))
            throw new IllegalArgumentException("Release/table SHA mismatch");
    }

    public static BrainiacRelease load(Path dir) throws IOException {
        dir = dir.toAbsolutePath().normalize();
        String manifest = Files.readString(dir.resolve("manifest.json"), StandardCharsets.UTF_8);
        return new BrainiacRelease(dir, JsonParser.parseString(manifest).getAsJsonObject());
    }

    /** The actor tensors of one encoded decision (a 12-field schema-4 record line; a training envelope is refused). */
    public ActorTensors.Tensors tensors(String recordLine) {
        var record = JsonParser.parseString(recordLine).getAsJsonArray();
        if (record.size() != 12)
            throw new IllegalArgumentException("Brainiac encodes the actor view only");
        return ActorTensors.fromRecord(record, tables, false);
    }

    public Path modelPath() {
        return dir.resolve(modelFile);
    }

    /** A file of the browser runtime (repository/runtime/{runtime}/{fileName}), or null for a name with a path in it. */
    public Path runtimeFile(String fileName) {
        if (!fileName.matches("[A-Za-z0-9._-]+") || fileName.startsWith("."))
            return null;
        return dir.getParent().getParent().resolve("runtime").resolve(runtime).resolve(fileName);
    }
}
