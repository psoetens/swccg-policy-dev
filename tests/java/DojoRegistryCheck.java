import com.gempukku.swccgo.bench.ObjectiveDojo;
import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Plan objective-dojo A-1 / A-3 (registry): loading ObjectiveDojo validates that every configured milestone has a
 * predicate and every predicate is used. Then each schedule or pool file given as an argument goes through the Java
 * coverage gate. Exit 0 = all covered; a missing reward throws with the gate's instruction.
 *
 * <p>Arguments: schedule files (TSV "ds&lt;TAB&gt;ls" or pool "DS|LS&lt;TAB&gt;path").
 */
public final class DojoRegistryCheck {
    public static void main(String[] args) throws Exception {
        SwccgCardBlueprintLibrary library = new SwccgCardBlueprintLibrary();
        System.out.println("dojo items: " + ObjectiveDojo.itemCount());
        for (String argument : args) {
            List<String> decks = new ArrayList<>();
            for (String line : Files.readAllLines(Path.of(argument))) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] parts = line.split("\t");
                if (parts[0].trim().equalsIgnoreCase("DS") || parts[0].trim().equalsIgnoreCase("LS")) decks.add(parts[1].trim());
                else for (String part : parts) if (part.trim().endsWith(".xml")) decks.add(part.trim());
            }
            ObjectiveDojo.requireFiles(decks, library);
            System.out.println("covered: " + argument + " (" + decks.stream().distinct().count() + " decks)");
        }
    }
}
