import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;
import com.gempukku.swccgo.common.CardCategory;
import com.gempukku.swccgo.logic.GameUtils;
import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Read-only static display metadata for exactly the immutable card-table IDs. */
public final class TitleCatalogDump {
    record Title(String blueprint, String title, String display, boolean perSystem, String system,
                 boolean virtual, boolean alternateImage, boolean unavailableInEngine) {}

    public static void main(String[] args) throws Exception {
        SwccgCardBlueprintLibrary library = new SwccgCardBlueprintLibrary();
        List<Title> titles = new ArrayList<>();
        for (String row : Files.readAllLines(Path.of(args[0]))) {
            String[] fields = row.split("\t", -1);
            String id = fields[0];
            var blueprint = library.getSwccgoCardBlueprint(id);
            if (blueprint == null) {
                titles.add(new Title(id, fields[1], fields[1], false, null, false, false, true));
                continue;
            }
            titles.add(new Title(id, blueprint.getTitle(), GameUtils.getFullName(blueprint),
                blueprint.getUniqueness() != null && blueprint.getUniqueness().isPerSystem(),
                blueprint.getCardCategory() == CardCategory.LOCATION ? blueprint.getSystemName() : null,
                blueprint.hasVirtualSuffix(), blueprint.hasAlternateImageSuffix(), false));
        }
        Path output = Path.of(args[1]);
        if (Files.exists(output)) {
            throw new IllegalArgumentException("Refusing to overwrite title metadata: " + output);
        }
        Files.writeString(output, new Gson().toJson(titles) + "\n");
        System.out.println("Static title blueprints: " + titles.size());
    }
}
