import com.gempukku.swccgo.common.GameTextActionId;
import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.logic.modifiers.querying.ModifiersLogic;
import com.google.gson.Gson;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Map;

/** Only class metadata and enum constants; no game objects or modifier evaluation. */
public final class SeedCounterCatalog {
    public static void main(String[] args) {
        String[] families = Arrays.stream(ModifiersLogic.class.getDeclaredFields()).map(Field::getName)
            .filter(name -> name.endsWith("LimitCounters")).sorted().toArray(String[]::new);
        String[] actions = Arrays.stream(GameTextActionId.values()).map(Enum::name).toArray(String[]::new);
        String[] phases = Arrays.stream(Phase.values()).map(Enum::name).toArray(String[]::new);
        System.out.println(new Gson().toJson(Map.of("families", families, "actions", actions, "phases", phases)));
    }
}
