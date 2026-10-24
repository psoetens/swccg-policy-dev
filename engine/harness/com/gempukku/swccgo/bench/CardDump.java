package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.common.Icon;
import com.gempukku.swccgo.common.Keyword;
import com.gempukku.swccgo.common.Persona;
import com.gempukku.swccgo.game.SwccgCardBlueprint;
import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Dumps card rows (same field names as card_blueprint_database.json) straight from Gemp blueprints, for blueprint ids the
 * JSON database lacks. Usage: CardDump IDS_FILE OUT_JSONL (one blueprint id per line).
 */
public class CardDump {
    public static void main(String[] args) throws Exception {
        SwccgCardBlueprintLibrary lib = new SwccgCardBlueprintLibrary();
        List<String> ids = Files.readAllLines(Path.of(args[0]));
        int ok = 0;
        try (PrintWriter out = new PrintWriter(args[1], StandardCharsets.UTF_8)) {
            for (String id : ids) {
                id = id.trim();
                if (id.isEmpty()) continue;
                SwccgCardBlueprint b = lib.getSwccgoCardBlueprint(id);
                if (b == null) {
                    System.err.println("no blueprint: " + id);
                    continue;
                }
                StringBuilder sb = new StringBuilder("{");
                field(sb, "cardId", StateEncoder.q(id));
                field(sb, "title", q(() -> b.getTitle()));
                field(sb, "side", q(() -> b.getSide().name()));
                field(sb, "cardCategory", q(() -> b.getCardCategory().name()));
                field(sb, "cardSubtype", q(() -> b.getCardSubtype() == null ? null : b.getCardSubtype().name()));
                field(sb, "uniqueness", q(() -> b.getUniqueness() == null ? null : b.getUniqueness().name()));
                List<String> types = new ArrayList<>();
                try { b.getCardTypes().forEach(t -> types.add(t.name())); } catch (RuntimeException ignored) { }
                field(sb, "cardTypes", list(types));
                field(sb, "lore", q(() -> b.getLore()));
                field(sb, "gameText", q(() -> b.getGameText()));
                field(sb, "locationDarkSideGameText", q(() -> b.getLocationDarkSideGameText()));
                field(sb, "locationLightSideGameText", q(() -> b.getLocationLightSideGameText()));
                field(sb, "deployCost", n(() -> b.getDeployCost()));
                field(sb, "power", n(() -> b.hasPowerAttribute() ? b.getPower() : null));
                field(sb, "ability", n(() -> b.hasAbilityAttribute() ? b.getAbility() : null));
                field(sb, "forfeit", n(() -> b.hasForfeitAttribute() ? b.getForfeit() : null));
                field(sb, "destiny", n(() -> b.getDestiny()));
                field(sb, "armor", n(() -> b.hasArmorAttribute() ? b.getArmor() : null));
                field(sb, "maneuver", n(() -> b.hasManeuverAttribute() ? b.getManeuver() : null));
                field(sb, "landspeed", n(() -> b.hasLandspeedAttribute() ? b.getLandspeed() : null));
                field(sb, "hyperspeed", n(() -> b.hasHyperspeedAttribute() ? b.getHyperspeed() : null));
                field(sb, "ferocity", n(() -> b.hasFerocityAttribute() ? b.getFerocity() : null));
                field(sb, "species", q(() -> b.hasSpeciesAttribute() && b.getSpecies() != null ? b.getSpecies().name() : null));
                List<String> kw = new ArrayList<>();
                for (Keyword k : Keyword.values()) try { if (b.hasKeyword(k)) kw.add(k.name()); } catch (RuntimeException ignored) { }
                field(sb, "keywords", list(kw));
                List<String> ps = new ArrayList<>();
                try { for (Persona p : b.getPersonas()) ps.add(p.name()); } catch (RuntimeException ignored) { }
                field(sb, "personas", list(ps));
                StringBuilder ic = new StringBuilder("[");
                for (Icon i : Icon.values()) {
                    try {
                        int c = b.getIconCount(i);
                        if (c > 0) ic.append(ic.length() > 1 ? "," : "").append("{\"icon\":\"").append(i.name()).append("\",\"count\":").append(c).append('}');
                    } catch (RuntimeException ignored) { }
                }
                field(sb, "icons", ic.append(']').toString());
                sb.setLength(sb.length() - 1);
                out.println(sb.append('}'));
                ok++;
            }
        }
        System.err.println("dumped " + ok + " of " + ids.size());
    }

    private static void field(StringBuilder sb, String k, String v) {
        sb.append('"').append(k).append("\":").append(v).append(',');
    }

    private static String q(Supplier<String> s) {
        try {
            return StateEncoder.q(s.get());
        } catch (RuntimeException e) {
            return "null";
        }
    }

    private static String n(Supplier<Float> s) {
        try {
            Float f = s.get();
            if (f == null || f.isNaN()) return "null";
            return f == Math.floor(f) ? String.valueOf(f.intValue()) : String.valueOf(f);
        } catch (RuntimeException e) {
            return "null";
        }
    }

    private static String list(List<String> xs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < xs.size(); i++) sb.append(i == 0 ? "" : ",").append(StateEncoder.q(xs.get(i)));
        return sb.append(']').toString();
    }
}
