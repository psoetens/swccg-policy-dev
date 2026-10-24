package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.cards.GameConditions;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.StringJoiner;
import java.util.zip.GZIPOutputStream;

/**
 * Plan objective-dojo BL-20(b), diagnostic only (-Dpolicy.locationProbe=true with TrainGen --export): for every exported
 * decision, one label line next to the shard line, in the same order. Per location (permanent-card order, as the
 * observation's location tokens): the engine's own occupy/control, total power and ability, and Force drain amount,
 * for the deciding player ("us") and the opponent ("them"). Read-only: nothing here changes play.
 */
public final class LocationProbe {
    public static final boolean ENABLED = Boolean.getBoolean("policy.locationProbe");
    private static final ThreadLocal<Writer> WRITER = new ThreadLocal<>();

    private LocationProbe() {
    }

    public static void open(String exportDir, String tag, int thread) {
        if (!ENABLED || exportDir == null) return;
        try {
            WRITER.set(new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(
                new File(exportDir, "labels-" + tag + "-t" + thread + ".jsonl.gz")), 1 << 16), StandardCharsets.UTF_8), 1 << 16));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static void close() {
        Writer writer = WRITER.get();
        if (writer == null) return;
        try {
            writer.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            WRITER.remove();
        }
    }

    public static void write(SwccgGame game, String player, String gameId, int decision) {
        Writer writer = WRITER.get();
        if (writer == null) return;
        GameState gs = game.getGameState();
        String opponent = game.getOpponent(player);
        List<PhysicalCard> locations = new ArrayList<>();
        for (PhysicalCard card : gs.getAllPermanentCards())
            if (card.getZone() == Zone.LOCATIONS || card.getZone() == Zone.CONVERTED_LOCATIONS) locations.add(card);
        locations.sort(Comparator.comparingInt(PhysicalCard::getPermanentCardId));
        StringJoiner rows = new StringJoiner(",", "[", "]");
        for (PhysicalCard location : locations) {
            StringJoiner row = new StringJoiner(",", "[", "]");
            row.add("\"" + location.getBlueprintId(true) + "\"");
            for (String side : new String[]{player, opponent}) {
                row.add(GameConditions.occupies(game, side, location) ? "1" : "0");
                row.add(GameConditions.controls(game, side, location) ? "1" : "0");
            }
            for (String side : new String[]{player, opponent}) {
                row.add(number(game.getModifiersQuerying().getTotalPowerAtLocation(gs, location, side, false, false)));
                row.add(number(game.getModifiersQuerying().getTotalAbilityAtLocation(gs, side, location)));
            }
            for (String side : new String[]{player, opponent}) {
                boolean occupies = GameConditions.occupies(game, side, location);
                row.add(number(occupies ? game.getModifiersQuerying().getForceDrainAmount(gs, location, side) : 0));
            }
            rows.add(row.toString());
        }
        try {
            writer.write("{\"g\":\"" + gameId + "\",\"d\":" + decision + ",\"p\":\"" + player + "\",\"locs\":" + rows + "}\n");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static String number(float value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }
}
