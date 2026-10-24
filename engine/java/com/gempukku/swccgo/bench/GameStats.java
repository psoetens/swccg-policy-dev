package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.cards.GameConditions;
import com.gempukku.swccgo.common.CardCategory;
import com.gempukku.swccgo.filters.Filters;
import com.gempukku.swccgo.logic.TriggerConditions;
import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.game.AbstractActionProxy;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.actions.TriggerAction;
import com.gempukku.swccgo.logic.timing.EffectResult;
import com.gempukku.swccgo.logic.timing.results.DoubleSidedCardFlippedResult;
import com.gempukku.swccgo.logic.timing.results.ForfeitedCardToLostPileFromTableResult;
import com.gempukku.swccgo.logic.timing.results.LostForceResult;

import com.gempukku.swccgo.communication.GameStateListener;
import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;
import com.gempukku.swccgo.logic.vo.SwccgDeck;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Passive per-game statistics (plan game-stats-objectives): objective flips / flip-backs, Force lost
 * to drains and battle damage, drains, battle forfeits, and life force at game end. Observes effect
 * results through an until-end-of-game proxy that never returns actions, so play is unchanged.
 */
public final class GameStats {
    private static final String[] SIDES = {"ds", "ls"};
    private final String[] players;
    private final Set<EffectResult> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    private final List<String> flips = new ArrayList<>();
    private final int[] flipCount = new int[2], flipBackCount = new int[2], drains = new int[2],
        lostDrain = new int[2], lostBattle = new int[2], lostOther = new int[2], forfeits = new int[2],
        activated = new int[2];
    // -Dtraingen.statsCheck=true: independent count of the engine's own chat messages, for validation only.
    private final boolean check = Boolean.getBoolean("traingen.statsCheck");
    /** -Dtraingen.flipLog=DIR: keep the engine's game messages and write them to DIR/<game>.log if an earned objective flip happened. */
    private static final String FLIP_LOG = System.getProperty("traingen.flipLog");
    /** Plan dojo-sweep: -Dtraingen.gameLog=DIR writes every game's engine messages to DIR/<game>.log (debugging). */
    private static final String GAME_LOG = System.getProperty("traingen.gameLog");
    private final List<String> messages = new ArrayList<>();
    private boolean earnedFlip;
    private boolean listening;
    private final int[] messageLoses = new int[2];
    private int messageFlips;

    private String gameId;
    /** Plan objective-dojo: the always-on Dojo reward model of this game. */
    private ObjectiveDojo.Game dojo;

    private GameStats(String ds, String ls) {
        players = new String[]{ds, ls};
    }

    public static GameStats attach(SwccgGame game, String ds, String ls, String gameId, Map<String, SwccgDeck> decks,
                                   SwccgCardBlueprintLibrary library) {
        GameStats stats = new GameStats(ds, ls);
        stats.gameId = gameId;
        stats.dojo = new ObjectiveDojo.Game(gameId, ds, ls, decks, library);
        game.getActionsEnvironment().addUntilEndOfGameActionProxy(new AbstractActionProxy() {
            @Override
            public List<TriggerAction> getRequiredAfterTriggers(SwccgGame g, EffectResult result) {
                if (stats.seen.add(result)) stats.observe(g, result);
                // Gemp offers a result to triggers again after later effects (cards re-check their conditions then):
                // the Dojo re-evaluates its state milestones each time; events are counted once.
                else stats.dojo.restate(g, result);
                return null;
            }
        });
        return stats;
    }

    private int side(String player) {
        return players[0].equals(player) ? 0 : players[1].equals(player) ? 1 : -1;
    }

    private static int turn(GameState gs) {
        return gs.getPlayersLatestTurnNumber(gs.getDarkPlayer()) + gs.getPlayersLatestTurnNumber(gs.getLightPlayer());
    }

    private void listen(SwccgGame game) {
        listening = true;
        game.addGameStateListener(players[0], (GameStateListener) Proxy.newProxyInstance(
            GameStateListener.class.getClassLoader(), new Class<?>[]{GameStateListener.class}, (proxy, method, args) -> {
                if (method.getName().equals("sendMessage") && args != null && args[0] instanceof String) {
                    String m = (String) args[0];
                    for (int s = 0; s < 2; s++) if (m.startsWith(players[s] + " loses a Force")) messageLoses[s]++;
                    if (m.contains(" is flipped to ")) messageFlips++;
                    if (FLIP_LOG != null || GAME_LOG != null) messages.add(m.replaceAll("<[^>]*>", ""));
                }
                Class<?> r = method.getReturnType();
                return r == boolean.class ? Boolean.FALSE : r == int.class ? 0 : null;
            }));
    }

    private void observe(SwccgGame game, EffectResult result) {
        if ((check || FLIP_LOG != null || GAME_LOG != null) && !listening) listen(game);
        dojo.observe(game, result);
        switch (result.getType()) {
            case DOUBLE_SIDED_CARD_FLIPPED: {
                PhysicalCard card = ((DoubleSidedCardFlippedResult) result).getCardFlipped();
                int s = side(card.getOwner());
                if (s < 0) return;
                boolean toBack = card.isFlipped();
                (toBack ? flipCount : flipBackCount)[s]++;
                boolean objective = card.getBlueprint().getCardCategory() == CardCategory.OBJECTIVE;
                if (objective && toBack && game.getGameState().getCurrentPhase() != Phase.PLAY_STARTING_CARDS) earnedFlip = true;
                // Automatic flips while starting cards are played (e.g. Carbon Chamber Testing without a Rebel) are not earned.
                boolean starting = game.getGameState().getCurrentPhase() == Phase.PLAY_STARTING_CARDS;
                flips.add(String.format(Locale.ROOT, "{\"side\":\"%s\",\"card\":\"%s\",\"title\":%s,\"objective\":%b,\"to_back\":%b,\"turn\":%d,\"starting\":%b}",
                    SIDES[s], card.getBlueprintId(true), json(card.getBlueprint().getTitle()), objective, toBack, turn(game.getGameState()), starting));
                return;
            }
            case FORCE_LOST: {
                LostForceResult lost = (LostForceResult) result;
                int s = side(lost.getPerformingPlayerId());
                if (s < 0) return;
                (lost.isFromForceDrain() ? lostDrain : lost.isBattleDamage() ? lostBattle : lostOther)[s]++;
                return;
            }
            case FORCE_ACTIVATED: {
                // Plan critic-replay: Force activated per side (one result per card moved to the Force Pile).
                int s = side(result.getPerformingPlayerId());
                if (s >= 0) activated[s]++;
                return;
            }
            case FORCE_DRAIN_COMPLETED: {
                int s = side(result.getPerformingPlayerId());
                if (s >= 0) drains[s]++;
                return;
            }
            case FORFEITED_TO_LOST_PILE_FROM_TABLE: {
                PhysicalCard card = ((ForfeitedCardToLostPileFromTableResult) result).getCard();
                int s = card == null ? -1 : side(card.getOwner());
                if (s >= 0) forfeits[s]++;
                return;
            }
            default:
        }
    }

    /** One JSON object; force_lost_* counts are Force a side LOST (drains it suffered, damage it took). */
    public String json(SwccgGame game) {
        // The game ends as soon as life force is depleted, before the last results reach the proxies;
        // the game is over, so draining the pending results cannot change play.
        if (game.isFinished())
            for (EffectResult result : game.getActionsEnvironment().consumeEffectResults())
                if (seen.add(result)) observe(game, result);
        if (gameId != null) BinaryTrajectoryWriter.releaseDojo(gameId);
        GameState gs = game.getGameState();
        String logDir = GAME_LOG != null ? GAME_LOG : FLIP_LOG != null && earnedFlip ? FLIP_LOG : null;
        if (logDir != null && gameId != null) {
            try {
                java.nio.file.Path dir = java.nio.file.Path.of(logDir);
                java.nio.file.Files.createDirectories(dir);
                java.nio.file.Files.write(dir.resolve(gameId + ".log"), messages);
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
        StringBuilder b = new StringBuilder("{");
        for (int s = 0; s < 2; s++) {
            PhysicalCard objective = gs == null ? null : gs.getObjectivePlayed(players[s]);
            b.append(String.format(Locale.ROOT,
                "\"%s\":{\"life_force\":%d,\"objective\":%s,\"objective_flipped_at_end\":%b,\"objective_zone_at_end\":%s,\"objective_in_play_at_end\":%b,\"flips\":%d,\"flip_backs\":%d,"
                    + "\"drains\":%d,\"force_lost_drain\":%d,\"force_lost_battle\":%d,\"force_lost_other\":%d,\"forfeits\":%d,\"force_activated\":%d},",
                SIDES[s], gs == null ? -1 : gs.getPlayerLifeForce(players[s]),
                objective == null ? "null" : json(objective.getBlueprintId(true)), objective != null && objective.isFlipped(),
                objective == null || objective.getZone() == null ? "null" : json(objective.getZone().name()),
                objective != null && objective.getZone() != null && objective.getZone().isInPlay(),
                flipCount[s], flipBackCount[s], drains[s], lostDrain[s], lostBattle[s], lostOther[s], forfeits[s], activated[s]));
        }
        if (check) b.append(String.format(Locale.ROOT, "\"check\":{\"ds_loses_msg\":%d,\"ls_loses_msg\":%d,\"flips_msg\":%d},",
            messageLoses[0], messageLoses[1], messageFlips));
        b.append("\"dojo\":").append(dojo.json()).append(",");
        b.append("\"flip_events\":[").append(String.join(",", flips)).append("]}");
        return b.toString();
    }

    private static String json(String s) {
        return s == null ? "null" : "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
