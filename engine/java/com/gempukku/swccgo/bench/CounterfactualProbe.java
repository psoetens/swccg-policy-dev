package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.filters.Filters;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.game.state.BattleState;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan critic-counterfactuals, part K: the same position twice for the critic, as played and with one change made in the
 * Gemp game state, encoded by the normal encoder. Pile changes are reverted exactly (pile lists, card zones and the
 * knowledge tracker are restored), so the game goes on unchanged; a character change (K2 / K2o) cannot be reverted, so
 * the game is no longer probed after it.
 *
 * <p>Switched on by {@code -Dpolicy.counterfactual=DIR}; needs {@code policy.binaryTrajectory} and
 * {@code policy.captureCritic}. Each pair is written as a two-step pseudo-game ({@code cf|...}) with the existing binary
 * trajectory writer into {@code DIR/traj}, and described by one JSON line in {@code DIR/pairs.jsonl}.
 *
 * <p>Positions (own turns 3-12 of the scored seat): "neutral", the first decision of an own turn; "battle", a decision
 * in the weapons segment of a battle ("Choose weapons segment action to play or Pass", battle started, power segment
 * not reached), before any weapon or battle destiny is drawn.
 */
public final class CounterfactualProbe {
    private static final String DIR = System.getProperty("policy.counterfactual");
    /** User 2026-10-13: K1 leaves the top KEEP cards of the Reserve Deck in place. A weapon destiny, a battle destiny
     * and an added destiny already draw 3, a lightsaber 2 of them: so the next 4 draws stay as they were. */
    private static final int KEEP = 4;
    /** Share of neutral positions that end with a character test (K2 or K2o, alternating). */
    private static final double K2_SHARE = Double.parseDouble(System.getProperty("policy.counterfactual.k2Share", "0.2"));
    private static final String WEAPONS = "Choose weapons segment action to play or Pass";
    private static final AtomicLong PAIRS = new AtomicLong();
    private static final AtomicLong R_MISMATCH = new AtomicLong();
    private static final Map<SwccgGame, Set<String>> SEEN = Collections.synchronizedMap(new WeakHashMap<>());
    private static final Map<SwccgGame, Boolean> TAINTED = Collections.synchronizedMap(new WeakHashMap<>());
    private static final Map<SwccgGame, Random> RNG = Collections.synchronizedMap(new WeakHashMap<>());
    private static PolicyTensors.Maps maps;
    private static BinaryTrajectoryWriter writer;
    private static PrintWriter pairs;

    private CounterfactualProbe() {
    }

    static boolean enabled() {
        return DIR != null;
    }

    // ---------------------------------------------------------------- set-up

    private static synchronized void open() {
        if (writer != null) return;
        try {
            maps = PolicyTensors.Maps.load(Paths.get(System.getProperty("policy.native.bundle")));
            Path dir = Paths.get(DIR);
            Files.createDirectories(dir);
            Field active = BinaryTrajectoryWriter.class.getDeclaredField("active");
            active.setAccessible(true);
            Object previous = active.get(null);
            writer = new BinaryTrajectoryWriter(dir, 50);
            active.set(null, previous);  // the run's own writer keeps the Dojo marks
            pairs = new PrintWriter(Files.newBufferedWriter(dir.resolve("pairs.jsonl"), StandardCharsets.UTF_8), true);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    writer.close();
                } catch (InterruptedException ignored) {
                }
                synchronized (CounterfactualProbe.class) {
                    pairs.printf(Locale.ROOT, "{\"summary\":true,\"pairs\":%d,\"r_mismatch\":%d}%n", PAIRS.get(), R_MISMATCH.get());
                    pairs.close();
                }
            }));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    // ---------------------------------------------------------------- entry

    /** Called by KnowledgeStateEncoder after the as-played encoding of a decision (original critic not yet converted). */
    static void probe(SwccgGame game, String player, AwaitingDecision decision, StateEncoder.Encoded encoded) {
        if (encoded.binary == null || encoded.binary.privileged == null || Boolean.TRUE.equals(TAINTED.get(game))) return;
        GameState gs = game.getGameState();
        int turn = gs.getPlayersLatestTurnNumber(player);
        if (turn < 3 || turn > 12) return;
        String kind = null;
        BattleState battle = gs.getBattleState();
        if (WEAPONS.equals(decision.getText()) && battle != null && battle.isBattleStarted() && !battle.isReachedPowerSegment()) {
            kind = "battle";
        } else if (player.equals(gs.getCurrentPlayerId())) {
            kind = "neutral";
        }
        if (kind == null) return;
        String key = player + "|" + turn + "|" + kind + (battle == null ? "" : "|" + System.identityHashCode(battle));
        Set<String> seen = SEEN.computeIfAbsent(game, g -> Collections.synchronizedSet(new HashSet<>()));
        if (!seen.add(key)) return;
        open();
        Random rng = RNG.computeIfAbsent(game, g -> new Random(Long.getLong("policy.native.seed", 0L) * 1_000_003L
            + System.identityHashCode(g)));
        String opponent = game.getOpponent(player);
        PolicyTensors original = encoded.binary.critic(encoded, maps);  // before any change: location state reads the live game
        PolicyTensors actor = encoded.binary.actor(encoded, maps);
        Position at = new Position(game, player, opponent, decision, kind, turn, original, actor);
        if (kind.equals("battle")) {
            at.shuffle("K3", player, Zone.RESERVE_DECK);
            at.shuffle("K3o", opponent, Zone.RESERVE_DECK);
            return;
        }
        at.lostToReserve("K1+", player, rng);
        at.lostToReserve("K1o+", opponent, rng);
        at.reserveToLost("K1-", player, rng);
        at.reserveToLost("K1o-", opponent, rng);
        at.shuffle("K3", player, Zone.RESERVE_DECK);
        at.shuffle("K3o", opponent, Zone.RESERVE_DECK);
        at.shuffle("S", player, Zone.LOST_PILE);
        at.shuffle("So", opponent, Zone.LOST_PILE);
        at.shuffle("U", player, Zone.USED_PILE);
        at.shuffle("Uo", opponent, Zone.USED_PILE);
        at.roundTrip(rng);
        if (rng.nextDouble() < K2_SHARE) {
            boolean mirror = (PAIRS.get() & 1) == 1;
            if (at.characterToLost(mirror ? "K2o" : "K2", mirror ? opponent : player, rng)) TAINTED.put(game, true);
        }
    }

    // ---------------------------------------------------------------- one position

    private static final class Position {
        final SwccgGame game;
        final GameState gs;
        final String player;
        final String opponent;
        final AwaitingDecision decision;
        final String kind;
        final int turn;
        final PolicyTensors original;
        final PolicyTensors actor;

        Position(SwccgGame game, String player, String opponent, AwaitingDecision decision, String kind, int turn,
                 PolicyTensors original, PolicyTensors actor) {
            this.game = game;
            this.gs = game.getGameState();
            this.player = player;
            this.opponent = opponent;
            this.decision = decision;
            this.kind = kind;
            this.turn = turn;
            this.original = original;
            this.actor = actor;
        }

        /** Encodes the changed state (checks off) and converts it at once: {actor, critic}. The writer needs the
         * changed actor view too, since both views share the header (globals: life force, pile sizes). */
        PolicyTensors[] changed() {
            StateEncoder.Encoded e = KnowledgeStateEncoder.comparison(game, player, decision, 4);
            return new PolicyTensors[]{e.binary.actor(e, maps), e.binary.critic(e, maps)};
        }

        void lostToReserve(String test, String owner, Random rng) {
            List<PhysicalCard> lost = new ArrayList<>(gs.getLostPile(owner));
            if (lost.size() < 3 || gs.getReserveDeck(owner, false).size() < KEEP) return;
            Saved saved = new Saved(gs, owner);
            Map<String, Object> facts = before(owner);
            for (int i = 0; i < 3; i++) {
                PhysicalCard card = gs.getLostPile(owner).get(0);  // from the top, so the tracker keeps the Lost Pile order
                gs.getKnowledge().expectRemoval(card, 0);
                gs.removeCardsFromZone(List.of(card), true, true);
                int size = gs.getReserveDeck(owner, false).size();
                int position = KEEP + rng.nextInt(size - KEEP + 1);  // below the top KEEP cards (the bottom included)
                addAt(card, Zone.RESERVE_DECK, owner, position);
            }
            finish(test, owner, saved, facts);
        }

        void reserveToLost(String test, String owner, Random rng) {
            if (gs.getReserveDeck(owner, false).size() < KEEP + 3) return;
            Saved saved = new Saved(gs, owner);
            Map<String, Object> facts = before(owner);
            for (int i = 0; i < 3; i++) {
                List<PhysicalCard> reserve = gs.getReserveDeck(owner, false);
                int position = KEEP + rng.nextInt(reserve.size() - KEEP);
                PhysicalCard card = reserve.get(position);
                gs.getKnowledge().expectRemoval(card, position);
                gs.removeCardsFromZone(List.of(card), true, true);
                addAt(card, Zone.LOST_PILE, owner, 0);
            }
            finish(test, owner, saved, facts);
        }

        void shuffle(String test, String owner, Zone pile) {
            if (gs.getCardPile(owner, pile, false).size() < 2) return;
            Saved saved = new Saved(gs, owner);
            Map<String, Object> facts = before(owner);
            gs.shufflePile(owner, pile);
            finish(test, owner, saved, facts);
        }

        /** Control R: K1+ and its revert, then the encoding must equal the original exactly. */
        void roundTrip(Random rng) {
            if (gs.getLostPile(player).size() < 3) return;
            Saved saved = new Saved(gs, player);
            Map<String, Object> facts = before(player);
            for (int i = 0; i < 3; i++) {
                PhysicalCard card = gs.getLostPile(player).get(0);
                gs.getKnowledge().expectRemoval(card, 0);
                gs.removeCardsFromZone(List.of(card), true, true);
                addAt(card, Zone.RESERVE_DECK, player, 0);
            }
            saved.restore(gs);
            PolicyTensors[] after = changed();
            saved.restore(gs);  // the encoding tightens the tracker in place
            boolean exact = same(original, after[1]) && same(actor, after[0]);
            if (!exact) R_MISMATCH.incrementAndGet();
            facts.put("exact", exact);
            write("R", player, after, facts);
        }

        /** K2 / K2o: a character present at a battleground site where the other side also has presence. Not reverted. */
        boolean characterToLost(String test, String owner, Random rng) {
            String other = game.getOpponent(owner);
            List<PhysicalCard> candidates = new ArrayList<>();
            for (PhysicalCard card : gs.getAllPermanentCards()) {
                if (!owner.equals(card.getOwner()) || card.getZone() != Zone.AT_LOCATION
                        || !Filters.character.accepts(gs, game.getModifiersQuerying(), card)) continue;
                PhysicalCard site = card.getAtLocation();
                if (site == null || card.getAttachedTo() != null
                        || !Filters.battleground_site.accepts(gs, game.getModifiersQuerying(), site)
                        || !Filters.occupies(other).accepts(gs, game.getModifiersQuerying(), site)
                        || !gs.getAttachedCards(card).isEmpty() || !gs.getStackedCards(card).isEmpty()) continue;
                candidates.add(card);
            }
            if (candidates.isEmpty()) return false;
            PhysicalCard card = candidates.get(rng.nextInt(candidates.size()));
            Map<String, Object> facts = before(owner);
            facts.put("character", card.getBlueprintId(true));
            gs.removeCardsFromZone(List.of(card), true, true);
            gs.addCardToTopOfZone(card, Zone.LOST_PILE, owner, true, true, true);
            write(test, owner, changed(), facts);
            return true;
        }

        private void addAt(PhysicalCard card, Zone zone, String owner, int position) {
            try {
                Method add = GameState.class.getDeclaredMethod("addCardToZone", PhysicalCard.class, Zone.class, String.class,
                    boolean.class, boolean.class, boolean.class, boolean.class, Integer.class);
                add.setAccessible(true);
                add.invoke(gs, card, zone, owner, true, position == 0, true, true, position == 0 ? null : position);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
        }

        private Map<String, Object> before(String owner) {
            Map<String, Object> facts = new LinkedHashMap<>();
            facts.put("lf_player", gs.getPlayerLifeForce(player));
            facts.put("lf_opponent", gs.getPlayerLifeForce(opponent));
            facts.put("top_before", topDestiny(owner));
            facts.put("top2_before", top2Destiny(owner));
            facts.put("top4_before", topIds(owner));
            return facts;
        }

        private void finish(String test, String owner, Saved saved, Map<String, Object> facts) {
            facts.put("top_after", topDestiny(owner));
            facts.put("top2_after", top2Destiny(owner));
            facts.put("top4_after", topIds(owner));
            facts.put("lf_player_after", gs.getPlayerLifeForce(player));
            facts.put("lf_opponent_after", gs.getPlayerLifeForce(opponent));
            PolicyTensors[] changed = changed();
            saved.restore(gs);
            write(test, owner, changed, facts);
        }

        private float topDestiny(String owner) {
            List<PhysicalCard> reserve = gs.getReserveDeck(owner, false);
            return reserve.isEmpty() ? Float.NaN : destiny(reserve.get(0));
        }

        private float top2Destiny(String owner) {
            List<PhysicalCard> reserve = gs.getReserveDeck(owner, false);
            return reserve.size() < 2 ? Float.NaN : destiny(reserve.get(0)) + destiny(reserve.get(1));
        }

        /** The top KEEP cards of the Reserve Deck, by permanent card id (K1 must leave them in place). */
        private String topIds(String owner) {
            List<PhysicalCard> reserve = gs.getReserveDeck(owner, false);
            StringBuilder ids = new StringBuilder();
            for (int i = 0; i < Math.min(KEEP, reserve.size()); i++) ids.append(i == 0 ? "" : "-").append(reserve.get(i).getPermanentCardId());
            return ids.toString();
        }

        private static float destiny(PhysicalCard card) {
            Float value = card.getBlueprint().getDestiny();
            return value == null ? 0f : value;
        }

        private void write(String test, String owner, PolicyTensors[] changed, Map<String, Object> facts) {
            long n = PAIRS.incrementAndGet();
            String side = game.getDarkPlayer().equals(player) ? "ds" : "ls";
            String id = "cf|" + n;
            synchronized (CounterfactualProbe.class) {
                writer.step(id, side, actor, original, 0, List.of(0), 0.0, 0, 1, null, "learner:0", maps);
                writer.step(id, side, changed[0], changed[1], 0, List.of(0), 0.0, 0, 1, null, "learner:0", maps);
                writer.finish(id, side, "finished", side, null, "cf", "cf");
                StringBuilder line = new StringBuilder();
                line.append("{\"id\":").append(StateEncoder.q(id)).append(",\"test\":").append(StateEncoder.q(test))
                    .append(",\"kind\":").append(StateEncoder.q(kind)).append(",\"turn\":").append(turn)
                    .append(",\"side\":").append(StateEncoder.q(side)).append(",\"changed_side\":")
                    .append(StateEncoder.q(owner.equals(player) ? "own" : "opponent"));
                for (Map.Entry<String, Object> fact : facts.entrySet()) {
                    Object value = fact.getValue();
                    String text = value instanceof Float && ((Float) value).isNaN() ? "null" : String.valueOf(value);
                    line.append(",\"").append(fact.getKey()).append("\":")
                        .append(value instanceof String ? StateEncoder.q((String) value) : text);
                }
                pairs.println(line.append('}'));
            }
        }
    }

    static boolean same(PolicyTensors a, PolicyTensors b) {
        return Arrays.equals(a.tok32, b.tok32) && Arrays.equals(a.tok, b.tok) && Arrays.equals(a.tokf, b.tokf)
            && Arrays.equals(a.head, b.head) && Arrays.equals(a.kgrid, b.kgrid);
    }

    // ---------------------------------------------------------------- exact revert

    /** Both players' Reserve / Used / Lost lists (order), their cards' zones and owners, and the knowledge tracker. */
    private static final class Saved {
        private static final Field KNOWLEDGE = field(GameState.class, "_knowledge");
        private static final Map<Zone, Field> PILES = new HashMap<>();

        static {
            PILES.put(Zone.RESERVE_DECK, field(GameState.class, "_reserveDecks"));
            PILES.put(Zone.USED_PILE, field(GameState.class, "_usedPiles"));
            PILES.put(Zone.LOST_PILE, field(GameState.class, "_lostPiles"));
        }

        final GameKnowledge knowledge;
        final Map<String, List<PhysicalCard>> lists = new HashMap<>();
        final Map<PhysicalCard, Zone> zones = new HashMap<>();
        final Map<PhysicalCard, String> owners = new HashMap<>();

        Saved(GameState gs, String unused) {
            knowledge = gs.getKnowledge().copy();
            for (String p : gs.getPlayerOrder().getAllPlayers()) {
                for (Map.Entry<Zone, Field> pile : PILES.entrySet()) {
                    List<PhysicalCard> list = list(gs, pile.getValue(), p);
                    lists.put(p + "|" + pile.getKey(), new ArrayList<>(list));
                    for (PhysicalCard card : list) {
                        zones.put(card, card.getZone());
                        owners.put(card, card.getZoneOwner());
                    }
                }
            }
        }

        void restore(GameState gs) {
            for (String p : gs.getPlayerOrder().getAllPlayers()) {
                for (Map.Entry<Zone, Field> pile : PILES.entrySet()) {
                    List<PhysicalCard> list = list(gs, pile.getValue(), p);
                    list.clear();
                    list.addAll(lists.get(p + "|" + pile.getKey()));
                }
            }
            for (Map.Entry<PhysicalCard, Zone> zone : zones.entrySet()) {
                zone.getKey().setZone(zone.getValue());
                zone.getKey().setZoneOwner(owners.get(zone.getKey()));
            }
            try {
                KNOWLEDGE.set(gs, knowledge.copy());
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        }

        @SuppressWarnings("unchecked")
        private static List<PhysicalCard> list(GameState gs, Field field, String player) {
            try {
                return ((Map<String, List<PhysicalCard>>) field.get(gs)).get(player);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        }

        private static Field field(Class<?> type, String name) {
            try {
                Field f = type.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException e) {
                throw new IllegalStateException(e);
            }
        }
    }
}
