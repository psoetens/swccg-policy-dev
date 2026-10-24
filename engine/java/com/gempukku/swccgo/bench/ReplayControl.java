package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.knowledge.GameKnowledge;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Plan critic-followups, BL-49 Stage A: record a game's answers, and replay a game from its shuffle seed and recorded
 * answers to a decision, force one option there, then let the AIs play on (option rollouts).
 *
 * <p>Record: {@code -Dreplay.record=DIR} writes DIR/GAMEID.tsv, one line per decision: index, player, decision type,
 * decision text (tabs and newlines replaced), answer.
 *
 * <p>Replay: {@code -Dreplay.jobs=FILE}, line g (0-based) for game g: {@code seed<TAB>answers-file<TAB>K<TAB>option
 * <TAB>verify}. The game's shuffles are seeded with {@code seed} (not the run's seed + g). Decisions before K take the
 * recorded answer; at K, option {@code option} of the encoder's answer list ({@code -1} = the recorded answer); after K
 * the AIs decide, except for the next {@code verify} decisions, which again take the recorded answers (the
 * exact-replay test). Each replayed decision's player, type and text are compared with the record; the first mismatch
 * of each game is reported to {@code -Dreplay.report=FILE} (and the game continues with its AIs). At K the report also
 * gets the option count and the forced answer.
 *
 * <p>Outcome states ({@code -Dreplay.states=DIR}, needs {@code policy.binaryTrajectory} and {@code policy.captureCritic}):
 * the first policy decision of the branching seat after K is written as a one-step pseudo-game {@code rb|g} (actor and
 * critic tensors) with the binary trajectory writer into DIR/traj, and described in DIR/states.jsonl (game, decision
 * index, side). Its critic value is the rollout's one-step outcome of the forced option.
 *
 * <p>Position states (plan critic-noise, BL-52 position selection): record mode with {@code -Dreplay.states=DIR} (same
 * needs) also writes the first policy encoding of every recorded decision with 2 to {@code replay.states.maxOptions}
 * (default 6) options as a one-step pseudo-game {@code rs|g|index} (actor and critic tensors at the decision: the
 * policy's option probabilities in the encoder's answer order, which a replay's forced option indexes, and the
 * deciding seat's critic value), described in DIR/states.jsonl (game, decision index, side, options, decision type,
 * min / max picks, the seat's turn number, whether it is the seat's own turn). Off unless both are set; the game
 * itself is unchanged.
 *
 * <p>{@code -Dreplay.offset=N}: game g of this run is job N + g (a long job list run in chunks of fresh JVMs; reports
 * and state ids use the global job index).
 *
 * <p>Rollout trajectories (plan turn-credit stage A; {@code -Dreplay.trace=DIR}, replay mode, needs
 * {@code policy.binaryTrajectory} and {@code policy.captureCritic}): the branching seat's decisions of each job as ONE
 * multi-step pseudo-game {@code rt|g} (g = the global job index) in DIR/traj: step 0 = decision K itself (its encoding
 * before the forced answer; seq = the forced option), then the first policy encoding (2+ options) of every later
 * decision of that seat until the game ends (actor and critic tensors; head[4] = the seat's turn number, head[7] = its
 * my-turn flag). The game's outcome finishes it, so its reward is +1 / -1 / 0 from the branching seat's view; a game
 * with a replay mismatch or an exception is dropped. DIR/trace.jsonl gets one line per traced job at its end (id, game,
 * K, side, option, forced answer index, steps, outcome, winner, decisions, mismatch). Off unless set; independent of
 * {@code replay.states}.
 *
 * <p>Plan critic-ceiling: job option {@code -2} = the AI answers at K (the position's value under the policy; {@code -1}
 * remains the recorded answer). {@code -Dreplay.reshuffle=self} reshuffles the branching seat's Reserve Deck and Force
 * Pile right before decision K is answered, {@code both} those of both seats (per-rollout seed: the job's seed mixed
 * with the global job index; the game thread's shuffle generator is reseeded with it). The report gets one
 * {@code reshuffle} line per job: for both seats and the piles Reserve Deck, Force Pile, Used Pile, Lost Pile and Hand,
 * {@code owner:pile:size:same-cards:same-order:known} (known = positions the branching seat knew in that pile before the
 * reshuffle). Off unless set; without it nothing changes.
 *
 * <p>Plan probe-economics stage 3a: {@code -Dreplay.timing=true} (replay mode) adds one {@code timing} report line per
 * job at its end: the wall milliseconds from the game's first decision to decision K (the prefix replay), to the
 * branching seat's first decision of its 1st, 2nd and 3rd own turn after K's turn (-1 if the game ended first) and to
 * the game end, then the number of decisions. Off unless set.
 *
 * <p>Plan probe-economics stage 3b (feasibility bench): {@code -Dreplay.branch=N} (replay mode, decision K must be a
 * phase action-or-pass decision, where Gemp takes its revert snapshots) plays every option of K N times for one round:
 * snapshots are switched on for K's answer only, the playout runs until the branching seat's next own turn starts,
 * then the snapshot is restored, Gemp asks K again, and the next playout answers it with the next option. Report
 * lines: {@code branch-restore} (the re-asked K's state equals the original: piles, hands, every card's zone, phase,
 * turn numbers and the option list) and {@code branch-playout} (option, wall milliseconds of the playout and of the
 * restore until K is asked again). The game then continues to its end with the AIs. Off unless set.
 * Finding (2026-10-21, 20 / 20 jobs): Gemp's revert snapshot resumes the phase's action loop with the NEXT player in
 * play order, as if K's action had been taken, so the restored game is not at K ({@code branch-fail}). A faithful
 * restore would need re-entry into the turn procedure; kept as the record of that test.
 */
public final class ReplayControl {
    private static final String RECORD = System.getProperty("replay.record");
    private static final String JOBS = System.getProperty("replay.jobs");
    private static final PrintWriter REPORT = open(System.getProperty("replay.report"));
    private static final List<String[]> JOB_LINES = loadJobs(JOBS);
    private static final Map<String, List<String>> RECORDED = new ConcurrentHashMap<>();
    private static final Map<String, List<String[]>> SOURCES = new ConcurrentHashMap<>();
    private static final Map<String, Boolean> MISMATCHED = new ConcurrentHashMap<>();
    private static final String STATES = System.getProperty("replay.states");
    private static final int OFFSET = Integer.getInteger("replay.offset", 0);
    private static final int MAX_OPTIONS = Integer.getInteger("replay.states.maxOptions", 6);
    private static final ThreadLocal<Object[]> CURRENT = new ThreadLocal<>();  // {gameId, index, player}
    private static final Map<String, String> BRANCH_PLAYER = new ConcurrentHashMap<>();
    private static final Map<String, Boolean> WRITTEN = new ConcurrentHashMap<>();
    private static final String TRACE = System.getProperty("replay.trace");
    private static final String RESHUFFLE = System.getProperty("replay.reshuffle", "");
    private static final boolean TIMING = Boolean.getBoolean("replay.timing");
    private static final int BRANCH = Integer.getInteger("replay.branch", 0);
    private static final Map<String, Branch> BRANCHES = new ConcurrentHashMap<>();

    /** Plan probe-economics 3b: the playout loop of one job (game thread). */
    private static final class Branch {
        final String player, text, stateKey;
        final List<String> answers;
        final int turnAtK, snapshotsBefore;
        int cursor, snapshotId = -1;
        boolean awaitSnapshot = true, awaitRedo;
        long playoutStart, restoreStart;

        Branch(String player, String text, String stateKey, List<String> answers, int turnAtK, int snapshotsBefore) {
            this.player = player;
            this.text = text;
            this.stateKey = stateKey;
            this.answers = answers;
            this.turnAtK = turnAtK;
            this.snapshotsBefore = snapshotsBefore;
        }

        int total() {
            return answers.size() * BRANCH;
        }
    }
    private static final Map<String, long[]> TIMES = new ConcurrentHashMap<>();  // {t0, tK, r1, r2, r3, turnAtK}
    private static final Zone[] PILES = {Zone.RESERVE_DECK, Zone.FORCE_PILE, Zone.USED_PILE, Zone.LOST_PILE, Zone.HAND};
    private static final Map<String, Trace> TRACES = new ConcurrentHashMap<>();
    private static PolicyTensors.Maps maps;
    private static BinaryTrajectoryWriter writer;
    private static PrintWriter states;
    private static BinaryTrajectoryWriter traceWriter;
    private static PrintWriter traceLog;

    /** One job's rollout trajectory (plan turn-credit): the branching seat's steps from K on. Game-thread confined. */
    private static final class Trace {
        final String id, player, side;
        final int k, option;
        int forced = -1, steps, lastIndex = -1;

        Trace(String id, String player, String side, int k, int option) {
            this.id = id;
            this.player = player;
            this.side = side;
            this.k = k;
            this.option = option;
        }
    }

    private ReplayControl() {
    }

    public static boolean active() {
        return RECORD != null || JOBS != null;
    }

    private static PrintWriter open(String path) {
        if (path == null || path.isBlank()) return null;
        try {
            return new PrintWriter(Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static List<String[]> loadJobs(String path) {
        List<String[]> out = new ArrayList<>();
        if (path == null || path.isBlank()) return out;
        try {
            for (String line : Files.readAllLines(Paths.get(path))) {
                if (!line.isBlank()) out.add(line.split("\t"));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out;
    }

    /** The shuffle seed of game g: the job's source seed in replay mode, else the run's. */
    public static long seed(int g, long runSeed) {
        g += OFFSET;
        if (g < JOB_LINES.size()) return Long.parseLong(JOB_LINES.get(g)[0]);
        return runSeed;
    }

    private static int gameIndex(String gameId) {
        return OFFSET + Integer.parseInt(gameId.substring(gameId.lastIndexOf('-') + 1));
    }

    private static String clean(String text) {
        return text == null ? "" : text.replace('\t', ' ').replace('\n', ' ');
    }

    private static List<String[]> source(String path) {
        return SOURCES.computeIfAbsent(path, p -> {
            List<String[]> rows = new ArrayList<>();
            try {
                for (String line : Files.readAllLines(Paths.get(p))) rows.add(line.split("\t", -1));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            return rows;
        });
    }

    private static synchronized void report(String line) {
        if (REPORT != null) {
            REPORT.println(line);
            REPORT.flush();
        }
    }

    /** The answer of decision {@code index} of game {@code gameId}: recorded, forced or the AI's. */
    public static String decide(SwccgGame game, String gameId, int index, String player, AwaitingDecision decision,
                                Supplier<String> ai) {
        String type = String.valueOf(decision.getDecisionType()), text = clean(decision.getText());
        CURRENT.set(new Object[]{gameId, index, player});
        if (TIMING && JOBS != null) time(game, gameId, index, player);
        if (BRANCH > 0 && JOBS != null) {
            String branched = branchStep(game, gameId, player, decision, text, ai);
            if (branched != null) return record(gameId, index, player, type, text, branched);
        }
        String answer;
        if (JOBS != null && gameIndex(gameId) < JOB_LINES.size()) {
            String[] job = JOB_LINES.get(gameIndex(gameId));
            List<String[]> rows = source(job[1]);
            int k = Integer.parseInt(job[2]), option = Integer.parseInt(job[3]), verify = Integer.parseInt(job[4]);
            boolean scripted = index < k || (index > k && index <= k + verify);
            String[] row = index < rows.size() ? rows.get(index) : null;
            if ((scripted || index == k) && !MISMATCHED.containsKey(gameId)) {
                if (row == null || !row[1].equals(player) || !row[2].equals(type) || !row[3].equals(text)) {
                    MISMATCHED.put(gameId, true);
                    report("job-" + gameIndex(gameId) + "\tmismatch\t" + index + "\t" + player + "|" + type + "|" + text + "\t"
                            + (row == null ? "end of record" : row[1] + "|" + row[2] + "|" + row[3]));
                }
            }
            if (MISMATCHED.containsKey(gameId)) {
                answer = ai.get();
            } else if (index == k) {
                BRANCH_PLAYER.put(gameId, player);
                if (!RESHUFFLE.isEmpty()) reshuffle(game, gameId, player, Long.parseLong(job[0]));
                Trace trace = null;
                if (TRACE != null) {  // before the encoding below: observe() writes it as the trace's step 0
                    trace = new Trace("rt|" + gameIndex(gameId), player,
                            game.getDarkPlayer().equals(player) ? "ds" : "ls", k, option);
                    TRACES.put(gameId, trace);
                }
                List<String> answers = StateEncoder.encode(game, player, decision).answers;
                if (option == -2) answer = ai.get();
                else answer = option >= 0 && option < answers.size() ? answers.get(option) : row[4];
                if (trace != null) trace.forced = answers.indexOf(answer);
                report("job-" + gameIndex(gameId) + "\tbranch\t" + index + "\t" + answers.size() + "\t" + option + "\t" + answer
                        + "\t" + answers.indexOf(row[4]));
            } else if (scripted) {
                answer = row[4];
            } else {
                answer = ai.get();
            }
        } else {
            answer = ai.get();
        }
        return record(gameId, index, player, type, text, answer);
    }

    private static String record(String gameId, int index, String player, String type, String text, String answer) {
        if (RECORD != null) {
            RECORDED.computeIfAbsent(gameId, id -> new ArrayList<>())
                    .add(index + "\t" + player + "\t" + type + "\t" + text + "\t" + clean(answer));
        }
        return answer;
    }

    /** Plan probe-economics 3b: the state compared after a restore (piles, hands, every card's zone, phase, turns). */
    private static String stateKey(SwccgGame game) {
        com.gempukku.swccgo.game.state.GameState state = game.getGameState();
        StringBuilder key = new StringBuilder(String.valueOf(state.getCurrentPhase())).append('|')
                .append(state.getCurrentPlayerId());
        for (String owner : List.of(game.getDarkPlayer(), game.getLightPlayer())) {
            key.append('|').append(state.getPlayersLatestTurnNumber(owner));
            for (Zone zone : PILES) key.append('|').append(ids(pile(state, owner, zone)));
        }
        List<String> zones = new ArrayList<>();
        for (PhysicalCard card : state.getAllPermanentCards())
            zones.add(card.getPermanentCardId() + ":" + card.getZone() + ":" + card.getZoneOwner());
        java.util.Collections.sort(zones);
        return key.append('|').append(String.join(",", zones)).toString();
    }

    /** Plan probe-economics 3b: answers K and the decisions of the playouts; null = not a branch decision. */
    private static String branchStep(SwccgGame game, String gameId, String player, AwaitingDecision decision,
                                     String text, Supplier<String> ai) {
        int g = gameIndex(gameId);
        if (g >= JOB_LINES.size() || MISMATCHED.containsKey(gameId)) return null;
        com.gempukku.swccgo.logic.timing.DefaultSwccgGame dg = (com.gempukku.swccgo.logic.timing.DefaultSwccgGame) game;
        Branch b = BRANCHES.get(gameId);
        long now = System.nanoTime();
        if (b == null) {
            Object[] current = CURRENT.get();
            int index = (Integer) current[1], k = Integer.parseInt(JOB_LINES.get(g)[2]);
            if (index != k) return null;
            if (!text.startsWith("Choose ") || !text.endsWith(" or Pass")) {
                report("job-" + g + "\tbranch-skip\tnot an action-or-pass decision: " + text);
                return null;
            }
            BRANCH_PLAYER.put(gameId, player);
            List<String> answers = StateEncoder.encode(game, player, decision).answers;
            b = new Branch(player, text, stateKey(game), answers, game.getGameState().getPlayersLatestTurnNumber(player),
                    game.getSnapshots().size());
            BRANCHES.put(gameId, b);
            dg.setSnapshotsEnabled(true);
            b.playoutStart = now;
            return answers.get(0);
        }
        if (b.cursor >= b.total()) return null;  // all playouts done: the game goes on with the AIs
        if (b.awaitSnapshot) {
            List<com.gempukku.swccgo.logic.timing.GameSnapshot> snapshots = game.getSnapshots();
            for (int i = snapshots.size() - 1; i >= 0; i--) {
                if (snapshots.get(i).getDescription().startsWith(b.player + ":")) {
                    b.snapshotId = snapshots.get(i).getId();
                    break;
                }
            }
            if (b.snapshotId >= 0) {
                b.awaitSnapshot = false;
                dg.setSnapshotsEnabled(false);
            } else if (!player.equals(b.player)) {
                report("job-" + g + "\tbranch-fail\tno snapshot after K's answer");
                b.cursor = b.total();
                dg.setSnapshotsEnabled(false);
                return null;
            }
        }
        if (b.awaitRedo) {
            if (!player.equals(b.player) || !text.equals(b.text)) {
                report("job-" + g + "\tbranch-fail\tafter the restore the first decision is not K: " + player + "|" + text
                        + " (K: " + b.player + "|" + b.text + ", restore pending " + game.isRestoreSnapshotPending()
                        + ", turn " + game.getGameState().getPlayersLatestTurnNumber(player) + " vs " + b.turnAtK + ")");
                b.cursor = b.total();
                return null;
            }
            List<String> answers = StateEncoder.encode(game, player, decision).answers;
            boolean same = stateKey(game).equals(b.stateKey) && answers.equals(b.answers);
            report("job-" + g + "\tbranch-restore\t" + b.cursor + "\t" + (same ? "same" : "different") + "\t"
                    + (now - b.restoreStart) / 1_000_000);
            b.awaitRedo = false;
            b.awaitSnapshot = true;
            b.snapshotId = -1;
            dg.setSnapshotsEnabled(true);
            b.playoutStart = now;
            return b.answers.get(b.cursor / BRANCH);
        }
        boolean horizon = !b.awaitSnapshot && player.equals(b.player) && player.equals(game.getGameState().getCurrentPlayerId())
                && game.getGameState().getPlayersLatestTurnNumber(player) > b.turnAtK;
        if (!horizon) return null;
        report("job-" + g + "\tbranch-playout\t" + b.cursor + "\t" + (b.cursor / BRANCH) + "\t"
                + (now - b.playoutStart) / 1_000_000);
        b.cursor++;
        if (b.cursor < b.total()) {
            game.requestRestoreSnapshot(b.snapshotId);
            b.awaitRedo = true;
            b.restoreStart = now;
        }
        return null;  // this decision is answered by the AI; the restore follows
    }

    /** Plan probe-economics stage 3a: the prefix and round boundaries of a job (game thread). */
    private static void time(SwccgGame game, String gameId, int index, String player) {
        long now = System.nanoTime();
        long[] t = TIMES.computeIfAbsent(gameId, id -> new long[]{now, -1, -1, -1, -1, -1});
        int g = gameIndex(gameId);
        if (g >= JOB_LINES.size()) return;
        int k = Integer.parseInt(JOB_LINES.get(g)[2]);
        if (index == k) {
            t[1] = now;
            t[5] = game.getGameState().getPlayersLatestTurnNumber(player);
        } else if (index > k && t[5] >= 0 && player.equals(BRANCH_PLAYER.get(gameId))
                && player.equals(game.getGameState().getCurrentPlayerId())) {
            int rounds = game.getGameState().getPlayersLatestTurnNumber(player) - (int) t[5];
            for (int r = 1; r <= Math.min(rounds, 3); r++) if (t[1 + r] < 0) t[1 + r] = now;
        }
    }

    /** Plan critic-ceiling: reshuffle the named piles at K and report what changed (see the class comment). */
    private static void reshuffle(SwccgGame game, String gameId, String player, long jobSeed) {
        if (!RESHUFFLE.equals("self") && !RESHUFFLE.equals("both"))
            throw new IllegalArgumentException("replay.reshuffle must be self or both: " + RESHUFFLE);
        com.gempukku.swccgo.game.state.GameState state = game.getGameState();
        String opponent = game.getDarkPlayer().equals(player) ? game.getLightPlayer() : game.getDarkPlayer();
        List<String> owners = List.of(player, opponent);
        List<List<Integer>> before = new ArrayList<>();
        List<Integer> known = new ArrayList<>();
        GameKnowledge knowledge = state.getKnowledge();
        for (String owner : owners) {
            for (Zone zone : PILES) {
                before.add(ids(pile(state, owner, zone)));
                int count = 0;
                if (knowledge != null && zone != Zone.HAND) {
                    for (String seen : knowledge.views[knowledge.player(player)][knowledge.player(owner)].order
                            .get(GameKnowledge.cell(zone)))
                        if (seen != null) count++;
                }
                known.add(count);
            }
        }
        CorpusRandom.seed(jobSeed * 0x9E3779B97F4A7C15L + gameIndex(gameId) + 1);
        for (String owner : RESHUFFLE.equals("both") ? owners : List.of(player)) {
            state.shufflePile(owner, Zone.RESERVE_DECK);
            state.shufflePile(owner, Zone.FORCE_PILE);
        }
        StringBuilder line = new StringBuilder("job-" + gameIndex(gameId) + "\treshuffle\t" + RESHUFFLE + "\t");
        int i = 0;
        for (int o = 0; o < owners.size(); o++) {
            for (Zone zone : PILES) {
                List<Integer> after = ids(pile(state, owners.get(o), zone)), was = before.get(i);
                List<Integer> sortedAfter = new ArrayList<>(after), sortedWas = new ArrayList<>(was);
                java.util.Collections.sort(sortedAfter);
                java.util.Collections.sort(sortedWas);
                line.append(i == 0 ? "" : " ").append(o == 0 ? "self" : "opp").append(':').append(zone).append(':')
                        .append(after.size()).append(':').append(sortedAfter.equals(sortedWas) ? 1 : 0).append(':')
                        .append(after.equals(was) ? 1 : 0).append(':').append(known.get(i));
                i++;
            }
        }
        report(line.toString());
    }

    private static List<PhysicalCard> pile(com.gempukku.swccgo.game.state.GameState state, String owner, Zone zone) {
        return zone == Zone.HAND ? state.getHand(owner) : state.getCardPile(owner, zone, false);
    }

    private static List<Integer> ids(List<PhysicalCard> cards) {
        List<Integer> out = new ArrayList<>();
        for (PhysicalCard card : cards) out.add(card.getCardId());
        return out;
    }

    public static boolean observing() {
        return STATES != null && (JOBS != null || RECORD != null) || TRACE != null && JOBS != null;
    }

    private static synchronized void loadMaps() throws IOException {
        if (maps == null) maps = PolicyTensors.Maps.load(Paths.get(System.getProperty("policy.native.bundle")));
    }

    /** A trajectory writer into DIR/traj that does not become the active one (Dojo rewards keep their writer). */
    private static BinaryTrajectoryWriter quietWriter(Path dir, int gamesPerFile) throws ReflectiveOperationException {
        java.lang.reflect.Field active = BinaryTrajectoryWriter.class.getDeclaredField("active");
        active.setAccessible(true);
        Object previous = active.get(null);
        BinaryTrajectoryWriter created = new BinaryTrajectoryWriter(dir, gamesPerFile);
        active.set(null, previous);
        return created;
    }

    private static synchronized void openTrace() {
        if (traceWriter != null) return;
        try {
            loadMaps();
            Path dir = Paths.get(TRACE);
            Files.createDirectories(dir);
            traceWriter = quietWriter(dir, 10);  // multi-step pseudo-games: small files keep the reader's memory low
            traceLog = new PrintWriter(Files.newBufferedWriter(dir.resolve("trace.jsonl"), StandardCharsets.UTF_8), true);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    traceWriter.close();
                } catch (InterruptedException ignored) {
                }
                synchronized (ReplayControl.class) {
                    traceLog.close();
                }
            }));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    private static synchronized void openStates() {
        if (writer != null) return;
        try {
            loadMaps();
            Path dir = Paths.get(STATES);
            Files.createDirectories(dir);
            writer = quietWriter(dir, 50);
            states = new PrintWriter(Files.newBufferedWriter(dir.resolve("states.jsonl"), StandardCharsets.UTF_8), true);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    writer.close();
                } catch (InterruptedException ignored) {
                }
                synchronized (ReplayControl.class) {
                    states.close();
                }
            }));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Called by KnowledgeStateEncoder after a policy encoding: writes the branching seat's first state after K. */
    static void observe(SwccgGame game, String player, StateEncoder.Encoded encoded) {
        Object[] current = CURRENT.get();
        if (current == null || encoded.binary == null || encoded.binary.privileged == null) return;
        String gameId = (String) current[0];
        int index = (Integer) current[1];
        if (JOBS == null) {
            if (STATES != null) observeRecorded(game, player, encoded, gameId, index, current);
            return;
        }
        if (TRACE != null) observeTrace(player, encoded, gameId, index, current);
        if (STATES == null) return;
        if (!player.equals(current[2]) || !player.equals(BRANCH_PLAYER.get(gameId)) || WRITTEN.containsKey(gameId)) return;
        int g = gameIndex(gameId);
        if (g >= JOB_LINES.size() || index <= Integer.parseInt(JOB_LINES.get(g)[2])) return;
        if (MISMATCHED.containsKey(gameId)) return;
        WRITTEN.put(gameId, true);
        openStates();
        PolicyTensors critic = encoded.binary.critic(encoded, maps);
        PolicyTensors actor = encoded.binary.actor(encoded, maps);
        String side = game.getDarkPlayer().equals(player) ? "ds" : "ls";
        String id = "rb|" + g;
        synchronized (ReplayControl.class) {
            writer.step(id, side, actor, critic, 0, List.of(0), 0.0, 0, 1, null, "learner:0", maps);
            writer.finish(id, side, "finished", side, null, "rb", "rb");
            states.println("{\"id\":" + StateEncoder.q(id) + ",\"game\":" + g + ",\"index\":" + index
                    + ",\"side\":" + StateEncoder.q(side) + "}");
        }
    }

    /** Replay mode with replay.trace: one step of the branching seat's trajectory (decision K, then its first policy
     * encoding with 2+ options of every later decision). */
    private static void observeTrace(String player, StateEncoder.Encoded encoded, String gameId, int index,
                                     Object[] current) {
        Trace trace = TRACES.get(gameId);
        if (trace == null || !player.equals(current[2]) || !player.equals(trace.player) || index < trace.k
                || index <= trace.lastIndex || encoded.answers.size() < 2 || MISMATCHED.containsKey(gameId)) return;
        trace.lastIndex = index;
        openTrace();
        PolicyTensors critic = encoded.binary.critic(encoded, maps);
        PolicyTensors actor = encoded.binary.actor(encoded, maps);
        int chosen = index == trace.k && trace.option >= 0 && trace.option < actor.options ? trace.option : 0;
        traceWriter.step(trace.id, trace.side, actor, critic, 0, List.of(chosen), 0.0, 0, 1, null, "learner:0", maps);
        trace.steps++;
    }

    /** Record mode: the first policy encoding of a recorded decision with 2..MAX_OPTIONS options (position states). */
    private static void observeRecorded(SwccgGame game, String player, StateEncoder.Encoded encoded, String gameId,
                                        int index, Object[] current) {
        int options = encoded.answers.size();
        if (!player.equals(current[2]) || options < 2 || options > MAX_OPTIONS) return;
        if (WRITTEN.putIfAbsent(gameId + "|" + index, true) != null) return;
        openStates();
        PolicyTensors critic = encoded.binary.critic(encoded, maps);
        PolicyTensors actor = encoded.binary.actor(encoded, maps);
        String side = game.getDarkPlayer().equals(player) ? "ds" : "ls";
        int g = gameIndex(gameId);
        String id = "rs|" + g + "|" + index;
        int turn = game.getGameState().getPlayersLatestTurnNumber(player);
        boolean own = player.equals(game.getGameState().getCurrentPlayerId());
        synchronized (ReplayControl.class) {
            writer.step(id, side, actor, critic, 0, List.of(0), 0.0, 0, 1, null, "learner:0", maps);
            writer.finish(id, side, "finished", side, null, "rb", "rb");
            states.println("{\"id\":" + StateEncoder.q(id) + ",\"game\":" + g + ",\"index\":" + index
                    + ",\"side\":" + StateEncoder.q(side) + ",\"options\":" + options + ",\"type\":"
                    + StateEncoder.q(String.valueOf(encoded.type)) + ",\"min\":" + encoded.min + ",\"max\":" + encoded.max
                    + ",\"turn\":" + turn + ",\"own\":" + own + "}");
        }
    }

    /** Game end: write the record, report a replayed game's outcome. */
    public static void finish(String gameId, String outcome, String winner, int decisions) {
        if (RECORD != null) {
            List<String> lines = RECORDED.remove(gameId);
            try {
                Path dir = Paths.get(RECORD);
                Files.createDirectories(dir);
                Files.write(dir.resolve(gameId + ".tsv"), lines == null ? List.of() : lines, StandardCharsets.UTF_8);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        long[] t = TIMING ? TIMES.remove(gameId) : null;
        if (t != null && JOBS != null && gameIndex(gameId) < JOB_LINES.size()) {
            long end = System.nanoTime();
            StringBuilder line = new StringBuilder("job-" + gameIndex(gameId) + "\ttiming");
            for (int i = 1; i <= 4; i++) line.append('\t').append(t[i] < 0 ? -1 : (t[i] - t[0]) / 1_000_000);
            line.append('\t').append((end - t[0]) / 1_000_000).append('\t').append(decisions);
            report(line.toString());
        }
        if (JOBS != null && gameIndex(gameId) < JOB_LINES.size()) {
            report("job-" + gameIndex(gameId) + "\tend\t" + decisions + "\t" + outcome + "\t" + winner + "\t"
                    + (MISMATCHED.containsKey(gameId) ? "mismatch" : "exact"));
        }
        Trace trace = TRACE == null ? null : TRACES.remove(gameId);
        if (trace != null && trace.steps > 0) {
            boolean mismatch = MISMATCHED.containsKey(gameId);
            // The writer drops an "exception" pseudo-game: a mismatched replay or a crashed game has no usable result.
            traceWriter.finish(trace.id, trace.side, mismatch ? "exception" : outcome, winner == null ? "" : winner, null,
                    "rt", "rt");
            synchronized (ReplayControl.class) {
                traceLog.println("{\"id\":" + StateEncoder.q(trace.id) + ",\"game\":" + gameIndex(gameId) + ",\"index\":"
                        + trace.k + ",\"side\":" + StateEncoder.q(trace.side) + ",\"option\":" + trace.option
                        + ",\"forced\":" + trace.forced + ",\"steps\":" + trace.steps + ",\"outcome\":"
                        + StateEncoder.q(outcome) + ",\"winner\":" + StateEncoder.q(winner == null ? "" : winner)
                        + ",\"decisions\":" + decisions + ",\"mismatch\":" + mismatch + "}");
            }
        }
    }
}
