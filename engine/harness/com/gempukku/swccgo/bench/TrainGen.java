package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.ai.SwccgAiController;
import com.gempukku.swccgo.ai.models.AdvancedAi;
import com.gempukku.swccgo.ai.models.BeginnerAi;
import com.gempukku.swccgo.ai.models.rando.RandoCalAi;
import com.gempukku.swccgo.common.Zone;
import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;
import com.gempukku.swccgo.game.formats.SwccgoFormatLibrary;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.decisions.DecisionResultInvalidException;
import com.gempukku.swccgo.logic.timing.DefaultSwccgGame;
import com.gempukku.swccgo.logic.timing.DefaultUserFeedback;
import com.gempukku.swccgo.logic.vo.SwccgDeck;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPOutputStream;

/**
 * Training-data generation and play-strength runs for swccg-policy.
 *
 * Decks: --schedule FILE (one "dsPath<TAB>lsPath" per game), --pool FILE (lines "DS|LS<TAB>path"; a game picks a random DS and LS deck with Random(pairSeed, game index)) or
 * --matchups "ds.txt::ls.txt;..." (cycled). Relative paths resolve against --decks DIR.
 * AIs: --agent and --opponent (beginner|advanced|random|policy|policy:KEY), --agentSide ds|ls|alternate (alternate: even games agent plays DS).
 *   policy:KEY speaks the RL envelope protocol (swccg_policy.rl) with model KEY; --opponent may be a weighted league
 *   "policy:learner=0.4,policy:bc=0.2,advanced=0.2" (one opponent per game, chosen with Random(pairSeed, game index)).
 * --stackLog FILE: full stack trace of every exception that ends a game (with game id and decks).
 * Export: --export DIR writes gzipped JSONL shards (StateEncoder format) of every decision with >= 2 options, both sides.
 * Other: --games, --threads, --warmup, --cap (decisions), --maxGameMs, --server host:port, --out CSV, --traceDump FILE,
 * --seed (single thread only: seeded shuffles).
 */
public class TrainGen {
    private static final String DS = "ds", LS = "ls";

    /** True when every card of the deck (inside and outside) is from a Decipher set (blueprint set number below 200). */
    static boolean decipherOnly(SwccgDeck deck) {
        List<String> cards = new ArrayList<>(deck.getCards());
        cards.addAll(deck.getCardsOutsideDeck());
        for (String id : cards) {
            String set = id.split("_", 2)[0].replaceAll("[^0-9]", "");
            if (set.isEmpty() || Integer.parseInt(set) >= 200) return false;
        }
        return true;
    }
    // -Dtraingen.invalidLog=FILE: one line per rejected answer (game, decision, player, type, text, answer, warning, parameters)
    private static final PrintWriter INVALID_LOG = openLog(System.getProperty("traingen.invalidLog"));
    // Plan decision-cap-fixes F2: a game whose decisions repeat one cycle of 1-4 (player, type, text) LOOP_REPEATS times
    // within a turn ends as "loop" (instead of running to the decision cap). On by default; -Dtraingen.loopGuard=false.
    private static final boolean LOOP_GUARD = Boolean.parseBoolean(System.getProperty("traingen.loopGuard", "true"));
    // Plan v-cards-fixes: a game whose two decks hold only Decipher cards (sets < 200) plays Gemp's "Decipher Cards Only"
    // format, so name-a-card / name-a-card-type decisions list only Decipher titles; any virtual card: "open" (as before).
    private static final boolean FORMAT_BY_DECKS = Boolean.parseBoolean(System.getProperty("traingen.formatByDecks", "true"));
    private static final int LOOP_REPEATS = Integer.getInteger("traingen.loopRepeats", 200);

    /** Detects a decision sequence that repeats one short cycle within a turn. */
    static final class LoopGuard {
        private static final int MAX_PERIOD = 4;
        private final String[] last = new String[MAX_PERIOD];
        private final int[] run = new int[MAX_PERIOD + 1];
        private int size;
        private String turn;
        private int period;

        /** Adds a decision; true once some period p in 1..4 has repeated LOOP_REPEATS times in a row in this turn. */
        boolean add(String turnKey, String key) {
            if (!turnKey.equals(turn)) {
                turn = turnKey;
                size = 0;
                java.util.Arrays.fill(run, 0);
            }
            for (int p = 1; p <= MAX_PERIOD; p++) {
                run[p] = size >= p && key.equals(last[(size - p) % MAX_PERIOD]) ? run[p] + 1 : 0;
                if (run[p] >= (LOOP_REPEATS - 1) * p) period = p;
            }
            last[size % MAX_PERIOD] = key;
            size++;
            return period > 0;
        }

        /** The repeating cycle, oldest first. */
        String describe() {
            StringBuilder cycle = new StringBuilder("loop of " + period + ":");
            for (int i = period; i >= 1; i--) cycle.append(" [").append(last[(size - i) % MAX_PERIOD]).append(']');
            return cycle.toString();
        }
    }

    private static PrintWriter openLog(String path) {
        try {
            return path == null ? null : new PrintWriter(new java.io.FileWriter(path, StandardCharsets.UTF_8, true));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--")) continue;
            String k = args[i].substring(2);
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) opt.put(k, args[++i]);
            else opt.put(k, "true");
        }
        File deckDir = new File(opt.getOrDefault("decks", "decks"));
        int games = Integer.parseInt(opt.getOrDefault("games", "10"));
        int threads = Integer.parseInt(opt.getOrDefault("threads", "1"));
        int warmup = Integer.parseInt(opt.getOrDefault("warmup", "0"));
        // --gameOffset: global index of this run's first game (seeds, deck choice and agent side follow the global index),
        // so several single-threaded seeded JVMs can each play a slice of one reproducible game set.
        int gameOffset = Integer.parseInt(opt.getOrDefault("gameOffset", "0"));
        int cap = Integer.parseInt(opt.getOrDefault("cap", "6000"));
        long maxGameMs = Long.parseLong(opt.getOrDefault("maxGameMs", "600000"));
        long pairSeed = Long.parseLong(opt.getOrDefault("pairSeed", "1"));
        String agent = opt.getOrDefault("agent", "advanced"), opponent = opt.getOrDefault("opponent", "advanced");
        String agentSide = opt.getOrDefault("agentSide", "alternate");
        String server = opt.getOrDefault("server", "127.0.0.1:47000");
        String out = opt.getOrDefault("out", "traingen.csv");
        String exportDir = opt.get("export");
        Long seed = opt.containsKey("seed") ? Long.parseLong(opt.get("seed")) : null;
        if (seed != null && threads != 1) throw new IllegalArgumentException("--seed needs --threads 1");
        PrintWriter traceDump = opt.containsKey("traceDump") ? new PrintWriter(opt.get("traceDump"), StandardCharsets.UTF_8) : null;
        PrintWriter stackLog = opt.containsKey("stackLog") ? new PrintWriter(new FileWriter(opt.get("stackLog"), StandardCharsets.UTF_8, true), true) : null;

        SwccgCardBlueprintLibrary library = new SwccgCardBlueprintLibrary();
        SwccgoFormatLibrary formats = new SwccgoFormatLibrary(library);

        List<String> dsDecks = new ArrayList<>(), lsDecks = new ArrayList<>();
        List<String[]> matchups = new ArrayList<>();
        if (opt.containsKey("schedule")) {
            // One game per line: "dsPath<TAB>lsPath" (game index = line number); balanced pairings are built in Python.
            for (String line : Files.readAllLines(new File(opt.get("schedule")).toPath())) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] p = line.split("\t");
                matchups.add(new String[]{p[0].trim(), p[1].trim()});
            }
            if (!opt.containsKey("games")) games = matchups.size();
        } else if (opt.containsKey("pool")) {
            for (String line : Files.readAllLines(new File(opt.get("pool")).toPath())) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] p = line.split("\t");
                (p[0].trim().equalsIgnoreCase("DS") ? dsDecks : lsDecks).add(p[1].trim());
            }
        } else {
            for (String m : opt.getOrDefault("matchups", "").split(";")) {
                if (m.isBlank()) continue;
                String[] p = m.split("::");
                matchups.add(new String[]{p[0].trim(), p[1].trim()});
            }
        }
        Map<String, SwccgDeck> decks = new HashMap<>();
        for (String d : dsDecks) decks.computeIfAbsent(d, k -> DeckLoader.loadDeck(resolve(deckDir, k), library));
        for (String d : lsDecks) decks.computeIfAbsent(d, k -> DeckLoader.loadDeck(resolve(deckDir, k), library));
        for (String[] m : matchups) for (String d : m) decks.computeIfAbsent(d, k -> DeckLoader.loadDeck(resolve(deckDir, k), library));
        System.err.printf("decks loaded: %d (pool DS %d, LS %d, matchups %d)%n", decks.size(), dsDecks.size(), lsDecks.size(), matchups.size());

        if (exportDir != null) {
            new File(exportDir).mkdirs();
            try (PrintWriter zw = new PrintWriter(new File(exportDir, "zones.json"), StandardCharsets.UTF_8)) {
                zw.print("[");
                Zone[] z = Zone.values();
                for (int i = 0; i < z.length; i++)
                    zw.printf("%s{\"name\":\"%s\",\"public\":%b,\"visibleByOwner\":%b,\"inPlay\":%b,\"faceDown\":%b}", i == 0 ? "" : ",",
                            z[i].name(), z[i].isPublic(), z[i].isVisibleByOwner(), z[i].isInPlay(), z[i].isFaceDown());
                zw.printf(",{\"name\":\"OWN_UNSEEN\",\"public\":false,\"visibleByOwner\":true,\"inPlay\":false}]%n");
            }
        }

        Runner r = new Runner(library, formats, decks, dsDecks, lsDecks, matchups, pairSeed, agent, opponent, agentSide, server,
                cap, maxGameMs, exportDir, seed, traceDump, gameOffset);
        r.stackLog = stackLog;
        if (warmup > 0) r.run(warmup, threads, null, "warmup");
        try (PrintWriter pw = new PrintWriter(out, StandardCharsets.UTF_8)) {
            pw.println("game,thread,ds_deck,ls_deck,ds_ai,ls_ai,outcome,winner,decisions,exported,unmapped,invalid,turns,wall_ms,error");
            long t0 = System.nanoTime();
            r.run(games, threads, pw, opt.getOrDefault("tag", "run"));
            double secs = (System.nanoTime() - t0) / 1e9;
            System.err.printf("%d games on %d threads in %.1f s = %.2f games/s; exported %d decisions (%d unmapped); policy queries %d "
                            + "(%.2f ms avg), fallbacks %d, loop-guard %d, real-loops %d, rejected %d, decisions %d%n", games, threads, secs, games / secs, r.exported.get(), r.unmapped.get(),
                    PolicyAi.QUERIES.get(), PolicyAi.QUERIES.get() == 0 ? 0 : PolicyAi.QUERY_NANOS.get() / 1e6 / PolicyAi.QUERIES.get(),
                    PolicyAi.FALLBACKS.get(), PolicyAi.LOOP_GUARD.get(), PolicyAi.REAL_LOOPS.get(), PolicyAi.REJECTED.get(),
                    PolicyAi.DECISIONS.get());
            if (PolicyAi.CHECK_ENCODER)
                System.err.printf("encoder check: %d decisions, %d mismatches%n", PolicyAi.ENCODER_CHECKS.get(), PolicyAi.ENCODER_MISMATCHES.get());
            System.err.printf("harness: countFirst=%b encoderCache=%b checkEncoder=%b native=%b loopGuardK=%d%n", PolicyAi.COUNT_FIRST,
                    StateEncoder.QUERY_CACHE, PolicyAi.CHECK_ENCODER, System.getProperty("policy.native.run") != null, PolicyAi.LOOP_K);
            PolicyEngine eng = PolicyEngine.get();
            if (eng != null) {
                System.err.println(eng.stats());
                eng.close();
            }
            System.err.printf(Locale.ROOT, "policy latency ms: p50 %.3f p90 %.3f p99 %.3f p99.9 %.3f%n", PolicyAi.latencyQuantileMs(0.5),
                    PolicyAi.latencyQuantileMs(0.9), PolicyAi.latencyQuantileMs(0.99), PolicyAi.latencyQuantileMs(0.999));
        }
        if (traceDump != null) traceDump.close();
        if (stackLog != null) stackLog.close();
        System.exit(0);
    }

    static File resolve(File dir, String name) {
        File f = new File(name);
        return f.isAbsolute() ? f : new File(dir, name);
    }

    static final class Runner {
        final SwccgCardBlueprintLibrary library;
        final SwccgoFormatLibrary formats;
        final Map<String, SwccgDeck> decks;
        final List<String> dsDecks, lsDecks;
        final List<String[]> matchups;
        final long pairSeed;
        final String agent, opponent, agentSide, server;
        final int cap;
        final long maxGameMs;
        final String exportDir;
        final Long seed;
        final PrintWriter traceDump;
        final AtomicLong exported = new AtomicLong(), unmapped = new AtomicLong();
        final int gameOffset;
        PrintWriter stackLog;

        Runner(SwccgCardBlueprintLibrary library, SwccgoFormatLibrary formats, Map<String, SwccgDeck> decks, List<String> dsDecks,
               List<String> lsDecks, List<String[]> matchups, long pairSeed, String agent, String opponent, String agentSide,
               String server, int cap, long maxGameMs, String exportDir, Long seed, PrintWriter traceDump, int gameOffset) {
            this.gameOffset = gameOffset;
            this.library = library;
            this.formats = formats;
            this.decks = decks;
            this.dsDecks = dsDecks;
            this.lsDecks = lsDecks;
            this.matchups = matchups;
            this.pairSeed = pairSeed;
            this.agent = agent;
            this.opponent = opponent;
            this.agentSide = agentSide;
            this.server = server;
            this.cap = cap;
            this.maxGameMs = maxGameMs;
            this.exportDir = exportDir;
            this.seed = seed;
            this.traceDump = traceDump;
        }

        void run(int games, int threads, PrintWriter pw, String tag) throws InterruptedException {
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            AtomicInteger next = new AtomicInteger();
            for (int t = 0; t < threads; t++) {
                final int thread = t;
                pool.submit(() -> {
                    Writer shard = null;
                    try {
                        if (pw != null && exportDir != null)
                            shard = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(
                                    new File(exportDir, "shard-" + tag + "-t" + thread + ".jsonl.gz")), 1 << 16), StandardCharsets.UTF_8), 1 << 16);
                        int g;
                        while ((g = next.getAndIncrement()) < games) {
                            String row = playOne(tag, g + (pw != null ? gameOffset : 0), thread, pw != null ? shard : null);
                            if (pw != null) synchronized (pw) { pw.println(row); pw.flush(); }
                            if (pw != null && (g + 1) % 100 == 0) System.err.printf("  %d/%d games%n", g + 1, games);
                        }
                    } catch (Throwable e) {
                        e.printStackTrace();
                    } finally {
                        if (shard != null) try { shard.close(); } catch (IOException ignored) { }
                    }
                });
            }
            pool.shutdown();
            pool.awaitTermination(30, TimeUnit.DAYS);
        }

        SwccgAiController newAi(String kind) {
            switch (kind) {
                case "advanced": return new AdvancedAi();
                case "beginner": return new BeginnerAi();
                // Plan rando-sweep: GEMP's elite hall bot (no BotStatsDAO: no record lookups in its chat).
                case "rando": return new RandoCalAi();
                case "random": return new RandomAi();
                case "policy": {
                    String[] hp = server.split(":");
                    return new PolicyAi(hp[0], Integer.parseInt(hp[1]));
                }
                default:
                    if (kind.startsWith("policy:")) {
                        String[] hp = server.split(":");
                        return new PolicyAi(hp[0], Integer.parseInt(hp[1]), kind.substring("policy:".length()));
                    }
                    throw new IllegalArgumentException("unknown AI " + kind);
            }
        }

        /** Opponent for game g: a fixed kind, or a weighted league "kindA=wA,kindB=wB" drawn with Random(pairSeed, g). */
        String opponentFor(int g) {
            if (!opponent.contains("=")) return opponent;
            String[] parts = opponent.split(",");
            double total = 0;
            for (String part : parts) total += Double.parseDouble(part.substring(part.lastIndexOf('=') + 1));
            // SplittableRandom scrambles its seed; java.util.Random's first draw is correlated across consecutive seeds.
            double x = new SplittableRandom(pairSeed * 1_000_003L + g).nextDouble() * total;
            for (String part : parts) {
                int eq = part.lastIndexOf('=');
                x -= Double.parseDouble(part.substring(eq + 1));
                if (x <= 0) return part.substring(0, eq).trim();
            }
            String last = parts[parts.length - 1];
            return last.substring(0, last.lastIndexOf('=')).trim();
        }

        String playOne(String tag, int g, int thread, Writer shard) {
            String dsName, lsName;
            if (!dsDecks.isEmpty()) {
                Random rnd = new Random(pairSeed * 1_000_003L + g);
                dsName = dsDecks.get(rnd.nextInt(dsDecks.size()));
                lsName = lsDecks.get(rnd.nextInt(lsDecks.size()));
            } else {
                String[] m = matchups.get(g % matchups.size());
                dsName = m[0];
                lsName = m[1];
            }
            boolean agentDs = agentSide.equals("ds") || (agentSide.equals("alternate") && g % 2 == 0);
            String opp = opponentFor(g);
            String dsAi = agentDs ? agent : opp, lsAi = agentDs ? opp : agent;
            if (seed != null) seedShuffles(seed + g);
            Map<String, SwccgAiController> ais = new HashMap<>();
            ais.put(DS, newAi(dsAi));
            ais.put(LS, newAi(lsAi));
            String gameId = tag + "-" + g;
            for (SwccgAiController ai : ais.values()) if (ai instanceof PolicyAi) ((PolicyAi) ai).setGameId(gameId);
            String outcome = "finished", winner = "", error = "";
            int decisions = 0, exportedHere = 0, unmappedHere = 0, invalid = 0, turns = 0;
            long start = System.nanoTime();
            try {
                Map<String, SwccgDeck> dmap = new HashMap<>();
                dmap.put(DS, decks.get(dsName));
                dmap.put(LS, decks.get(lsName));
                Map<String, Integer> clocks = new HashMap<>();
                clocks.put(DS, 0);
                clocks.put(LS, 0);
                DefaultUserFeedback feedback = new DefaultUserFeedback();
                String format = FORMAT_BY_DECKS && decipherOnly(dmap.get(DS)) && decipherOnly(dmap.get(LS)) ? "open_no_virtual" : "open";
                DefaultSwccgGame game = new DefaultSwccgGame(formats.getFormat(format), dmap, feedback, library, clocks, false);
                feedback.setGame(game);
                game.setSnapshotsEnabled(false);
                game.startGame();
                int consecutiveInvalid = 0;
                LoopGuard guard = LOOP_GUARD ? new LoopGuard() : null;
                while (!game.isFinished()) {
                    Set<String> pending = new TreeSet<>(feedback.getUsersPendingDecision());
                    if (pending.isEmpty()) { outcome = "stuck"; break; }
                    if (decisions >= cap) { outcome = "cap"; break; }
                    if (System.nanoTime() - start > maxGameMs * 1_000_000L) { outcome = "timeout"; break; }
                    for (String p : pending) {
                        AwaitingDecision decision = feedback.getAwaitingDecision(p);
                        if (decision == null) continue;
                        if (guard != null && guard.add(game.getGameState().getPlayersLatestTurnNumber(DS) + "/"
                                + game.getGameState().getPlayersLatestTurnNumber(LS),
                                // Digits (card ids, values) masked: a card that moves gets a new id every cycle.
                                p + "|" + decision.getDecisionType() + "|" + decision.getText().replaceAll("[0-9]+", "#"))) {
                            outcome = "loop";
                            error = guard.describe();
                            if (stackLog != null) synchronized (stackLog) {
                                stackLog.printf("=== %s  ds=%s ls=%s ds_ai=%s ls_ai=%s decisions=%d LOOP %s%n", gameId, dsName, lsName,
                                    dsAi, lsAi, decisions, error);
                                stackLog.flush();
                            }
                            break;
                        }
                        SwccgAiController ai = ais.get(p);
                        ai.setGame(game);
                        StateEncoder.Encoded enc = shard != null ? StateEncoder.encode(game, p, decision) : null;
                        String answer = ai.decide(p, decision, game.getGameState());
                        if (enc != null && enc.answers.size() >= 2) {
                            List<Integer> ch = StateEncoder.chosen(enc, answer);
                            if (ch.isEmpty()) { unmappedHere++; unmapped.incrementAndGet(); }
                            shard.write(enc.line(gameId, decisions, ch));
                            shard.write('\n');
                            exportedHere++;
                            exported.incrementAndGet();
                            if (traceDump != null) synchronized (traceDump) {
                                traceDump.println(gameId + "\t" + decisions + "\t" + p + "\t" + enc.type + "\t" + answer);
                            }
                        }
                        try {
                            feedback.participantDecided(p);
                            decision.decisionMade(answer);
                            game.carryOutPendingActionsUntilDecisionNeeded();
                            consecutiveInvalid = 0;
                        } catch (DecisionResultInvalidException ex) {
                            invalid++;
                            if (INVALID_LOG != null) synchronized (INVALID_LOG) {
                                StringBuilder params = new StringBuilder();
                                for (Map.Entry<String, String[]> e : decision.getDecisionParameters().entrySet())
                                    params.append(e.getKey()).append('=').append(String.join(",", e.getValue())).append(' ');
                                INVALID_LOG.println(gameId + "\t" + decisions + "\t" + p + "\t" + decision.getDecisionType() + "\t"
                                        + decision.getText() + "\t" + answer + "\t" + ex.getWarningMessage() + "\t" + params);
                                INVALID_LOG.flush();
                            }
                            feedback.sendAwaitingDecision(p, decision);
                            if (++consecutiveInvalid > 20) { outcome = "invalid_loop"; error = ex.getMessage(); }
                        }
                        if (!outcome.equals("finished")) break;
                        decisions++;
                        if (game.isFinished()) break;
                    }
                    if (!outcome.equals("finished")) break;
                }
                if (game.isFinished() && game.getWinner() != null) winner = game.getWinner();
                turns = game.getGameState().getPlayersLatestTurnNumber(DS) + game.getGameState().getPlayersLatestTurnNumber(LS);
            } catch (Throwable ex) {
                outcome = "exception";
                error = ex.toString();
                if (stackLog != null) synchronized (stackLog) {
                    stackLog.printf("=== %s  ds=%s ls=%s ds_ai=%s ls_ai=%s decisions=%d%n", gameId, dsName, lsName, dsAi, lsAi, decisions);
                    ex.printStackTrace(stackLog);
                    stackLog.flush();
                }
            } finally {
                for (Map.Entry<String, SwccgAiController> e : ais.entrySet()) {
                    if (e.getValue() instanceof PolicyAi) {
                        ((PolicyAi) e.getValue()).finish(outcome, winner, e.getKey().equals(DS) ? lsAi : dsAi);
                        ((PolicyAi) e.getValue()).close();
                    }
                }
            }
            return String.format(Locale.ROOT, "%s,%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%.1f,%s", gameId, thread, csv(dsName), csv(lsName),
                    dsAi, lsAi, outcome, winner, decisions, exportedHere, unmappedHere, invalid, turns, (System.nanoTime() - start) / 1e6, csv(error));
        }
    }

    static void seedShuffles(long s) {
        try {
            var f = Collections.class.getDeclaredField("r");
            f.setAccessible(true);
            f.set(null, new Random(s));
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    static String csv(String s) {
        return s == null ? "" : "\"" + s.replace("\"", "'").replace("\n", " ") + "\"";
    }
}
