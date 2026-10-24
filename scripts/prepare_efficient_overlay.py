"""Stage isolated bench schema-4 text dispatch."""

from pathlib import Path
import hashlib
import os
import tempfile


ROOT = Path(__file__).resolve().parents[1]
# Plan decision-cap-fixes: the harness sources come from the engine tree the run uses (SWCCG_ENGINE_TREE, as
# scripts/efficient_java_check.sh), so a test build's harness is what runs; default: the main build.
# Plan v-cards: every engine source read here comes from that tree (three paths were pinned to the main build).
TREE = Path(os.environ.get("SWCCG_ENGINE_TREE", ROOT / ".cache/gemp-knowledge-v5"))
SOURCE = TREE / "gemp-swccg-server/src/test/java/com/gempukku/swccgo/bench/StateEncoder.java"
OUT = ROOT / "runs/knowledge-v3.1/overlay-source/StateEncoder.java"


def write_source(path, text):
    # Unchanged content keeps the file (and its mtime): the sync guard (plan host-migration) sees no change after a run.
    if path.exists() and path.read_text() == text:
        return
    # Independent fixture collectors can compile concurrently; never expose a truncated source.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def prepare():
    baseline = SOURCE.read_text()
    source = baseline.replace(
        "import com.gempukku.swccgo.common.Zone;",
        "import com.gempukku.swccgo.common.Zone;\n"
        "import com.gempukku.swccgo.game.knowledge.TitleMatcher;",
    )
    source = source.replace(
        "List<String> tb = new ArrayList<>();",
        "List<String> tb = e.version == 4 ? new TitleMatcher.References() : new ArrayList<>();",
    )
    source = source.replace("template(texts[i], tb)", "templateFor(e, texts[i], tb)")
    source = source.replace("template(res[i], tb)", "templateFor(e, res[i], tb)")
    source = source.replace(
        "e.observation.reference(textBps.get(i),e.answers.size()+1,i);",
        "referenceFor(e,textBps,e.answers.size()+1,i);",
    )
    source = source.replace(
        "public String criticTokens;", "public String criticTokens;\n        public String criticGrid;"
    )
    source = source.replace(
        'return sb.append("]]").toString();',
        'sb.append("]");\n'
        '            if (version == 4) sb.append(",").append('
        'com.gempukku.swccgo.game.knowledge.KnowledgeGrid.json(observation.kgrid));\n'
        '            return sb.append("]").toString();',
    )
    source = source.replace(
        '+","+criticTokens+","+knowledgeSequence+"]";',
        '+","+criticTokens+","+knowledgeSequence+(version == 4 ? ","+criticGrid : "")+"]";',
    )
    helper = """
    private static void referenceFor(Encoded encoded, List<String> references, int option, int ordinal) {
        if (encoded.version == 4) {
            encoded.observation.reference(references, option, ordinal);
        } else {
            encoded.observation.reference(references.get(ordinal), option, ordinal);
        }
    }

    private static String templateFor(Encoded encoded, String text, List<String> references) {
        if (encoded.version == 4) {
            return TitleMatcher.shared().template(text, (TitleMatcher.References) references,
                TitleMatcher.permitted(encoded.observation));
        }
        return template(text, references);
    }

"""
    source = source.replace("    static boolean visibleTo(", helper + "    static boolean visibleTo(")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_source(OUT, source)
    write_source(OUT.with_suffix(".baseline-sha256"), hashlib.sha256(baseline.encode()).hexdigest() + "\n")
    bench = SOURCE.parent
    train = (bench / "TrainGen.java").read_text()
    before = "StateEncoder.Encoded enc = shard != null ? StateEncoder.encode(game, p, decision) : null;"
    after = """StateEncoder.Encoded enc = shard == null ? null
                            : Boolean.getBoolean("policy.exportSchema4")
                                ? KnowledgeStateEncoder.comparison(game, p, decision, 4)
                                : StateEncoder.encode(game, p, decision);"""
    if train.count(before) != 1:
        raise ValueError("TrainGen export anchor changed")
    train = train.replace(before, after)
    train = train.replace(
        "if (seed != null && threads != 1)",
        'if (seed != null && threads != 1 && !Boolean.getBoolean("policy.corpusThreadSeeds"))',
    )
    train = train.replace(
        "    static void seedShuffles(long s) {",
        """    static void seedShuffles(long s) {
        if (Boolean.getBoolean("policy.corpusThreadSeeds")) {
            CorpusRandom.seed(s);
            return;
        }""",
    )
    train = train.replace(
        "while ((g = next.getAndIncrement()) < games) {",
        """while ((g = next.getAndIncrement()) < games) {
                            if (Boolean.getBoolean("policy.vocabularyOnly") && CorpusControl.stopped()) break;""",
    )
    train = train.replace(
        "while (!game.isFinished()) {",
        """while (!game.isFinished()) {
                    if (Boolean.getBoolean("policy.vocabularyOnly") && CorpusControl.stopped()) {
                        outcome = "stopped";
                        break;
                    }""",
    )
    train = train.replace(
        '                outcome = "exception";',
        '                if (Boolean.getBoolean("policy.vocabularyOnly")) CorpusControl.stop();\n'
        '                outcome = "exception";',
    )
    # Plan objective-dojo: GameStats and the Dojo reward model are always on (one trailing stats CSV column), and every
    # deck must have a Dojo reward for each of its objectives, Epic Events and Jedi Tests (fatal coverage gate).
    before = 'System.err.printf("decks loaded: %d (pool DS %d, LS %d, matchups %d)%n", decks.size(), dsDecks.size(), lsDecks.size(), matchups.size());'
    if train.count(before) != 1:
        raise ValueError("TrainGen deck-loading anchor changed")
    # Plan v-cards: -Dtraingen.dojoCoverage=false skips the gate (evaluation only, evaluate_v31 --no-dojo-coverage).
    train = train.replace(before, before + '\n        if (!"false".equals(System.getProperty("traingen.dojoCoverage")))'
                          ' ObjectiveDojo.require(decks, library);')
    for before, after in (
        ("game.setSnapshotsEnabled(false);",
         'game.setSnapshotsEnabled(false);\n'
         '                GameStats stats = GameStats.attach(game, DS, LS, gameId, dmap, library);'),
        ("turns = game.getGameState().getPlayersLatestTurnNumber(DS) + game.getGameState().getPlayersLatestTurnNumber(LS);",
         "turns = game.getGameState().getPlayersLatestTurnNumber(DS) + game.getGameState().getPlayersLatestTurnNumber(LS);\n"
         "                if (stats != null) statsJson = stats.json(game);"),
        ('String outcome = "finished", winner = "", error = "";',
         'String outcome = "finished", winner = "", error = "", statsJson = "";'),
        ('pw.println("game,thread,ds_deck,ls_deck,ds_ai,ls_ai,outcome,winner,decisions,exported,unmapped,invalid,turns,wall_ms,error");',
         'pw.println("game,thread,ds_deck,ls_deck,ds_ai,ls_ai,outcome,winner,decisions,exported,unmapped,invalid,turns,wall_ms,error"\n'
         '                + ",stats");'),
        ("(System.nanoTime() - start) / 1e6, csv(error));",
         '(System.nanoTime() - start) / 1e6, csv(error))\n'
         '                    + ",\\"" + statsJson.replace("\\"", "\\"\\"") + "\\"";'),
    ):
        if train.count(before) != 1:
            raise ValueError("TrainGen stats anchor changed: " + before[:40])
        train = train.replace(before, after)
    # Plan objective-dojo BL-20(b), diagnostic: engine labels per location next to every exported decision
    # (-Dpolicy.locationProbe=true with --export); no effect on play or on runs without the flag.
    for before, after in (
        ('new File(exportDir, "shard-" + tag + "-t" + thread + ".jsonl.gz")), 1 << 16), StandardCharsets.UTF_8), 1 << 16);',
         'new File(exportDir, "shard-" + tag + "-t" + thread + ".jsonl.gz")), 1 << 16), StandardCharsets.UTF_8), 1 << 16);\n'
         '                        LocationProbe.open(pw != null ? exportDir : null, tag, thread);'),
        ("                            shard.write('\\n');",
         "                            shard.write('\\n');\n                            LocationProbe.write(game, p, gameId, decisions);"),
        ("                        if (shard != null) try { shard.close(); } catch (IOException ignored) { }",
         "                        if (shard != null) try { shard.close(); } catch (IOException ignored) { }\n                        LocationProbe.close();"),
    ):
        if train.count(before) != 1:
            raise ValueError("TrainGen location-probe anchor changed: " + before[:50])
        train = train.replace(before, after)
    # Plans activation-and-cancel B and free-move-rule, permanent (plan permanent-play-rules): for every player.
    before = "SwccgAiController ai = ais.get(p);"
    if train.count(before) != 1:
        raise ValueError("TrainGen visible-cancel anchor changed")
    train = train.replace(before, "com.gempukku.swccgo.game.rules.VisibleCancel.restrict(decision, p);\n"
                          "                        com.gempukku.swccgo.game.rules.FreeMoveRule.restrict(game, decision);\n"
                          "                        " + before)
    # Plan free-move-rule: remember which card embarks (for the follow-up target filter).
    before = "String answer = ai.decide(p, decision, game.getGameState());"
    if train.count(before) != 1:
        raise ValueError("TrainGen free-move anchor changed")
    train = train.replace(before, before + "\n                        com.gempukku.swccgo.game.rules.FreeMoveRule.decided(game, decision, answer);")
    # Plan dojo-focus: --opponentPool FILE ("DS|LS<TAB>path" lines). With --pool, the agent seat keeps its draw from
    # --pool and the other seat draws a deck of its side from this pool (own seeded stream). Absent: unchanged.
    for before, after in (
        ("        Map<String, SwccgDeck> decks = new HashMap<>();\n",
         """        if (opt.containsKey("opponentPool")) {
            for (String line : Files.readAllLines(new File(opt.get("opponentPool")).toPath())) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] p = line.split("\\t");
                (p[0].trim().equalsIgnoreCase("DS") ? OPPONENT_DS : OPPONENT_LS).add(p[1].trim());
            }
            if (dsDecks.isEmpty() || OPPONENT_DS.isEmpty() || OPPONENT_LS.isEmpty())
                throw new IllegalArgumentException("--opponentPool needs --pool and DS and LS opponent decks");
            System.err.printf("opponent pool: DS %d, LS %d%n", OPPONENT_DS.size(), OPPONENT_LS.size());
        }
        Map<String, SwccgDeck> decks = new HashMap<>();
        for (String d : OPPONENT_DS) decks.computeIfAbsent(d, k -> DeckLoader.loadDeck(resolve(deckDir, k), library));
        for (String d : OPPONENT_LS) decks.computeIfAbsent(d, k -> DeckLoader.loadDeck(resolve(deckDir, k), library));
"""),
        ("    static void seedShuffles(long s) {",
         "    static final List<String> OPPONENT_DS = new ArrayList<>(), OPPONENT_LS = new ArrayList<>();\n\n"
         "    static void seedShuffles(long s) {"),
        ("            String opp = opponentFor(g);\n",
         """            if (!OPPONENT_DS.isEmpty() && !dsDecks.isEmpty()) {
                Random opponentRnd = new Random(pairSeed * 1_000_003L + g + 0x0DD5EEDL);
                if (agentDs) lsName = OPPONENT_LS.get(opponentRnd.nextInt(OPPONENT_LS.size()));
                else dsName = OPPONENT_DS.get(opponentRnd.nextInt(OPPONENT_DS.size()));
            }
            String opp = opponentFor(g);
"""),
    ):
        if train.count(before) != 1:
            raise ValueError("TrainGen opponent-pool anchor changed: " + before[:50])
        train = train.replace(before, after)
    # Plan links-round2 (balanced matchups): -Dtraingen.pairWeights=FILE ("dsPath<TAB>lsPath<TAB>weight", weight in
    # (0, 1]). With --pool, the final pairing (after the opponent-pool draw) is kept with probability weight; otherwise
    # both seats are redrawn from their pools (attempt-salted seeded streams), so a listed pairing is played at weight
    # times its usual rate. Unset: unchanged.
    for before, after in (
        ("    static void seedShuffles(long s) {",
         """    public static final Map<String, Double> PAIR_WEIGHTS = loadPairWeights(System.getProperty("traingen.pairWeights"));

    static Map<String, Double> loadPairWeights(String path) {
        Map<String, Double> weights = new HashMap<>();
        if (path == null || path.isBlank()) return weights;
        try {
            for (String line : Files.readAllLines(new File(path).toPath())) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] p = line.split("\\t");
                double w = Double.parseDouble(p[2].trim());
                if (!(w > 0 && w <= 1)) throw new IllegalArgumentException("pair weight must be in (0, 1]: " + line);
                weights.put(p[0].trim() + "\\t" + p[1].trim(), w);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read pair weights " + path, e);
        }
        System.err.printf("pair weights: %d pairings from %s%n", weights.size(), path);
        return weights;
    }

    // Plan links-round2 W-bal: -Dtraingen.deckWeights=FILE, lines "deckPath<TAB>factor" and optionally "*<TAB>default"
    // (factors in (0, 1], unlisted decks get the default, 1 if absent). A pairing is kept with probability
    // pairWeight * factor(ds) * factor(ls), so each seat's deck draw is reweighted by its factor.
    public static final Map<String, Double> DECK_WEIGHTS = loadDeckWeights(System.getProperty("traingen.deckWeights"));

    static Map<String, Double> loadDeckWeights(String path) {
        Map<String, Double> weights = new HashMap<>();
        if (path == null || path.isBlank()) return weights;
        try {
            for (String line : Files.readAllLines(new File(path).toPath())) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] p = line.split("\\t");
                double w = Double.parseDouble(p[1].trim());
                if (!(w > 0 && w <= 1)) throw new IllegalArgumentException("deck weight must be in (0, 1]: " + line);
                weights.put(p[0].trim(), w);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read deck weights " + path, e);
        }
        System.err.printf("deck weights: %d entries from %s%n", weights.size(), path);
        return weights;
    }

    static double deckWeight(String deck) {
        Double w = DECK_WEIGHTS.get(deck);
        return w != null ? w : DECK_WEIGHTS.getOrDefault("*", 1.0);
    }

    /** {ds, ls} of game g: the first draw (ds, ls) is kept with its pair weight, else both seats are redrawn. */
    public static String[] weightedPairing(String ds, String ls, List<String> dsDecks, List<String> lsDecks, boolean agentDs,
                                           long pairSeed, int g) {
        for (int attempt = 1; ; attempt++) {
            double w = PAIR_WEIGHTS.getOrDefault(ds + "\\t" + ls, 1.0);
            if (!DECK_WEIGHTS.isEmpty()) w *= deckWeight(ds) * deckWeight(ls);
            SplittableRandom rnd = new SplittableRandom(pairSeed * 1_000_003L + g + attempt * 0x5DEECE66DL);
            if (w >= 1.0 || rnd.nextDouble() < w) return new String[]{ds, ls};
            ds = dsDecks.get(rnd.nextInt(dsDecks.size()));
            ls = lsDecks.get(rnd.nextInt(lsDecks.size()));
            if (!OPPONENT_DS.isEmpty()) {
                if (agentDs) ls = OPPONENT_LS.get(rnd.nextInt(OPPONENT_LS.size()));
                else ds = OPPONENT_DS.get(rnd.nextInt(OPPONENT_DS.size()));
            }
        }
    }

    static void seedShuffles(long s) {"""),
        ("            String opp = opponentFor(g);\n",
         """            if ((!PAIR_WEIGHTS.isEmpty() || !DECK_WEIGHTS.isEmpty()) && !dsDecks.isEmpty()) {
                String[] weighted = weightedPairing(dsName, lsName, dsDecks, lsDecks, agentDs, pairSeed, g);
                dsName = weighted[0];
                lsName = weighted[1];
            }
            String opp = opponentFor(g);
"""),
    ):
        if train.count(before) != 1:
            raise ValueError("TrainGen pair-weights anchor changed: " + before[:50])
        train = train.replace(before, after)
    # Plan titles-and-learner-ablation A: the on/off identity dump, and an export that encodes with marks visible
    # (normalizer v2) to collect tables-v7's keys.
    before = "                        String answer = ai.decide(p, decision, game.getGameState());"
    if train.count(before) != 1:
        raise ValueError("TrainGen decide anchor changed")
    train = train.replace(before, before + "\n                        NameMarkCheck.decision(gameId, decisions, p, decision, answer);")
    before = "                                : StateEncoder.encode(game, p, decision);"
    if train.count(before) != 1:
        raise ValueError("TrainGen export-encode anchor changed")
    train = train.replace(before, "                                : com.gempukku.swccgo.game.knowledge.NameMarks.raw("
                          "Boolean.getBoolean(\"policy.nameMarksExport\"), () -> StateEncoder.encode(game, p, decision));")
    # Plan critic-followups BL-49: record / replay answers (ReplayControl; off unless replay.record or replay.jobs).
    for before, after in (
        ("            if (seed != null) seedShuffles(seed + g);",
         "            if (seed != null) seedShuffles(ReplayControl.seed(g, seed + g));"),
        ("                        String answer = ai.decide(p, decision, game.getGameState());",
         "                        String answer = ReplayControl.active()\n"
         "                            ? ReplayControl.decide(game, gameId, decisions, p, decision,\n"
         "                                () -> ai.decide(p, decision, game.getGameState()))\n"
         "                            : ai.decide(p, decision, game.getGameState());"),
        ("            } finally {\n                for (Map.Entry<String, SwccgAiController> e : ais.entrySet()) {",
         "            } finally {\n                if (ReplayControl.active()) ReplayControl.finish(gameId, outcome, winner, decisions);\n"
         "                for (Map.Entry<String, SwccgAiController> e : ais.entrySet()) {"),
    ):
        if train.count(before) != 1:
            raise ValueError("TrainGen replay anchor changed: " + before[:60])
        train = train.replace(before, after)
    write_source(OUT.parent / "TrainGen.java", train)
    # Plan activation-and-cancel A: one Activate Force action per turn, no opponent "allow" cap (flag-guarded).
    cards = TREE / "gemp-swccg-cards/src/main/java/com/gempukku/swccgo/cards/AbstractSwccgCardBlueprint.java"
    blueprint = cards.read_text()
    for before, after in (
        ("if (GameConditions.canActivateForceWithCardPileAction(playerId, gameState, modifiersQuerying, self)) {",
         "if (GameConditions.canActivateForceWithCardPileAction(playerId, gameState, modifiersQuerying, self)\n"
         "                && !ActivateOnce.used(game, playerId)) {"),
        ("""new IntegerAwaitingDecision("Choose amount of Force to activate", 0, maxToActivate, maxToActivate) {
                                @Override
                                public void decisionMade(final int result) throws DecisionResultInvalidException {""",
         """new IntegerAwaitingDecision("Choose amount of Force to activate", 0, maxToActivate, maxToActivate) {
                                @Override
                                public void decisionMade(final int result) throws DecisionResultInvalidException {
                                    // Plan critic-replay: every answer, 0 included, is the turn's activation.
                                    ActivateOnce.mark(game, playerId);"""),
        ("""                                    else {
                                        // Confirm with opponent max number of Force to allow player to activate before giving opponent a chance to perform an action""",
         """                                    else if (true) { // ActivateOnce, always on: no opponent "allow" cap
                                        action.appendEffect(
                                                new ActivateForceEffect(action, playerId, result, true));
                                    }
                                    else {
                                        // Confirm with opponent max number of Force to allow player to activate before giving opponent a chance to perform an action"""),
    ):
        if blueprint.count(before) != 1:
            raise ValueError("AbstractSwccgCardBlueprint anchor changed: " + before[:50])
        blueprint = blueprint.replace(before, after)
    write_source(OUT.parent / "AbstractSwccgCardBlueprint.java", blueprint)
    # Plan critic-replay (user 2026-10-08, always on): the player's own Activate phase starts with the Force activation
    # amount (0 … max, the "Activate Force" card-pile action performed by the engine); the "You have not activated
    # Force. Do you want to Pass?" check is removed (it made a Pass -> No -> menu loop, ~45% of all decisions).
    process = (TREE / "gemp-swccg-logic/src/main/java/com/gempukku/swccgo/logic/timing/"
               "processes/turn/general/PlayersPlayPhaseActionsInOrderGameProcess.java").read_text()
    for before, after in (
        ("import com.gempukku.swccgo.common.Phase;\n",
         "import com.gempukku.swccgo.cards.ActivateOnce;\nimport com.gempukku.swccgo.common.Phase;\n"),
        ("import com.gempukku.swccgo.logic.decisions.CardActionSelectionDecision;\n",
         "import com.gempukku.swccgo.logic.actions.CardPileAction;\n"
         "import com.gempukku.swccgo.logic.decisions.CardActionSelectionDecision;\n"),
        ("""        final List<Action> playableActions = game.getActionsEnvironment().getTopLevelActions(playerId);
""",
         """        final List<Action> playableActions = game.getActionsEnvironment().getTopLevelActions(playerId);

        // Overlay (plan critic-replay): the own Activate phase starts with the activation amount, asked once per turn.
        if (playerId.equals(game.getGameState().getCurrentPlayerId()) && game.getGameState().getCurrentPhase() == Phase.ACTIVATE
                && !ActivateOnce.used(game, playerId)) {
            for (final Action action : playableActions) {
                if (action instanceof CardPileAction && "Activate Force".equals(action.getText())) {
                    game.takeSnapshot(playerId + ": " + action.getText());
                    action.appendAfterEffect(new PassthruEffect(action) {
                        @Override
                        protected void doPlayEffect(SwccgGame game) {
                            if (action.isChoosingTargetsComplete() || action.wasCarriedOut()) {
                                _nextProcess = new PlayersPlayPhaseActionsInOrderGameProcess(game.getGameState().getPlayerOrder().getPlayOrder(_playOrder.getNextPlayer(), true), 0, _followingGameProcess);
                            }
                            else {
                                checkPlayerAgain(game);
                            }
                        }
                    });
                    game.getActionsEnvironment().addActionToStack(action);
                    return;
                }
            }
        }
"""),
        ("""                                // If this is the player's Activate phase, and no Force was activated (but could have been),
                                // check if the player meant to click Pass.
                                if (game.getGameState().getCurrentPlayerId().equals(playerId) && game.getGameState().getCurrentPhase() == Phase.ACTIVATE
                                        && !game.getModifiersQuerying().isActivatingForceProhibited(game.getGameState(), playerId) && game.getGameState().getReserveDeckSize(playerId) > 0
                                        && game.getModifiersQuerying().getForceActivatedThisTurn(playerId, false) == 0) {

                                    game.getUserFeedback().sendAwaitingDecision(playerId,
                                            new YesNoDecision("You have not activated Force. Do you want to Pass?") {
                                                @Override
                                                protected void yes() {
                                                    playerPassed(game, playerId);
                                                }

                                                @Override
                                                protected void no() {
                                                    checkPlayerAgain(game);
                                                }
                                            });
                                } else {
                                    playerPassed(game, playerId);
                                }""",
         """                                // Overlay (plan critic-replay): no "You have not activated Force" check; the amount was asked.
                                playerPassed(game, playerId);"""),
    ):
        if process.count(before) != 1:
            raise ValueError("PlayersPlayPhaseActionsInOrderGameProcess anchor changed: " + before[:50])
        process = process.replace(before, after)
    write_source(OUT.parent / "PlayersPlayPhaseActionsInOrderGameProcess.java", process)
    # Engine crash fix (standing permission; 2026-10-04): a deploy "as react" to a non-location target (e.g. aboard a
    # starship) has PlayCardResult.getToLocation() == null, and reactedToLocation passed it into a location filter
    # (NPE via Admiral Ozzel, Card3_082). Resolve the react location from the deployed card instead; none -> false.
    triggers = (TREE / "gemp-swccg-logic/src/main/java/com/gempukku/swccgo/logic/"
                "TriggerConditions.java").read_text()
    for before, after in (
        ("""                return Filters.and(cardFilter).accepts(game, playCardResult.getPlayedCard())
                        && Filters.and(locationFilter).accepts(game.getGameState(), game.getModifiersQuerying(), playCardResult.getToLocation());""",
         """                PhysicalCard reactLocation = reactLocation(game, playCardResult);
                return reactLocation != null && Filters.and(cardFilter).accepts(game, playCardResult.getPlayedCard())
                        && Filters.and(locationFilter).accepts(game.getGameState(), game.getModifiersQuerying(), reactLocation);"""),
        ("""                return Filters.and(locationFilter).accepts(game.getGameState(), game.getModifiersQuerying(), playCardResult.getToLocation());
            }
        }
        return false;
    }""",
         """                PhysicalCard reactLocation = reactLocation(game, playCardResult);
                return reactLocation != null
                        && Filters.and(locationFilter).accepts(game.getGameState(), game.getModifiersQuerying(), reactLocation);
            }
        }
        return false;
    }

    /** Overlay fix: the location a card deployed "as react" went to, also when its target was not a location. */
    private static PhysicalCard reactLocation(SwccgGame game, PlayCardResult playCardResult) {
        PhysicalCard location = playCardResult.getToLocation();
        if (location != null) return location;
        PhysicalCard played = playCardResult.getPlayedCard();
        return played == null ? null : game.getModifiersQuerying().getLocationThatCardIsAt(game.getGameState(), played);
    }"""),
    ):
        if triggers.count(before) != 1:
            raise ValueError("TriggerConditions anchor changed: " + before[:60])
        triggers = triggers.replace(before, after)
    write_source(OUT.parent / "TriggerConditions.java", triggers)
    # Plan engine-on-feat-brainiac: the Malastare (Card12_168) and RelocateBetweenLocationsEffect crash fixes are commits
    # on feat/brainiac (the engine base) now, so the overlay no longer patches them.
    engine = (bench / "PolicyEngine.java").read_text()
    before = "this.record = record;"
    if engine.count(before) != 1:
        raise ValueError("PolicyEngine recording anchor changed")
    engine = engine.replace(
        before, 'this.record = record && !Boolean.getBoolean("policy.vocabularyOnly");'
    )
    # Plan objective-dojo: no life-force reward term; the writer takes only the games per file.
    before = 'new TrajectoryWriter(run, Double.parseDouble(System.getProperty("policy.native.lfBeta", "0")),'
    if engine.count(before) != 1:
        raise ValueError("PolicyEngine trajectory-writer anchor changed")
    engine = engine.replace(before, "new TrajectoryWriter(run,")
    engine = engine.replace("TrajectoryWriter, reward with\n * policy.native.lfBeta).", "TrajectoryWriter, Standard Gameplay\n * reward plus the Dojo rewards).")
    engine = engine.replace("PolicyTensors.packedBytes(sh[0], sh[1], sh[2])", "PolicyTensors.packedBytes(sh)")
    engine = engine.replace("PolicyTensors.pack(ts, sh[0], sh[1], sh[2], ins.get(i))",
                            "PolicyTensors.pack(ts, sh, ins.get(i))")
    engine = engine.replace("NativePolicy.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2])",
        "(sh.length == 5 ? NativePolicyV4.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2], sh[3])"
        " : NativePolicy.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2]))")
    engine = engine.replace("final Path source;", "final Path source;\n        final PolicyTensors.Maps maps;")
    engine = engine.replace("Slot(String name, long handle, Path source, long mtime, String label) {",
                            "Slot(String name, long handle, Path source, long mtime, String label, "
                            "PolicyTensors.Maps maps) {\n            this.maps = maps;")
    engine = engine.replace("long h = NativePolicy.load(ts.toString(), graphs);",
        'PolicyTensors.Maps owned = maps;\n'
        '        Path bundle = Paths.get(ts.toString() + ".bundle");\n'
        '        if (maps.schemaVersion == 4 && Files.isDirectory(bundle)) {\n'
        '            owned = PolicyTensors.Maps.load(bundle);\n'
        '            if (owned.schemaVersion != 4) {\n'
        '                throw new IllegalArgumentException("Historical bundle must use the current schema");\n'
        '            }\n'
        '        }\n'
        '        long h = NativePolicy.load(ts.toString(), graphs);')
    engine = engine.replace("return new Slot(name, h, ts, mtime(ts), labelOf(name, ts));",
        'if (owned.schemaVersion == 4 && NativePolicyV4.tensorAbi(h) != owned.tensorAbi) {\n'
        '            NativePolicy.free(h);\n'
        '            throw new IllegalStateException("Bundle tensor ABI " + owned.tensorAbi + " != module " + NativePolicyV4.tensorAbi(h));\n'
        '        }\n'
        'if (owned.schemaVersion == 4 && NativePolicyV4.checkTable(h, owned.tableSha) != 0) {\n'
        '            NativePolicy.free(h);\n'
        '            throw new IllegalStateException("Native table mismatch: " + NativePolicy.lastError());\n'
        '        }\n'
        '        return new Slot(name, h, ts, mtime(ts), labelOf(name, ts), owned);')
    engine = engine.replace("PolicyTensors.fromRecord(actorLine,maps)", "PolicyTensors.fromRecord(actorLine,slot.maps)")
    # Plan titles-and-learner-ablation A: does this seat's bundle read marked card names (normalizer v2)?
    before = "    private Slot slotFor(String base, String game) {"
    if engine.count(before) != 1:
        raise ValueError("PolicyEngine slotFor anchor changed")
    engine = engine.replace(before,
        "    public boolean nameMarks(String modelKey, String game) {\n"
        "        String base = modelKey.endsWith(\"@greedy\") ? modelKey.substring(0, modelKey.length() - 7) : modelKey;\n"
        "        Slot slot = slotFor(base, game);\n"
        "        return slot != null && slot.maps.actorTables != null && slot.maps.actorTables.nameMarks;\n"
        "    }\n\n" + before)
    engine = engine.replace("encoded.binary.actor(encoded,maps)", "encoded.binary.actor(encoded,slot.maps)")
    engine = engine.replace("BinaryTrajectoryCheck.check(encoded,maps,r.tensors)",
                            "BinaryTrajectoryCheck.check(encoded,slot.maps,r.tensors)")
    engine = engine.replace("encoded.binary.critic(encoded,maps)", "encoded.binary.critic(encoded,slot.maps)")
    engine = engine.replace("r.repeats,r.k,r.excluded,r.policyLabel,maps);",
                            "r.repeats,r.k,r.excluded,r.policyLabel,slot.maps);")
    write_source(OUT.parent / "PolicyEngine.java", engine)
    name_marks(bench)



def name_marks(bench):
    """Plan titles-and-learner-ablation A: getFullName marks card names (NameMarks, always on); every reader
    but a normalizer-v2 encoder gets today's text back (decision getters, duplicate-action checks, snapshots,
    messages). PolicyAi encodes inside NameMarks.raw when its seat's bundle uses normalizer v2."""
    from build_knowledge_engine import LOGIC, NAME_MARK_EDITS, name_marks as mark
    tree = TREE
    for relative in NAME_MARK_EDITS:
        write_source(OUT.parent / Path(relative).name, mark(relative, (tree / relative).read_text()))
    marks = "com.gempukku.swccgo.game.knowledge.NameMarks"
    policy_ai = (bench / "PolicyAi.java").read_text()
    for before in ("StateEncoder.encode(game, playerId, decision)",):
        if policy_ai.count(before) != 2:
            raise ValueError("PolicyAi encode anchor changed")
    policy_ai = policy_ai.replace(
        "StateEncoder.encode(game, playerId, decision)",
        f"{marks}.raw(engine != null && engine.nameMarks(modelKey, gameId), "
        "() -> StateEncoder.encode(game, playerId, decision))")
    write_source(OUT.parent / "PolicyAi.java", policy_ai)


if __name__ == "__main__":
    prepare()
