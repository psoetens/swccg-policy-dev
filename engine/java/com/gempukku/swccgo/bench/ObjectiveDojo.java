package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.cards.GameConditions;
import com.gempukku.swccgo.common.CardCategory;
import com.gempukku.swccgo.common.Icon;
import com.gempukku.swccgo.common.InactiveReason;
import com.gempukku.swccgo.logic.modifiers.ModifyGameTextType;
import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.common.SpotOverride;
import com.gempukku.swccgo.common.Title;
import com.gempukku.swccgo.filters.Filter;
import com.gempukku.swccgo.filters.Filters;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgCardBlueprint;
import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.TriggerConditions;
import com.gempukku.swccgo.logic.timing.Action;
import com.gempukku.swccgo.logic.timing.EffectResult;
import com.gempukku.swccgo.logic.timing.results.DestinyDrawnResult;
import com.gempukku.swccgo.logic.timing.results.DoubleSidedCardFlippedResult;
import com.gempukku.swccgo.logic.timing.results.DuelInitiatedResult;
import com.gempukku.swccgo.logic.timing.results.DuelResultDeterminedResult;
import com.gempukku.swccgo.logic.timing.results.ForceDrainInitiatedResult;
import com.gempukku.swccgo.logic.timing.results.JediTestCompletedResult;
import com.gempukku.swccgo.logic.timing.results.LostForceResult;
import com.gempukku.swccgo.logic.timing.results.LostFromTableResult;
import com.gempukku.swccgo.logic.timing.results.LightsaberCombatResultDeterminedResult;
import com.gempukku.swccgo.logic.timing.ActionStack;
import com.gempukku.swccgo.game.state.actions.DefaultActionsEnvironment;
import com.gempukku.swccgo.logic.timing.results.PlayCardResult;
import com.gempukku.swccgo.logic.timing.results.BlownAwayLastStepResult;
import com.gempukku.swccgo.logic.vo.SwccgDeck;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Plan objective-dojo, Part A: the Dojo reward model. It is always on, in every game the harness plays.
 *
 * <p>Each objective, Epic Event and Jedi Test in a deck is an item with a ladder of milestones and a success value
 * ({@code dojo/rewards.json} on the classpath, copied from {@code data/dojo/rewards.json}). Gaining a milestone on own
 * turn t pays w * max(f, 1 - (t - 1) / T); losing a held state milestone refunds what it paid. An earned objective
 * flip pays the success value; a flip-back, or the flipped objective leaving play, refunds it. Event milestones and
 * Epic Event / Jedi Test successes are permanent. Decks without any item get only the Standard Gameplay reward (the
 * win/loss reward in {@link BinaryTrajectoryWriter}).
 *
 * <p>Plan dojo-three-objectives: an objective milestone with {@code "when": "flipped"} moves only while the objective is
 * flipped (where the flip only stages the payoff); such a state milestone is refunded on a flip-back or when the
 * objective leaves play. An event milestone with {@code "repeat": n} pays each time it happens, at most n times.
 * A 'flipped' milestone decays from the own turn of the objective's first flip instead of from the start of the game.
 *
 * <p>The predicates only read game state; the observer adds no actions, so play is unchanged.
 */
public final class ObjectiveDojo {
    static final Set<CardCategory> CATEGORIES = EnumSet.of(CardCategory.OBJECTIVE, CardCategory.EPIC_EVENT,
        CardCategory.JEDI_TEST);
    private static final String RESOURCE = "/dojo/rewards.json";
    private static final Map<InactiveReason, Boolean> SPOT = SpotOverride.INCLUDE_EXCLUDED_FROM_BATTLE;
    private static final Map<InactiveReason, Boolean> SPOT_CAPTIVES = SpotOverride.INCLUDE_CAPTIVE_AND_EXCLUDED_FROM_BATTLE;

    // ---------------------------------------------------------------- configuration

    static final class Milestone {
        final String name;
        final double weight;
        final boolean necessary;
        final boolean event;
        /** Moves only while the objective is flipped (plan dojo-three-objectives); otherwise only while front side up. */
        final boolean afterFlip;
        /** Event milestones: how many times the event pays (default once). */
        final int repeat;

        Milestone(JSONObject json) {
            name = json.getString("name");
            weight = json.getDouble("w");
            necessary = json.getBoolean("necessary");
            event = "event".equals(json.getString("kind"));
            String when = json.optString("when", "front");
            if (!when.equals("front") && !when.equals("flipped")) throw new IllegalStateException("Dojo: unknown when " + when);
            afterFlip = when.equals("flipped");
            repeat = json.optInt("repeat", 1);
        }
    }

    static final class Item {
        final String key;
        final String blueprint;
        final String title;
        final String kind;
        final boolean counter;
        final double success;
        final double horizon;
        final double floor;
        final List<Milestone> milestones = new ArrayList<>();
        final Set<String> excludedBy = new HashSet<>();

        Item(String key, JSONObject json, JSONObject decay) {
            this.key = key;
            blueprint = key.contains("~") ? key.substring(0, key.indexOf('~')) : key;
            title = json.getString("title");
            kind = json.getString("kind");
            counter = json.optBoolean("counter", false);
            success = json.getDouble("success");
            horizon = json.optDouble("T", decay.getDouble("T"));
            floor = json.optDouble("f", decay.getDouble("f"));
            JSONArray ms = json.getJSONArray("milestones");
            for (int i = 0; i < ms.length(); i++) milestones.add(new Milestone(ms.getJSONObject(i)));
            JSONArray excluded = json.optJSONArray("excluded_by");
            if (excluded != null) for (int i = 0; i < excluded.length(); i++) excludedBy.add(excluded.getString(i));
        }

        boolean objective() {
            return "objective".equals(kind);
        }

        double decay(int ownTurn) {
            return Math.max(floor, 1 - (Math.max(1, ownTurn) - 1) / horizon);
        }
    }

    static final Map<String, Item> ITEMS = load();
    /** Card blueprint id to the item keys it activates (its own item and any counter item). */
    static final Map<String, List<Item>> BY_BLUEPRINT = new HashMap<>();

    private static Map<String, Item> load() {
        try (InputStream in = ObjectiveDojo.class.getResourceAsStream(RESOURCE)) {
            if (in == null) throw new IllegalStateException("Dojo rewards missing from the classpath: " + RESOURCE
                + " (the overlay build copies data/dojo/rewards.json there)");
            JSONObject json = new JSONObject(new String(in.readAllBytes(), StandardCharsets.UTF_8));
            JSONObject decay = json.getJSONObject("decay");
            JSONObject items = json.getJSONObject("items");
            Map<String, Item> result = new LinkedHashMap<>();
            for (String key : new TreeMap<>(items.toMap()).keySet()) result.put(key, new Item(key, items.getJSONObject(key), decay));
            return result;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    // ---------------------------------------------------------------- predicates

    /** Per game and side: what a predicate may look at. */
    static final class Ctx {
        final SwccgGame game;
        final String player;
        final String opponent;

        Ctx(SwccgGame game, String player, String opponent) {
            this.game = game;
            this.player = player;
            this.opponent = opponent;
        }

        GameState gs() {
            return game.getGameState();
        }

        /** A card of this side on table, used as the spotting source (as the cards themselves pass {@code self}). */
        PhysicalCard self() {
            PhysicalCard objective = gs().getObjectivePlayed(player);
            if (objective != null) return objective;
            for (PhysicalCard card : gs().getAllPermanentCards())
                if (player.equals(card.getOwner()) && card.getZone() != null && card.getZone().isInPlay()) return card;
            return null;
        }

        boolean spot(int count, Map<InactiveReason, Boolean> overrides, Filter filter) {
            PhysicalCard self = self();
            return self != null && GameConditions.canSpot(game, self, count, overrides, filter);
        }

        boolean spot(Filter filter) {
            return spot(1, SPOT, filter);
        }

        int count(Filter filter) {
            PhysicalCard self = self();
            return self == null ? 0 : Filters.filterActive(game, self, SPOT, filter).size();
        }

        Filter yours(Filter filter) {
            return Filters.and(Filters.your(player), filter);
        }

        boolean onTable(String blueprint) {
            return tableCard(blueprint) != null;
        }

        PhysicalCard tableCard(String blueprint) {
            for (PhysicalCard card : gs().getAllPermanentCards())
                if (player.equals(card.getOwner()) && blueprint.equals(card.getBlueprintId(true))
                        && card.getZone() != null && card.getZone().isInPlay()) return card;
            return null;
        }

        int ownTurn() {
            return gs().getPlayersLatestTurnNumber(player);
        }
    }

    interface State {
        boolean test(Ctx c);
    }

    interface Event {
        boolean test(Ctx c, EffectResult r);
    }

    /** Games in which an epic duel is running (set on DUEL_INITIATED, cleared when the duel ends). */
    private static final Map<SwccgGame, Boolean> EPIC_DUEL = java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    static final Map<String, State> STATES = new HashMap<>();
    static final Map<String, Event> EVENTS = new HashMap<>();

    private static void state(String item, String milestone, State s) {
        if (STATES.put(item + "/" + milestone, s) != null) throw new IllegalStateException("duplicate " + item + "/" + milestone);
    }

    private static void event(String item, String milestone, Event e) {
        if (EVENTS.put(item + "/" + milestone, e) != null) throw new IllegalStateException("duplicate " + item + "/" + milestone);
    }

    /** The card behind the action that drew a destiny (an Epic Event attempt draws one), or null. */
    private static PhysicalCard destinySource(EffectResult r) {
        if (r.getType() != EffectResult.Type.DESTINY_DRAWN) return null;
        Action action = ((DestinyDrawnResult) r).getSourceAction();
        return action == null ? null : action.getActionSource();
    }

    private static boolean attempted(Ctx c, EffectResult r, String blueprint) {
        PhysicalCard source = destinySource(r);
        if (source == null || !blueprint.equals(source.getBlueprintId(true))) return false;
        Action action = ((DestinyDrawnResult) r).getSourceAction();
        return c.player.equals(action.getPerformingPlayer());
    }

    private static boolean blownAway(EffectResult r, Ctx c, Filter filter) {
        if (r.getType() != EffectResult.Type.BLOWN_AWAY_LAST_STEP) return false;
        PhysicalCard card = ((BlownAwayLastStepResult) r).getBlownAwayCard();
        return card != null && filter.accepts(c.gs(), c.game.getModifiersQuerying(), card);
    }

    private static PhysicalCard deathStar(Ctx c) {
        for (PhysicalCard card : c.gs().getAllPermanentCards())
            if (card.getZone() != null && card.getZone().isInPlay()
                    && Filters.Death_Star_system.accepts(c.gs(), c.game.getModifiersQuerying(), card)) return card;
        return null;
    }

    /** Plan dojo-focus rev. 3: your Superlaser on the Death Star system (CPI cannot fire without it). */
    private static boolean superlaserOn(Ctx c) {
        return c.spot(c.yours(Filters.and(Filters.Superlaser, Filters.attachedTo(Filters.Death_Star_system))));
    }

    /** Your Death Star Gunners on the Death Star (each adds 2 to the CPI total, Card2_086). */
    private static int gunnersOnDeathStar(Ctx c) {
        return c.count(c.yours(Filters.and(Filters.Death_Star_Gunner, Filters.on(Title.Death_Star))));
    }

    /** CPI's Y: Death Star sites where the opponent has no presence (each adds 1 to the CPI total, Card2_130). */
    private static int deathStarSitesFree(Ctx c) {
        return Filters.countTopLocationsOnTable(c.game, Filters.and(Filters.Death_Star_site,
            Filters.notIgnoredDuringEpicEventCalculation(true), Filters.not(Filters.occupies(c.opponent))));
    }

    /**
     * A CPI destiny draw with a good set-up: the card's own total without destiny (Y - Z plus CPI total modifiers such as
     * Gunners and Conference Room), plus 3, is at least 7. Z is the opponent's best choice: max(X, Hoth sites, Yavin 4 sites).
     */
    private static boolean goodCpiAttempt(Ctx c, EffectResult r) {
        if (!attempted(c, r, "2_130")) return false;
        PhysicalCard cpi = destinySource(r);
        PhysicalCard deathStar = deathStar(c);
        String target = deathStar == null ? null : deathStar.getSystemOrbited();
        if (target == null) return false;
        int x = Filters.countTopLocationsOnTable(c.game, Filters.and(Filters.site, Filters.notIgnoredDuringEpicEventCalculation(true), Filters.partOfSystem(target)));
        int y = deathStarSitesFree(c);
        int hoth = Filters.countTopLocationsOnTable(c.game, Filters.and(Filters.Hoth_site, Filters.notIgnoredDuringEpicEventCalculation(true)));
        int yavin = Filters.countTopLocationsOnTable(c.game, Filters.and(Filters.Yavin_4_site, Filters.notIgnoredDuringEpicEventCalculation(true)));
        int z = Math.max(x, Math.max(hoth, yavin));
        float setup = c.game.getModifiersQuerying().getEpicEventCalculationTotal(c.gs(), cpi, y - z);
        return setup + 3 >= 7;
    }

    private static PhysicalCard locationOf(Ctx c, PhysicalCard card) {
        return c.game.getModifiersQuerying().getLocationThatCardIsAt(c.gs(), card);
    }

    private static PhysicalCard first(Ctx c, Map<InactiveReason, Boolean> overrides, Filter filter) {
        PhysicalCard self = c.self();
        if (self == null) return null;
        Collection<PhysicalCard> cards = Filters.filterActive(c.game, self, overrides, filter);
        return cards.isEmpty() ? null : cards.iterator().next();
    }

    static {
        // --- Objectives. Each state mirrors a part of the card's own flip condition (gemp card class in brackets).
        // Mind What You Have Learned [Card7_138]: flips at Jedi Test #5; the tests themselves are Jedi Training items.
        // User 2026-10-05: the apprentice is Luke, or Leia once Daughter Of Skywalker (8_8) made the objective target her;
        // the mentor is any character of ability > 2 with the apprentice (not only Yoda).
        state("7_138", "apprentice_at_dagobah", c -> {
            PhysicalCard apprentice = apprentice(c);
            PhysicalCard at = apprentice == null ? null : locationOf(c, apprentice);
            return at != null && Filters.Dagobah_location.accepts(c.gs(), c.game.getModifiersQuerying(), at);
        });
        state("7_138", "mentor_with_apprentice", c -> {
            PhysicalCard apprentice = apprentice(c);
            PhysicalCard at = apprentice == null ? null : locationOf(c, apprentice);
            return at != null && Filters.Dagobah_location.accepts(c.gs(), c.game.getModifiersQuerying(), at)
                && c.spot(c.yours(Filters.and(Filters.character, Filters.abilityMoreThan(2), Filters.not(apprentice), Filters.at(at))));
        });
        // Plan dojo-sweep (user 2026-10-13): each of your completed Jedi Tests #1-#5 (Gemp's Jedi Test keyword) on the way
        // to the flip at Test #5.
        Filter[] jediTests = {Filters.Jedi_Test_1, Filters.Jedi_Test_2, Filters.Jedi_Test_3, Filters.Jedi_Test_4, Filters.Jedi_Test_5};
        for (int n = 1; n <= 5; n++) {
            Filter jediTest = jediTests[n - 1];
            event("7_138", "jedi_test_" + n + "_passed", (c, r) -> r.getType() == EffectResult.Type.JEDI_TEST_COMPLETED
                && c.player.equals(((JediTestCompletedResult) r).getJediTest().getOwner())
                && jediTest.accepts(c.gs(), c.game.getModifiersQuerying(), ((JediTestCompletedResult) r).getJediTest()));
        }
        // Carbon Chamber Testing [Card7_296]: flips when a frozen captive is moved to Audience Chamber by its escort.
        state("7_296", "escort_captive", c -> c.spot(c.yours(Filters.escorting(Filters.captive))));
        state("7_296", "escort_frozen_captive", c -> c.spot(c.yours(Filters.escorting(Filters.frozenCaptive))));
        state("7_296", "frozen_escort_at_cloud_city_docking_bay_or_aboard_starship", c -> c.spot(c.yours(Filters.and(
            Filters.escorting(Filters.frozenCaptive), Filters.or(Filters.at(Filters.and(Filters.Cloud_City_location, Filters.docking_bay)),
                Filters.aboardAnyStarship)))));
        state("7_296", "frozen_escort_at_tatooine_site", c -> c.spot(c.yours(Filters.and(Filters.escorting(Filters.frozenCaptive),
            Filters.at(Filters.Tatooine_site)))));
        // User 2026-10-13: Ugnaughts at the Carbonite Chamber add 1 each to Carbon-Freezing destiny (Card5_105; Ugloste,
        // an Ugnaught, doubles the others, Card5_104). With the Console's +3, two make destiny > 10 likely and < 5 impossible.
        state("7_296", "ugnaughts_at_carbonite_chamber_2",
            c -> c.count(c.yours(Filters.and(Filters.Ugnaught, Filters.at(Filters.Carbonite_Chamber)))) >= 2);
        // Endor Operations [Card8_167]: Ominous Rumors and Establish Secret Base both on table. Control 3 is ESB's deploy
        // requirement (objective-learnability Amendment 6).
        Filter endorScouts = Filters.or(Filters.biker_scout, Filters.and(Filters.piloted, Filters.AT_ST));
        state("8_167", "ominous_rumors_on_table", c -> c.spot(Filters.Ominous_Rumors));
        state("8_167", "endor_control_2", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 2, Filters.Endor_site, endorScouts));
        state("8_167", "endor_control_3", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 3, Filters.Endor_site, endorScouts));
        state("8_167", "establish_secret_base_on_table", c -> c.spot(Filters.Establish_Secret_Base));
        // Set Your Course For Alderaan [Card111_006]: flips when Alderaan is 'blown away' (by Commence Primary Ignition,
        // which may only target a system the Death Star orbits; the Death Star deploys at parsec 0, orbiting nothing).
        // Plan dojo-focus rev. 3: CPI needs the Superlaser (deployable only at parsec 0, before the Death Star moves); the
        // orbit pays only with the Superlaser on (user); Gunners add 2 each; only a well set-up attempt pays.
        state("111_6", "superlaser_on_death_star", ObjectiveDojo::superlaserOn);
        state("111_6", "death_star_orbits_alderaan", c -> {
            PhysicalCard deathStar = deathStar(c);
            return deathStar != null && Title.Alderaan.equals(deathStar.getSystemOrbited()) && superlaserOn(c);
        });
        state("111_6", "gunners_on_death_star_2", c -> gunnersOnDeathStar(c) >= 2);
        event("111_6", "cpi_good_attempt_at_alderaan", (c, r) -> goodCpiAttempt(c, r) && STATES.get("111_6/death_star_orbits_alderaan").test(c));
        // User 2026-10-13: CPI's Y as a set-up step, like the orbit only with the Superlaser on, so decks that keep the
        // objective unflipped (no Superlaser, no CPI) feel no pull towards it.
        state("111_6", "death_star_sites_free_3", c -> superlaserOn(c) && deathStarSitesFree(c) >= 3);
        // Court Of The Vile Gangster [Card110_006]: two captives (or one of ability > 2) at Jabba's Palace sites.
        state("110_6", "escort_captive", c -> c.spot(c.yours(Filters.escorting(Filters.captive))));
        state("110_6", "captive_at_jabbas_palace_site", c -> c.spot(1, SPOT_CAPTIVES, Filters.and(Filters.captive, Filters.at(Filters.Jabbas_Palace_site))));
        // This Deal Is Getting Worse All The Time [Card109_012]: Dark Deal on table, occupy Bespin and Cloud City.
        state("109_12", "occupy_bespin_system", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Bespin_system));
        state("109_12", "occupy_cloud_city", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Bespin_Cloud_City));
        state("109_12", "dark_deal_on_table", c -> c.spot(Filters.Dark_Deal));
        // Local Uprising [Card7_137] / Imperial Occupation [Card7_298]: matching operatives control 3 related battleground sites.
        for (String[] planet : new String[][]{{"7_137", "subjugated"}, {"7_298", "renegade"}}) {
            boolean subjugated = planet[1].equals("subjugated");
            Filter related = Filters.and(Filters.battleground_site,
                subjugated ? Filters.Subjugated_planet_location : Filters.Renegade_planet_location);
            Filter operative = subjugated ? Filters.matchingOperativeToSubjugatedPlanet : Filters.matchingOperativeToRenegadePlanet;
            state(planet[0], "three_related_battleground_sites", c -> GameConditions.canSpotLocation(c.game, 3, related));
            state(planet[0], "operatives_control_1", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1, related, SPOT, c.yours(operative)));
            state(planet[0], "operatives_control_2", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 2, related, SPOT, c.yours(operative)));
        }
        // Rebel Strike Team [Card8_078]: during the move phase, control 3 exterior Endor sites with two Rebel scouts each.
        for (int n = 1; n <= 2; n++) {
            int count = n;
            state("8_78", "scouts_control_" + n, c -> {
                PhysicalCard self = c.self();
                Filter scout = c.yours(Filters.Rebel_scout);
                return self != null && GameConditions.controls(c.game, c.player, count, SPOT, Filters.and(Filters.exterior_Endor_site,
                    Filters.sameSiteAs(self, SPOT, Filters.and(scout, Filters.with(self, scout)))));
            });
        }
        // Quiet Mining Colony [Card109_004]: control Cloud City and two Cloud City sites (one with Lando or Lobot), opponent
        // controls no Bespin location. That last part holds at the start of every game, so it is not a milestone.
        // User 2026-10-05: Keeping The Empire Out Forever (11_20) also flips it with Bespin system occupied.
        state("109_4", "occupy_bespin_system", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Bespin_system));
        state("109_4", "control_cloud_city", c -> GameConditions.controls(c.game, c.player, SPOT, Filters.Bespin_Cloud_City));
        state("109_4", "control_cloud_city_site_1", c -> GameConditions.controls(c.game, c.player, 1, SPOT, Filters.Cloud_City_site));
        state("109_4", "control_cloud_city_site_2", c -> GameConditions.controls(c.game, c.player, 2, SPOT, Filters.Cloud_City_site));
        // There Is Good In Him [Card9_061]: flips when Luke is captured (an Imperial at Luke's site captures him). Plan
        // dojo-focus: no front milestones; the adjacency milestone was mostly the opponent's doing and refunded at the capture.
        // Bring Him Before Me [Card9_151]: flips when Luke is captured (Luke present with Vader). User 2026-10-05: reward the
        // Emperor at the Throne Room and Vader at a site of the system where Luke is present.
        state("9_151", "emperor_at_throne_room", c -> c.spot(c.yours(Filters.and(Filters.Emperor, Filters.at(Filters.Throne_Room)))));
        state("9_151", "vader_at_lukes_system", c -> {
            // Captives included (plan dojo-focus): the capture itself must not refund this just before the flip.
            PhysicalCard luke = first(c, SPOT_CAPTIVES, Filters.and(Filters.your(c.opponent), Filters.Luke));
            PhysicalCard lukeAt = luke == null ? null : locationOf(c, luke);
            String system = lukeAt == null ? null : lukeAt.getPartOfSystem() != null ? lukeAt.getPartOfSystem() : lukeAt.getBlueprint().getTitle();
            if (system == null) return false;
            return c.spot(c.yours(Filters.and(Filters.Vader, Filters.at(Filters.and(Filters.site, Filters.partOfSystem(system))))));
        });
        // Agents Of Black Sun [Card10_029] / Hunt Down And Destroy The Jedi [Card7_297].
        state("10_29", "xizor_at_battleground_site", c -> c.spot(c.yours(Filters.and(Filters.Xizor, Filters.at(Filters.battleground_site)))));
        state("7_297", "vader_at_battleground_site", c -> c.spot(c.yours(Filters.and(Filters.Vader, Filters.at(Filters.battleground_site)))));
        // Plead My Case To The Senate [Card12_088] / My Lord, Is That Legal? [Card12_179]: 3 senators at Galactic Senate.
        for (String senate : new String[]{"12_88", "12_179"}) {
            state(senate, "senators_1", c -> c.count(c.yours(Filters.and(Filters.senator, Filters.at(Filters.Galactic_Senate)))) >= 1);
            state(senate, "senators_2", c -> c.count(c.yours(Filters.and(Filters.senator, Filters.at(Filters.Galactic_Senate)))) >= 2);
        }
        // Agents In The Court [Card112_001] / My Kind Of Scum [Card112_015]: occupy 2 battleground sites (3 off Tatooine).
        for (String court : new String[]{"112_1", "112_15"}) {
            state(court, "occupy_battleground_site_2", c -> GameConditions.occupies(c.game, c.player, 2, SPOT, Filters.battleground_site));
        }
        // Massassi Base Operations [Card111_004]: control 3 Yavin 4 sites.
        state("111_4", "control_yavin_4_site_1", c -> GameConditions.controls(c.game, c.player, 1, SPOT, Filters.Yavin_4_site));
        state("111_4", "control_yavin_4_site_2", c -> GameConditions.controls(c.game, c.player, 2, SPOT, Filters.Yavin_4_site));
        // Ralltiir Operations [Card7_300] / Dantooine Base Operations [Card7_135]: control 3 sites with Imperials / Rebels.
        state("7_300", "imperials_control_1", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1, Filters.Ralltiir_site, SPOT, Filters.Imperial));
        state("7_300", "imperials_control_2", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 2, Filters.Ralltiir_site, SPOT, Filters.Imperial));
        state("7_135", "rebels_control_2", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 2, Filters.Dantooine_site, SPOT, Filters.Rebel));
        // Hidden Base [Card7_136]: the card records the battleground systems and the 'Hidden Base' system deployed by its owner.
        state("7_136", "battleground_systems_3", c -> hiddenBase(c, 3, false));
        state("7_136", "battleground_systems_5", c -> hiddenBase(c, 5, false));
        state("7_136", "hidden_base_system_deployed", c -> hiddenBase(c, 0, true));
        // ISB Operations [Card7_299]: four ISB agents on table, or ISB agents control two Rebel Base locations.
        state("7_299", "isb_agents_2", c -> c.count(Filters.ISB_agent) >= 2);
        state("7_299", "isb_agents_3", c -> c.count(Filters.ISB_agent) >= 3);
        state("7_299", "isb_control_rebel_base_1", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1, Filters.Rebel_Base_location, SPOT, Filters.ISB_agent));
        // Watch Your Step [Card10_026]: two Kessel Runs, or occupy two battlegrounds with smugglers.
        state("10_26", "smugglers_occupy_battleground_1", c -> GameConditions.occupiesWith(c.game, c.self(), c.player, 1, Filters.battleground, SPOT, Filters.smuggler));
        state("10_26", "kessel_run_1", c -> GameConditions.hasCompletedUtinniEffect(c.game, c.player, 1, Filters.Kessel_Run));
        // You Can Either Profit By This... [Card110_004]: Han on Tatooine and not a captive (he starts frozen at Audience Chamber).
        state("110_4", "character_at_audience_chamber", c -> c.spot(c.yours(Filters.and(Filters.character, Filters.at(Filters.Audience_Chamber)))));
        // Invasion [Card14_113] / We Have A Plan [Card14_052].
        state("14_113", "neimoidian_at_throne_room", c -> c.spot(c.yours(Filters.and(Filters.Neimoidian, Filters.at(Filters.Theed_Palace_Throne_Room)))));
        // We Have A Plan [Card14_052]: Amidala may not deploy to interior Naboo sites, so she deploys to the Courtyard and
        // walks in (plan dojo-focus).
        state("14_52", "amidala_at_theed_palace_site", c -> c.spot(c.yours(Filters.and(Filters.Amidala, Filters.at(Filters.Theed_Palace_site)))));
        state("14_52", "amidala_at_interior_naboo_site", c -> c.spot(c.yours(Filters.and(Filters.Amidala, Filters.at(Filters.interior_Naboo_site)))));
        state("14_52", "amidala_at_throne_room", c -> c.spot(c.yours(Filters.and(Filters.Amidala, Filters.at(Filters.Theed_Palace_Throne_Room)))));
        // Rescue The Princess [Card7_139]: flips when Leia is moved to Yavin 4: War Room (she starts imprisoned).
        // Rescue The Princess [Card7_139]: flips when Leia moves to Yavin 4: War Room. Plan dojo-focus ladder: a spy at the
        // Detention Block Corridor (I'm Here To Rescue You), Leia released, at Docking Bay 327, at Yavin 4: Docking Bay
        // (docking bay transit needs no starship).
        state("7_139", "spy_at_detention_block", c -> c.spot(c.yours(Filters.and(Filters.spy, Filters.at(Filters.Detention_Block_Corridor)))));
        state("7_139", "leia_at_docking_bay_327", c -> c.spot(1, SPOT_CAPTIVES, c.yours(Filters.and(Filters.Leia, Filters.not(Filters.captive),
            Filters.at(Filters.Docking_Bay_327)))));
        state("7_139", "leia_at_yavin_4_docking_bay", c -> c.spot(1, SPOT_CAPTIVES, c.yours(Filters.and(Filters.Leia, Filters.not(Filters.captive),
            Filters.at(Filters.Yavin_4_Docking_Bay)))));
        state("7_139", "leia_not_captive", c -> c.spot(1, SPOT_CAPTIVES, c.yours(Filters.and(Filters.Leia, Filters.not(Filters.captive)))));
        // The Hyperdrive Generator's Gone [Card12_089]: flips at 4 or more cards beneath Credits Will Do Fine, which stacks a
        // lost Force each time you initiate a Force drain (or win a battle) at Watto's Junkyard. The objective deploys
        // Credits Will Do Fine at the start, so that is no milestone (user 2026-10-10).
        state("12_89", "occupy_wattos_junkyard", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Wattos_Junkyard));
        for (int n = 1; n <= 3; n++) {
            int count = n;
            state("12_89", "credits_" + n, c -> credits(c) >= count);
        }
        // No Money, No Parts, No Deal! [Card12_180]: the flip only (Watto at Watto's Junkyard, Mos Espa occupied); it is
        // expected to come by itself, and the flip-back refund rewards holding it (user 2026-10-10).
        // Let Them Make The First Move [Card13_073]: the flip needs the opponent's Jedi at an interior Naboo battleground
        // site, so the reward goes on after the flip (user 2026-10-10). Front: a Dark Jedi at an interior Theed Palace site
        // (the ability's source) and opponent's characters made lost by the control-phase ability. Flipped (At Last We Will
        // Have Revenge): the side's lightsaber combat won, and an opponent's Jedi lost by it (difference 5 or more).
        state("13_73", "dark_jedi_at_interior_theed_palace_site", c -> c.spot(c.yours(Filters.and(Filters.Dark_Jedi,
            Filters.at(Filters.interior_Theed_Palace_site)))));
        event("13_73", "target_lost", (c, r) -> lostOpponentCharacter(c, r) && actionOnStack(c, "13_73", false));
        event("13_73", "saber_combat_won", (c, r) -> r.getType() == EffectResult.Type.LIGHTSABER_COMBAT_RESULT_DETERMINED
            && c.player.equals(((LightsaberCombatResultDeterminedResult) r).getWinner()) && actionOnStack(c, "13_73", true));
        event("13_73", "jedi_lost_to_saber_combat", (c, r) -> lostOpponentCharacter(c, r) && actionOnStack(c, "13_73", true));
        // We'll Handle This [Card13_046]: the exact mirror of Let Them Make The First Move (plan amendment 2). Flipped: Duel
        // Of The Fates.
        state("13_46", "jedi_at_interior_theed_palace_site", c -> c.spot(c.yours(Filters.and(Filters.Jedi,
            Filters.at(Filters.interior_Theed_Palace_site)))));
        event("13_46", "target_lost", (c, r) -> lostOpponentCharacter(c, r) && actionOnStack(c, "13_46", false));
        event("13_46", "saber_combat_won", (c, r) -> r.getType() == EffectResult.Type.LIGHTSABER_COMBAT_RESULT_DETERMINED
            && c.player.equals(((LightsaberCombatResultDeterminedResult) r).getWinner()) && actionOnStack(c, "13_46", true));
        event("13_46", "dark_jedi_lost_to_saber_combat", (c, r) -> lostOpponentCharacter(c, r) && actionOnStack(c, "13_46", true));
        // Bring Him Before Me flipped (Take Your Father's Place, plan amendment 2): the duel's precondition (Vader, Emperor
        // and Luke, captive included, present at the Throne Room) and the duel started by the objective (or its Leia/Kanan
        // variants). A won duel or crossover ends the game or costs the opponent Force: the outcome rewards those.
        state("9_151", "luke_at_throne_room", c -> c.spot(c.yours(Filters.and(Filters.Vader, Filters.presentAt(Filters.Throne_Room))))
            && c.spot(c.yours(Filters.and(Filters.Emperor, Filters.presentAt(Filters.Throne_Room))))
            && first(c, SPOT_CAPTIVES, Filters.and(Filters.your(c.opponent), Filters.Luke, Filters.presentAt(Filters.Throne_Room))) != null);
        event("9_151", "duel_initiated", (c, r) -> r.getType() == EffectResult.Type.DUEL_INITIATED && actionOnStack(c, "9_151", true));
        // There Is Good In Him flipped (I Can Save Him, plan amendment 2): the once-per-turn crossover attempt (its destiny
        // draw while Vader is present with Luke), the one choice left to the Light Side; a crossover ends the game.
        event("9_61", "crossover_attempt", (c, r) -> r.getType() == EffectResult.Type.DESTINY_DRAWN && actionOnStack(c, "9_61", true));

        // --- Epic Events (success is an event; it is paid once per game).
        // Commence Primary Ignition [Card2_130]: targets a planet system the Death Star orbits.
        // Plan dojo-focus rev. 3: set-up ladder (see 111_6).
        state("2_130", "superlaser_on_death_star", ObjectiveDojo::superlaserOn);
        state("2_130", "death_star_orbits_planet", c -> {
            PhysicalCard deathStar = deathStar(c);
            return deathStar != null && c.player.equals(deathStar.getOwner()) && deathStar.getSystemOrbited() != null && superlaserOn(c);
        });
        state("2_130", "gunners_on_death_star_2", c -> gunnersOnDeathStar(c) >= 2);
        event("2_130", "cpi_good_attempt", ObjectiveDojo::goodCpiAttempt);
        state("2_130", "death_star_sites_free_3", c -> superlaserOn(c) && deathStarSitesFree(c) >= 3);  // user 2026-10-13
        event("2_130", "success", (c, r) -> blownAway(r, c, Filters.planet_system));
        // Target The Main Generator [Card3_115]: the exact weapon filter of the card's attempt action.
        state("3_115", "atat_cannon_in_range", c -> {
            PhysicalCard generators = Filters.findFirstFromTopLocationsOnTable(c.game, Filters.Main_Power_Generators);
            return generators != null && c.spot(c.yours(Filters.and(Filters.AT_AT_Cannon, Filters.attachedTo(Filters.and(Filters.AT_AT, Filters.piloted)),
                Filters.canBeFiredAtLocationInRange(generators))));
        });
        event("3_115", "ttmg_attempt", (c, r) -> attempted(c, r, "3_115"));
        event("3_115", "success", (c, r) -> blownAway(r, c, Filters.Main_Power_Generators));
        // Attack Run [Card2_042]: deploys on Death Star: Trench; starfighters enter the trench at the start of the run.
        state("2_42", "attack_run_on_trench", c -> c.onTable("2_42"));
        event("2_42", "entered_trench", (c, r) -> r.getType() == EffectResult.Type.MOVED_AT_START_OF_ATTACK_RUN && c.player.equals(r.getPerformingPlayerId()));
        event("2_42", "success", (c, r) -> blownAway(r, c, Filters.Death_Star_system));
        // Epic Duel [Card5_129]: Vader present with his target (Luke or Obi-Wan), duel won by Vader's side.
        state("5_129", "vader_with_target", c -> {
            PhysicalCard self = c.self();
            return self != null && c.onTable("5_129") && c.spot(c.yours(Filters.and(Filters.Vader,
                Filters.presentWith(self, Filters.and(Filters.your(c.opponent), Filters.or(Filters.Luke, Filters.ObiWan))))));
        });
        event("5_129", "success", (c, r) -> r.getType() == EffectResult.Type.DUEL_RESULT_DETERMINED && Boolean.TRUE.equals(EPIC_DUEL.get(c.game))
            && c.player.equals(((DuelResultDeterminedResult) r).getWinner()));
        // Boonta Eve Podrace [Card11_024 LS, Card11_079 DS] and I Did It! [Card11_025].
        // User 2026-10-05: no initiation milestone for the LS Podrace (96% of games by own turn 0).
        event("11_79", "podrace_initiated", (c, r) -> r.getType() == EffectResult.Type.PODRACE_INITIATED);
        for (String podrace : new String[]{"11_24", "11_79"}) {
            event(podrace, "success", (c, r) -> r.getType() == EffectResult.Type.PODRACE_FINISHED && GameConditions.hasWonPodrace(c.game, c.player));
        }
        event("11_25", "success", (c, r) -> r.getType() == EffectResult.Type.PLAY && played(r, c.player, "11_25"));
        // Deactivate The Shield Generator [Card8_043]: control Bunker with a Rebel, attempt, Bunker 'blown away'.
        // Plan dojo-focus: Explosive Charges on Bunker (+3 each; with none the attempt cannot exceed 12).
        state("8_43", "explosive_charge_on_bunker_1", c -> c.count(c.yours(Filters.and(Filters.Explosive_Charge, Filters.attachedTo(Filters.Bunker)))) >= 1);
        state("8_43", "explosive_charge_on_bunker_2", c -> c.count(c.yours(Filters.and(Filters.Explosive_Charge, Filters.attachedTo(Filters.Bunker)))) >= 2);
        state("8_43", "control_bunker_with_rebel", c -> GameConditions.controlsWith(c.game, c.self(), c.player, Filters.Bunker, Filters.Rebel));
        event("8_43", "dtsg_attempt", (c, r) -> attempted(c, r, "8_43"));
        event("8_43", "success", (c, r) -> blownAway(r, c, Filters.Bunker));
        // That Thing's Operational [Card9_135]. DS: deployed on Reactor Core; success when the opponent loses Force to it with
        // X > 2 (X = 2 + related battleground sites occupied). Counter item for LS: attempt to 'blow away' Death Star II.
        state("9_135", "on_reactor_core", c -> c.onTable("9_135"));
        event("9_135", "success", (c, r) -> {
            if (r.getType() != EffectResult.Type.FORCE_LOST) return false;
            LostForceResult lost = (LostForceResult) r;
            PhysicalCard source = lost.getSourceCard();
            return source != null && "9_135".equals(source.getBlueprintId(true)) && c.opponent.equals(lost.getPerformingPlayerId())
                && lost.getAmountOfForceLost() >= 3;
        });
        event("9_135~counter", "ds2_attempt", (c, r) -> attempted(c, r, "9_135"));
        event("9_135~counter", "success", (c, r) -> blownAway(r, c, Filters.Death_Star_II_system));
        // Inner Strength [Card13_024]: a combat card placed under a Jedi, then used instead of a destiny draw.
        // Plan night-pool-v3 (user 2026-10-13): the placing is a milestone again (its own set-up action, which the policy
        // never took). Stacking emits no result, so the state is seen at the next table change (at the latest, the phase end).
        state("13_24", "combat_card_under_your_jedi", c -> c.spot(c.yours(Filters.and(Filters.Jedi, Filters.hasStacked(Filters.combatCard)))));
        event("13_24", "success", (c, r) -> r.getType() == EffectResult.Type.ABOUT_TO_USE_COMBAT_CARD_INSTEAD_OF_DESTINY_DRAW
            && c.player.equals(r.getPerformingPlayerId()));
        // Deep Hatred [Card13_065]: Inner Strength's Dark Side mirror (combat cards under a Dark Jedi); plan
        // dojo-three-objectives, user 2026-10-10: the same reward. Every Let Them Make The First Move deck holds it.
        state("13_65", "combat_card_under_your_jedi", c -> c.spot(c.yours(Filters.and(Filters.Dark_Jedi, Filters.hasStacked(Filters.combatCard)))));
        event("13_65", "success", (c, r) -> r.getType() == EffectResult.Type.ABOUT_TO_USE_COMBAT_CARD_INSTEAD_OF_DESTINY_DRAW
            && c.player.equals(r.getPerformingPlayerId()));
        // If The Trace Was Correct [Card11_080]: its effects turn on while Maul escorts Amidala (she is captured instead
        // of lost when forfeited at Maul's site). Plan dojo-three-objectives, user 2026-10-10: success once that holds.
        event("11_80", "success", (c, r) -> c.onTable("11_80") && c.spot(c.yours(Filters.and(Filters.Maul, Filters.escorting(Filters.Amidala)))));
        // Plan night-pool-v3 (user 2026-10-13): Maul at the same site as the opponent's Amidala, with the card on table: the
        // part the Dark Side controls (the forfeit that captures her is the opponent's choice). Captives included: the capture
        // itself must not refund it.
        state("11_80", "maul_with_opponents_amidala", c -> {
            PhysicalCard self = c.self();
            return self != null && c.onTable("11_80")
                && c.spot(c.yours(Filters.and(Filters.Maul, Filters.at(Filters.sameSiteAs(self, SPOT_CAPTIVES, Filters.and(Filters.your(c.opponent), Filters.Amidala))))));
        });

        // --- Plan dojo-v-cards (user 2026-10-25): the V-card objectives and Epic Events of the 927 V decks. Flip-only
        // objectives (Shadow Collective 213_32, They Have No Idea We're Coming 209_29 (success 0, Amendment 2), Mind What You Have Learned (V)
        // 225_53, Yavin 4 Base Operations 208_26, and by D4 Hunt Down And Destroy The Jedi (V) 213_31 and He Is The
        // Chosen One 208_25) need no predicate: the earned flip is detected generically.
        // The First Order Reigns [Card225_032] flips when Tracked Fleet is 'annihilated' (blown away): Tracked Fleet
        // [Card225_034] does that at the start of your turn if you control its system (the opponent may relocate it).
        for (String trackedFleet : new String[]{"225_32", "225_34"})
            state(trackedFleet, "control_tracked_fleet_system", c -> {
                PhysicalCard fleet = first(c, SPOT, c.yours(Filters.Tracked_Fleet));
                PhysicalCard at = fleet == null ? null : locationOf(c, fleet);
                return at != null && GameConditions.controls(c.game, c.player, SPOT, Filters.sameCardId(at));
            });
        event("225_34", "success", (c, r) -> blownAway(r, c, Filters.Tracked_Fleet));
        // Imperial Entanglements [Card201_039] / This Deal Is Getting Worse All The Time (V) [Card226_012]: control 3
        // Tatooine sites / Bespin locations while the opponent controls fewer than 3.
        state("201_39", "control_tatooine_site_2", c -> GameConditions.controls(c.game, c.player, 2, SPOT, Filters.Tatooine_site));
        state("226_12", "control_bespin_location_2", c -> GameConditions.controls(c.game, c.player, 2, SPOT, Filters.Bespin_location));
        // The Shield Will Be Down In Moments [Card222_030, Card222_014] flips when Main Power Generators are 'blown away'
        // by Target The Main Generator (V) [Card222_013]: at the start of your deploy phase, if it is at 3rd Marker or
        // lower, your piloted AT-AT's Cannon at its site may fire (if within range of Main Power Generators).
        for (String shield : new String[]{"222_30", "222_14", "222_13"}) {
            state(shield, "ttmg_at_third_marker_or_lower", c -> {
                PhysicalCard at = ttmgSite(c);
                return at != null && Filters.or(Filters.First_Marker, Filters.Second_Marker, Filters.Third_Marker)
                    .accepts(c.gs(), c.game.getModifiersQuerying(), at);
            });
            state(shield, "cannon_in_range_at_ttmg", c -> {
                PhysicalCard ttmg = c.tableCard("222_13");
                PhysicalCard generators = Filters.findFirstFromTopLocationsOnTable(c.game, Filters.Main_Power_Generators);
                return ttmg != null && generators != null && c.spot(c.yours(Filters.and(Filters.here(ttmg), Filters.AT_AT_Cannon,
                    Filters.attachedTo(Filters.and(Filters.AT_AT, Filters.piloted)), Filters.canBeFiredAtLocationInRange(generators))));
            });
            event(shield, "ttmg_fired", (c, r) -> attempted(c, r, "222_13"));
        }
        event("222_13", "success", (c, r) -> blownAway(r, c, Filters.Main_Power_Generators));
        // A Great Tactician Creates Plans [Card219_001]: in a deploy phase, Thrawn at a battleground and two artwork cards
        // stacked on Thrawn's Art Collection.
        state("219_1", "artwork_2", c -> {
            PhysicalCard collection = first(c, SPOT, c.yours(Filters.Thrawns_Art_Collection));
            return collection != null && c.gs().getStackedCards(collection).size() >= 2;
        });
        state("219_1", "thrawn_at_battleground", c -> c.spot(c.yours(Filters.and(Filters.Thrawn, Filters.at(Filters.battleground)))));
        // I Want That Map [Card208_057]: First Order characters control 2 battlegrounds (and no Resistance Agent present
        // at a battleground site: the opponent's part).
        state("208_57", "first_order_control_battleground_1", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1,
            Filters.battleground, SPOT, Filters.First_Order_character));
        // On The Verge Of Greatness [Card216_011]: Krennic or Tarkin at a Scarif battleground site, Death Star orbiting Scarif.
        state("216_11", "death_star_orbits_scarif", c -> c.spot(Filters.and(Filters.Death_Star_system, Filters.isOrbiting(Title.Scarif))));
        state("216_11", "krennic_or_tarkin_at_scarif_battleground_site", c -> c.spot(c.yours(Filters.and(
            Filters.at(Filters.Scarif_battleground_site), Filters.or(Filters.Krennic, Filters.Tarkin)))));
        // A Stunning Move [Card211_026] flips when Insidious Prisoner [Card211_011] is on an Invisible Hand site; it starts
        // on 500 Republica and follows the first character of the player who controls its site moving to a battleground site.
        state("211_26", "invisible_hand_site_on_table", c -> GameConditions.canSpotLocation(c.game, Filters.Invisible_Hand_site));
        state("211_26", "control_prisoner_site", c -> {
            PhysicalCard at = prisonerSite(c);
            return at != null && GameConditions.controls(c.game, c.player, SPOT, Filters.sameCardId(at));
        });
        event("211_26", "prisoner_left_500_republica", (c, r) -> {
            PhysicalCard at = prisonerSite(c);
            return at != null && !Filters._500_Republica.accepts(c.gs(), c.game.getModifiersQuerying(), at);
        });
        // Rescue The Princess (V) [Card215_017]: Leia occupies a Death Star site and the Death Star's power is 'shut down'
        // (A Power Loss [Card215_002]: occupy Central Core and another Death Star site while no cards stacked on it).
        // The shut-down lasts for the rest of the game, so it is a state.
        state("215_17", "leia_not_captive", c -> c.spot(1, SPOT_CAPTIVES, c.yours(Filters.and(Filters.Leia, Filters.not(Filters.captive)))));
        for (String powerLoss : new String[]{"215_17", "215_2"})
            state(powerLoss, "occupy_central_core_and_another_death_star_site", c -> GameConditions.occupies(c.game, c.player, SPOT,
                Filters.Death_Star_Central_Core) && GameConditions.occupies(c.game, c.player, 2, SPOT, Filters.Death_Star_site));
        state("215_17", "power_shut_down", c -> GameConditions.isDeathStarPowerShutDown(c.game));
        event("215_2", "success", (c, r) -> GameConditions.isDeathStarPowerShutDown(c.game));
        // Diplomatic Mission To Alderaan [Card203_019]: Stolen Data Tapes 'delivered', Rebels control a battleground site
        // and a battleground system.
        state("203_19", "tapes_delivered", c -> c.spot(Filters.delivered_Stolen_Data_Tapes));
        state("203_19", "rebels_control_battleground_site", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1,
            Filters.battleground_site, SPOT, Filters.Rebel));
        state("203_19", "rebels_control_battleground_system", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 1,
            Filters.battleground_system, SPOT, Filters.Rebel));
        // The Hidden Path [Card226_028]: Jedi occupy two non-Mapuzo sites (Jedi survivors deploy only to Safehouse).
        state("226_28", "jedi_occupy_non_mapuzo_site_1", c -> GameConditions.occupiesWith(c.game, c.self(), c.player, 1,
            Filters.and(Filters.not(Filters.Mapuzo_location), Filters.site), SPOT, Filters.Jedi));
        // Old Allies [Card204_032]: control Jakku system and occupy 2 Jakku battleground sites (or the reverse).
        state("204_32", "occupy_jakku_system", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Jakku_system));
        state("204_32", "occupy_jakku_battleground_site_1", c -> GameConditions.occupies(c.game, c.player, 1, SPOT, Filters.Jakku_battleground_site));
        // The Galaxy May Need A Legend [Card211_036]: may flip when Luke is on Ahch-To and a battle with a Resistance
        // character was just initiated.
        state("211_36", "luke_on_ahch_to", c -> c.spot(c.yours(Filters.and(Filters.Luke, Filters.on(Title.Ahch_To)))));
        // Zero Hour [Card219_048]: Rebels control 3 Lothal locations (or Phoenix Squadron characters occupy 3), the
        // opponent controls none.
        state("219_48", "lothal_locations_2", c -> GameConditions.controlsWith(c.game, c.self(), c.player, 2, Filters.Lothal_location,
            SPOT, Filters.Rebel) || GameConditions.occupiesWith(c.game, c.self(), c.player, 2, Filters.Lothal_location, SPOT,
            Filters.Phoenix_Squadron_character));
        // The Hyperdrive Generator's Gone (V) [Card210_025]: as the original 12_89 (4 cards beneath Credits Will Do Fine).
        state("210_25", "occupy_wattos_junkyard", c -> GameConditions.occupies(c.game, c.player, SPOT, Filters.Wattos_Junkyard));
        for (int n = 2; n <= 3; n++) {
            int count = n;
            state("210_25", "credits_" + n, c -> credits(c) >= count);
        }
        // D3 (user 2026-10-25): no reward for a flip the opponent makes. The Empire Knows We're Here [Card222_027] flips when
        // the opponent occupies your Hoth location. Front: two battleground marker sites (its own retrieval condition).
        // Flipped (Prepare For Ground Assault, back until the opponent leaves): battles won on Hoth.
        state("222_27", "occupy_battleground_marker_site_2", c -> GameConditions.occupies(c.game, c.player, 2, SPOT,
            Filters.and(Filters.battleground, Filters.marker_site)));
        event("222_27", "won_battle_at_hoth_location", (c, r) -> TriggerConditions.wonBattleAt(c.game, r, c.player, Filters.Hoth_location));
        // Hunt For The Droid General [Card221_067] flips by itself after the start (Grievous Will Run And Hide on it unless
        // Grievous is alone at a battleground). Flipped (He's A Coward [Card221_067_BACK]): its X = battlegrounds your
        // [Clone Army] cards occupy; X > 1 and X > 2 unlock its battle actions.
        for (int n = 2; n <= 3; n++) {
            int count = n;
            state("221_67", "clone_army_occupy_battleground_" + n, c -> {
                PhysicalCard self = c.self();
                return self != null && Filters.filterTopLocationsOnTable(c.game, Filters.and(Filters.battleground,
                    Filters.occupiesWith(c.player, self, Filters.icon(Icon.CLONE_ARMY)))).size() >= count;
            });
        }
        // Restore Freedom To The Galaxy [Card208_017]: a Force drain at a battleground system (not 'liberated') lets it draw
        // destiny, +1 per piloted unique snub fighter present; > 5 'liberates' the system.
        event("208_17", "drain_at_battleground_system_with_snub_fighter", (c, r) -> {
            if (!TriggerConditions.forceDrainInitiatedBy(c.game, r, c.player, Filters.and(Filters.battleground_system,
                    Filters.except(Filters.liberated_system)))) return false;
            PhysicalCard system = ((ForceDrainInitiatedResult) r).getLocation();
            return c.spot(c.yours(Filters.and(Filters.snub_fighter, Filters.unique, Filters.piloted, Filters.presentAt(Filters.sameCardId(system)))));
        });
        event("208_17", "success", (c, r) -> c.onTable("208_17") && GameConditions.canSpotLocation(c.game, Filters.liberated_system));
        // D2: Epic Events with no success state of their own (rule modifiers, agendas, card supplies for their objective):
        // coverage only, success 0, never paid.
        for (String passive : new String[]{"217_20", "225_36", "215_25", "211_11", "217_50", "221_66", "217_29", "226_14",
                "225_57", "225_47", "216_24"})
            event(passive, "success", (c, r) -> false);

        // --- Jedi Training: each test is deployed (with its targets), then completed. 225_48: It Is The Future You See (V).
        for (String test : new String[]{"4_77", "4_75", "4_76", "4_79", "4_78", "9_55", "225_48"}) {
            state(test, "test_on_table", c -> c.onTable(test));
            event(test, "success", (c, r) -> r.getType() == EffectResult.Type.JEDI_TEST_COMPLETED
                && test.equals(((JediTestCompletedResult) r).getJediTest().getBlueprintId(true))
                && c.player.equals(((JediTestCompletedResult) r).getJediTest().getOwner()));
        }
        validateRegistry();
    }

    /** Plan dojo-v-cards: the site your Target The Main Generator (V) is on, or null. */
    private static PhysicalCard ttmgSite(Ctx c) {
        PhysicalCard ttmg = c.tableCard("222_13");
        return ttmg == null ? null : locationOf(c, ttmg);
    }

    /** Plan dojo-v-cards: the site Insidious Prisoner is on (either side's), or null. */
    private static PhysicalCard prisonerSite(Ctx c) {
        PhysicalCard prisoner = first(c, SPOT_CAPTIVES, Filters.Insidious_Prisoner);
        return prisoner == null ? null : locationOf(c, prisoner);
    }

    /** Cards beneath your Credits Will Do Fine (The Hyperdrive Generator's Gone). */
    private static int credits(Ctx c) {
        PhysicalCard credits = first(c, SPOT, c.yours(Filters.Credits_Will_Do_Fine));
        return credits == null ? 0 : c.gs().getStackedCards(credits).size();
    }

    /** An opponent's character was just lost from table. */
    private static boolean lostOpponentCharacter(Ctx c, EffectResult r) {
        if (r.getType() != EffectResult.Type.LOST_FROM_TABLE) return false;
        PhysicalCard card = ((LostFromTableResult) r).getCard();
        return card != null && c.opponent.equals(card.getOwner()) && card.getBlueprint().getCardCategory() == CardCategory.CHARACTER;
    }

    private static final java.lang.reflect.Field ENVIRONMENT_STACK = field(DefaultActionsEnvironment.class, "_actionStack");
    private static final java.lang.reflect.Field STACK_ACTIONS = field(ActionStack.class, "_actionStack");

    private static java.lang.reflect.Field field(Class<?> type, String name) {
        try {
            java.lang.reflect.Field f = type.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * True while an action of this side's card (front or flipped side) is being processed: a result's cause, which gemp's
     * results do not carry. The observer sees results while the action that caused them is still on gemp's action stack.
     */
    private static boolean actionOnStack(Ctx c, String blueprint, boolean flipped) {
        try {
            ActionStack stack = (ActionStack) ENVIRONMENT_STACK.get(c.game.getActionsEnvironment());
            if (stack == null) return false;
            for (Object o : (java.util.Stack<?>) STACK_ACTIONS.get(stack)) {
                Action action = (Action) o;
                PhysicalCard source = action.getActionSource();
                if (source != null && blueprint.equals(source.getBlueprintId(true)) && source.isFlipped() == flipped
                        && c.player.equals(source.getOwner()) && c.player.equals(action.getPerformingPlayer())) return true;
            }
            return false;
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Mind What You Have Learned's apprentice: Luke, or Leia when Daughter Of Skywalker redirected the objective. */
    private static PhysicalCard apprentice(Ctx c) {
        PhysicalCard objective = c.gs().getObjectivePlayed(c.player);
        boolean leia = objective != null && GameConditions.hasGameTextModification(c.game, objective,
            ModifyGameTextType.MIND_WHAT_YOU_HAVE_LEARNED_SAVE_YOU_IT_CAN__TARGETS_LEIA_INSTEAD_OF_LUKE);
        return first(c, SPOT, c.yours(leia ? Filters.Leia : Filters.Luke));
    }

    private static boolean played(EffectResult r, String player, String blueprint) {
        PhysicalCard card = ((PlayCardResult) r).getPlayedCard();
        return card != null && player.equals(card.getOwner()) && blueprint.equals(card.getBlueprintId(true));
    }

    /** Hidden Base cards whose deploy record has been seen; the card clears it once the flip condition is met. */
    private static final Set<PhysicalCard> HIDDEN_BASE_RECORDED = java.util.Collections.synchronizedSet(
        java.util.Collections.newSetFromMap(new java.util.WeakHashMap<>()));

    private static boolean hiddenBase(Ctx c, int systems, boolean hiddenBaseSystem) {
        PhysicalCard objective = c.gs().getObjectivePlayed(c.player);
        if (objective == null) return false;
        if (objective.getWhileInPlayData() == null || objective.getWhileInPlayData().getStringBooleanMap() == null)
            return HIDDEN_BASE_RECORDED.contains(objective);  // cleared after a recorded deploy = the condition was met
        HIDDEN_BASE_RECORDED.add(objective);
        Map<String, Boolean> deployed = objective.getWhileInPlayData().getStringBooleanMap();
        if (hiddenBaseSystem) {
            List<PhysicalCard> stacked = c.gs().getStackedCards(objective);
            return !stacked.isEmpty() && deployed.containsKey(stacked.get(0).getBlueprint().getTitle());
        }
        int battlegrounds = 0;
        for (Boolean battleground : deployed.values()) if (Boolean.TRUE.equals(battleground)) battlegrounds++;
        return battlegrounds >= systems;
    }

    /** Every configured milestone has a predicate of its kind, every non-objective item a success, and nothing is unused. */
    static void validateRegistry() {
        Set<String> used = new HashSet<>();
        List<String> problems = new ArrayList<>();
        for (Item item : ITEMS.values()) {
            for (Milestone m : item.milestones) {
                String key = item.key + "/" + m.name;
                if (m.afterFlip && (!item.objective() || m.necessary))
                    problems.add(key + ": a 'flipped' milestone needs an objective item and cannot be necessary");
                if (m.repeat < 1 || (m.repeat > 1 && !m.event)) problems.add(key + ": repeat needs an event milestone and n >= 1");
                if ((m.event ? EVENTS : STATES).containsKey(key)) used.add(key);
                else problems.add("no " + (m.event ? "event" : "state") + " predicate for " + key);
            }
            if (!item.objective()) {
                String key = item.key + "/success";
                if (EVENTS.containsKey(key)) used.add(key);
                else problems.add("no success predicate for " + key);
            }
        }
        for (String key : STATES.keySet()) if (!used.contains(key)) problems.add("unused state predicate " + key);
        for (String key : EVENTS.keySet()) if (!used.contains(key)) problems.add("unused event predicate " + key);
        for (Item item : ITEMS.values()) BY_BLUEPRINT.computeIfAbsent(item.blueprint, k -> new ArrayList<>()).add(item);
        if (!problems.isEmpty()) throw new IllegalStateException("Dojo registry: " + String.join("; ", problems));
    }

    // ---------------------------------------------------------------- coverage gate

    /** Objectives, Epic Events and Jedi Tests in a deck (cards inside and outside the deck). */
    static Set<String> dojoCards(SwccgDeck deck, SwccgCardBlueprintLibrary library) {
        Set<String> result = new LinkedHashSet<>();
        List<String> cards = new ArrayList<>(deck.getCards());
        cards.addAll(deck.getCardsOutsideDeck());
        for (String id : cards) {
            SwccgCardBlueprint blueprint = library.getSwccgoCardBlueprint(id);
            if (blueprint != null && CATEGORIES.contains(blueprint.getCardCategory())) result.add(id);
        }
        return result;
    }

    /**
     * Fatal coverage gate: every objective, Epic Event and Jedi Test in the decks must have a Dojo reward. Training (and
     * evaluation) with a deck that lacks one stops here, naming the cards and the decks that use them.
     */
    public static void require(Map<String, SwccgDeck> decks, SwccgCardBlueprintLibrary library) {
        Map<String, Set<String>> missing = new TreeMap<>();
        for (Map.Entry<String, SwccgDeck> deck : decks.entrySet())
            for (String id : dojoCards(deck.getValue(), library))
                if (!BY_BLUEPRINT.containsKey(id)) missing.computeIfAbsent(id, k -> new TreeSet<>()).add(deck.getKey());
        if (missing.isEmpty()) return;
        StringBuilder message = new StringBuilder("Dojo coverage: no reward for ").append(missing.size()).append(" card(s).");
        for (Map.Entry<String, Set<String>> entry : missing.entrySet()) {
            SwccgCardBlueprint blueprint = library.getSwccgoCardBlueprint(entry.getKey());
            message.append("\n  supply the Dojo reward for ").append(blueprint.getTitle()).append(" (").append(entry.getKey())
                .append(", ").append(blueprint.getCardCategory()).append(") in data/dojo/rewards.json + ObjectiveDojo.java before training"
                + " with these decks: ").append(String.join(", ", entry.getValue()));
        }
        throw new IllegalStateException(message.toString());
    }

    /** Number of configured Dojo items (loading the class validates the registry). */
    public static int itemCount() {
        return ITEMS.size();
    }

    /** The coverage gate on deck files (as TrainGen loads them). */
    public static void requireFiles(List<String> files, SwccgCardBlueprintLibrary library) {
        Map<String, SwccgDeck> decks = new TreeMap<>();
        for (String file : files) decks.computeIfAbsent(file, k -> DeckLoader.loadDeck(new java.io.File(k), library));
        require(decks, library);
    }

    // ---------------------------------------------------------------- per game

    private static final class Track {
        final Item item;
        final double[] paid;
        final boolean[] held;
        /** Times each event milestone has paid (repeat). */
        final int[] hits;
        boolean successHeld;
        double successPaid;
        boolean autoFlip;
        /** Own turn of the first flip to the back side (-1: not flipped yet); 'flipped' milestones decay from it. */
        int flipTurn = -1;

        Track(Item item) {
            this.item = item;
            paid = new double[item.milestones.size()];
            held = new boolean[item.milestones.size()];
            hits = new int[item.milestones.size()];
        }
    }

    /** The Dojo state of one game. Created by {@link GameStats#attach}; fed every effect result the observer sees. */
    public static final class Game {
        private final String gameId;
        private final String[] players;
        private final String[] sides = {"ds", "ls"};
        private final List<List<Track>> tracks = List.of(new ArrayList<>(), new ArrayList<>());
        private final double[] total = new double[2];
        private final List<String> events = new ArrayList<>();

        public Game(String gameId, String ds, String ls, Map<String, SwccgDeck> decks, SwccgCardBlueprintLibrary library) {
            this.gameId = gameId;
            players = new String[]{ds, ls};
            List<Set<String>> cards = List.of(dojoCards(decks.get(ds), library), dojoCards(decks.get(ls), library));
            for (int s = 0; s < 2; s++) {
                for (String id : cards.get(s)) {
                    List<Item> items = BY_BLUEPRINT.get(id);
                    if (items == null) {
                        // Plan v-cards: evaluation without the coverage gate (-Dtraingen.dojoCoverage=false) gives such a
                        // card no Dojo track; training keeps the gate.
                        if ("false".equals(System.getProperty("traingen.dojoCoverage"))) continue;
                        throw new IllegalStateException("Dojo coverage: no reward for " + id + " (see ObjectiveDojo.require)");
                    }
                    for (Item item : items) {
                        boolean excluded = false;
                        for (String other : item.excludedBy) excluded |= cards.get(s).contains(other);
                        if (!item.counter && !excluded) tracks.get(s).add(new Track(item));
                    }
                }
                // Counter items reward this side for answering a card in the opposing deck.
                for (String id : cards.get(1 - s)) {
                    List<Item> items = BY_BLUEPRINT.get(id);
                    if (items != null) for (Item item : items) if (item.counter) tracks.get(s).add(new Track(item));
                }
            }
        }

        public void observe(SwccgGame game, EffectResult result) {
            if (result.getType() == EffectResult.Type.DUEL_INITIATED) EPIC_DUEL.put(game, ((DuelInitiatedResult) result).isEpicDuel());
            for (int s = 0; s < 2; s++) {
                Ctx c = new Ctx(game, players[s], players[1 - s]);
                for (Track t : tracks.get(s)) events(c, s, t, result);
            }
            if (result.getType() == EffectResult.Type.DOUBLE_SIDED_CARD_FLIPPED) flipped(game, (DoubleSidedCardFlippedResult) result);
            if (result.getType() == EffectResult.Type.DUEL_ENDING || result.getType() == EffectResult.Type.DUEL_CANCELED) EPIC_DUEL.remove(game);
            if (TriggerConditions.isTableChanged(game, result) || result.getType() == EffectResult.Type.FROZEN
                    || result.getType() == EffectResult.Type.TRANSFERRED_CAPTIVE_TO_NEW_ESCORT
                    || result.getType() == EffectResult.Type.JEDI_TEST_COMPLETED)
                for (int s = 0; s < 2; s++) {
                    Ctx c = new Ctx(game, players[s], players[1 - s]);
                    for (Track t : tracks.get(s)) states(c, s, t);
                }
        }

        /** State milestones only (a result offered again after later effects). */
        public void restate(SwccgGame game, EffectResult result) {
            if (!TriggerConditions.isTableChanged(game, result)) return;
            for (int s = 0; s < 2; s++) {
                Ctx c = new Ctx(game, players[s], players[1 - s]);
                for (Track t : tracks.get(s)) states(c, s, t);
            }
        }

        private void events(Ctx c, int s, Track t, EffectResult result) {
            for (int i = 0; i < t.held.length; i++) {
                Milestone m = t.item.milestones.get(i);
                if (!m.event || t.hits[i] >= m.repeat || (t.item.objective() && !(m.afterFlip ? flipped(c) : front(c)))) continue;
                if (EVENTS.get(t.item.key + "/" + m.name).test(c, result)) {
                    t.held[i] = true;
                    t.hits[i]++;
                    double value = value(c, t, m);
                    t.paid[i] += value;
                    pay(c, s, value, t.item, m.name, "gain");
                }
            }
            if (!t.item.objective() && !t.successHeld && EVENTS.get(t.item.key + "/success").test(c, result)) {
                t.successHeld = true;
                t.successPaid = t.item.success;
                pay(c, s, t.successPaid, t.item, "success", "success");
            }
        }

        /**
         * A milestone's value now: decayed from the start of the game, or for a 'flipped' milestone from the first flip
         * (user 2026-10-10), so a late flip still leaves its payoff worth reaching. A flip back does not restart the clock.
         */
        private static double value(Ctx c, Track t, Milestone m) {
            if (!m.afterFlip) return m.weight * t.item.decay(c.ownTurn());
            if (t.flipTurn < 0) t.flipTurn = c.ownTurn();  // the flip result not seen yet
            return m.weight * t.item.decay(c.ownTurn() - t.flipTurn + 1);
        }

        /** True while this side's objective is in play, front side up (objective milestones only move then). */
        private static boolean front(Ctx c) {
            PhysicalCard objective = c.gs().getObjectivePlayed(c.player);
            return objective != null && !objective.isFlipped() && objective.getZone() != null && objective.getZone().isInPlay();
        }

        /** True while this side's objective is in play, flipped ("when": "flipped" milestones only move then). */
        private static boolean flipped(Ctx c) {
            PhysicalCard objective = c.gs().getObjectivePlayed(c.player);
            return objective != null && objective.isFlipped() && objective.getZone() != null && objective.getZone().isInPlay();
        }

        /** Refunds the held 'flipped' state milestones (flip-back, or the objective left play). */
        private void afterFlipLoss(Ctx c, int s, Track t) {
            for (int i = 0; i < t.held.length; i++) {
                Milestone m = t.item.milestones.get(i);
                if (m.afterFlip && !m.event && t.held[i]) loss(c, s, t, i);
            }
        }

        private void states(Ctx c, int s, Track t) {
            boolean flipped = false;
            if (t.item.objective()) {
                PhysicalCard objective = c.gs().getObjectivePlayed(c.player);
                boolean inPlay = objective != null && objective.getZone() != null && objective.getZone().isInPlay();
                if (t.successHeld && !inPlay) {
                    // The flipped objective left play: the flip is refunded; its front milestones stay frozen as held.
                    t.successHeld = false;
                    pay(c, s, -t.successPaid, t.item, "success", "lost");
                    t.successPaid = 0;
                    afterFlipLoss(c, s, t);
                    return;
                }
                if (!inPlay && !t.successHeld) {
                    for (int i = 0; i < t.held.length; i++) if (t.held[i] && !t.item.milestones.get(i).event) loss(c, s, t, i);
                    return;
                }
                flipped = inPlay && objective.isFlipped();
            }
            for (int i = 0; i < t.held.length; i++) {
                Milestone m = t.item.milestones.get(i);
                if (m.event) continue;
                if (m.afterFlip != flipped) {
                    // Front milestones stay frozen while flipped; 'flipped' ones are refunded once front side up again.
                    if (m.afterFlip && t.held[i]) loss(c, s, t, i);
                    continue;
                }
                // Without any own card in play no milestone can hold (and gemp's spotting needs a source card).
                boolean now = c.self() != null && STATES.get(t.item.key + "/" + m.name).test(c);
                if (now && !t.held[i]) {
                    t.held[i] = true;
                    t.paid[i] = value(c, t, m);
                    pay(c, s, t.paid[i], t.item, m.name, "gain");
                } else if (!now && t.held[i]) {
                    loss(c, s, t, i);
                }
            }
        }

        private void loss(Ctx c, int s, Track t, int i) {
            t.held[i] = false;
            pay(c, s, -t.paid[i], t.item, t.item.milestones.get(i).name, "loss");
            t.paid[i] = 0;
        }

        private void flipped(SwccgGame game, DoubleSidedCardFlippedResult result) {
            PhysicalCard card = result.getCardFlipped();
            if (card.getBlueprint().getCardCategory() != CardCategory.OBJECTIVE) return;
            int s = players[0].equals(card.getOwner()) ? 0 : players[1].equals(card.getOwner()) ? 1 : -1;
            if (s < 0) return;
            Ctx c = new Ctx(game, players[s], players[1 - s]);
            for (Track t : tracks.get(s)) {
                if (!t.item.objective() || !t.item.blueprint.equals(card.getBlueprintId(true))) continue;
                boolean toBack = card.isFlipped();
                if (toBack && t.flipTurn < 0) t.flipTurn = c.ownTurn();
                if (toBack && !t.successHeld) {
                    if (game.getGameState().getCurrentPhase() == Phase.PLAY_STARTING_CARDS) {
                        // Automatic flips while starting cards are played are not earned (e.g. no Rebel for Carbon Chamber Testing).
                        t.autoFlip = true;
                        record(c, s, 0, t.item, "success", "auto_flip");
                        continue;
                    }
                    // The card re-checks its flip condition after effects whose results this observer has already
                    // seen, so a milestone can first hold at the flip itself: evaluate the ladder once more first.
                    for (int i = 0; i < t.held.length; i++) {
                        Milestone m = t.item.milestones.get(i);
                        if (m.event || m.afterFlip || t.held[i] || c.self() == null || !STATES.get(t.item.key + "/" + m.name).test(c)) continue;
                        t.held[i] = true;
                        t.paid[i] = m.weight * t.item.decay(c.ownTurn());
                        pay(c, s, t.paid[i], t.item, m.name, "gain");
                    }
                    t.successHeld = true;
                    t.successPaid = t.item.success;
                    pay(c, s, t.successPaid, t.item, "success", "success");
                } else if (!toBack && t.successHeld) {
                    t.successHeld = false;
                    pay(c, s, -t.successPaid, t.item, "success", "flip_back");
                    t.successPaid = 0;
                    afterFlipLoss(c, s, t);
                }
            }
        }

        private void pay(Ctx c, int s, double value, Item item, String milestone, String kind) {
            total[s] += value;
            record(c, s, value, item, milestone, kind);
            if (gameId != null && value != 0) BinaryTrajectoryWriter.markDojo(gameId, players[s], value, item.key, milestone);
        }

        private void record(Ctx c, int s, double value, Item item, String milestone, String kind) {
            GameState gs = c.gs();
            events.add(String.format(Locale.ROOT, "{\"side\":\"%s\",\"item\":\"%s\",\"m\":\"%s\",\"ev\":\"%s\",\"own_turn\":%d,\"turn\":%d,\"v\":%.6f}",
                sides[s], item.key, milestone, kind, c.ownTurn(),
                gs.getPlayersLatestTurnNumber(gs.getDarkPlayer()) + gs.getPlayersLatestTurnNumber(gs.getLightPlayer()), value));
        }

        /** Stats JSON: active items per side, every gain/loss/success event, and the total reward paid per side. */
        public String json() {
            StringBuilder b = new StringBuilder("{");
            for (int s = 0; s < 2; s++) {
                List<String> keys = new ArrayList<>();
                for (Track t : tracks.get(s)) keys.add("\"" + t.item.key + "\"");
                b.append(String.format(Locale.ROOT, "\"%s\":{\"items\":[%s],\"total\":%.6f},", sides[s], String.join(",", keys), total[s]));
            }
            if (tracks.get(0).isEmpty()) b.append("\"ds_standard\":true,");
            if (tracks.get(1).isEmpty()) b.append("\"ls_standard\":true,");
            return b.append("\"events\":[").append(String.join(",", events)).append("]}").toString();
        }
    }

    private ObjectiveDojo() {
    }
}
