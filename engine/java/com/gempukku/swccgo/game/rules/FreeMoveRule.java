package com.gempukku.swccgo.game.rules;

import com.gempukku.swccgo.common.Phase;
import com.gempukku.swccgo.game.PhysicalCard;
import com.gempukku.swccgo.game.SwccgGame;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AbstractAwaitingDecision;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;
import com.gempukku.swccgo.logic.decisions.CardActionSelectionDecision;
import com.gempukku.swccgo.logic.decisions.CardsSelectionDecision;
import com.gempukku.swccgo.logic.timing.Action;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan free-move-rule: within a move phase, a free (unlimited) move is not
 * offered if it would put a card back into a position it already held earlier in that phase. Position = location the
 * card is at + what it is aboard / its escort + its role (pilot, passenger, escorted, imprisoned, unattended, plain).
 * Covered: disembark, embark and transfer between docked starships (filtered at the action choice from all their
 * possible targets and slots, and at their target choices, which are never emptied), capacity-slot moves and the four captive actions. A move menu whose every
 * action is a return is emptied down to Pass (plan free-move-churn, user 2026-10-14).
 *
 * Plan permanent-play-rules (user 2026-10-10, "for us this is a permanent rule"): always on in training, for every
 * player; production Brainiac applies it to its own seat only. No flag.
 *
 * Plan relocation-rule (user 2026-10-23): in every phase, a card's own top-level "Relocate ..." game-text action (e.g.
 * Stardust, Stolen Data Tapes: free, unlimited, "at any time") is not offered again once that card was relocated by such
 * an action in the current phase. Otherwise a policy can move the Effect between two holders without end.
 *
 * Plan capacity-slot-rule (user 2026-10-23): the position rule above also runs in the deploy phase, since Gemp offers
 * the capacity-slot move there too (e.g. Sergeant Irol swapping pilot / passenger of Tempest 1 without end). Positions
 * are tracked per phase; no other free move is offered in the deploy phase.
 *
 * Plan churn-and-drain-cancel (user 2026-10-24): two more move-phase actions are covered. A weapon / device transfer
 * (position = the card it is attached to) may not return it to an earlier holder in the phase; a ship-dock (one action:
 * dock, move cards, undock) is not offered again for a pair of starships that ship-docked earlier in the phase. Both are
 * filtered at the action menu (from a superset of their possible targets, so an action is only hidden when every
 * possible target is a return) and at their target choice.
 *
 * Plan mpg-retry-rule (user 2026-10-24): a location's deploy that was chosen in the current phase is not offered again
 * for that card in the phase (if it is offered again, the attempt failed: e.g. Hoth: Main Power Generators, which first
 * deploys the 4th Marker from the Reserve Deck, fails when it is not there and was retried up to 163 times a turn).
 *
 * Plan dock-churn-and-loop-watch-run (user 2026-10-24): a transfer between docked starships is judged per starship, not
 * per slot (Gemp reads the follow-up Pilot / Passenger or Vehicle / Starship choice by index, so it cannot be filtered):
 * it is hidden, and its target choice drops a starship, when the card was aboard that starship earlier in the phase.
 * Starships and vehicles in a cargo hold have a position too (holder = the carrying starship, role "cargo").
 */
public final class FreeMoveRule {
    private static final Map<SwccgGame, State> STATES = Collections.synchronizedMap(new WeakHashMap<>());
    private static final Map<String, AtomicLong> REMOVED = new ConcurrentHashMap<>();
    private static final Field ACTIONS, PARAMS, CARDS;
    private static final String[] INDEXED = {"actionId", "cardId", "blueprintId", "testingText", "backSideTestingText",
        "horizontal", "actionText"};

    static {
        try {
            ACTIONS = CardActionSelectionDecision.class.getDeclaredField("_actions");
            PARAMS = AbstractAwaitingDecision.class.getDeclaredField("_params");
            CARDS = CardsSelectionDecision.class.getDeclaredField("_physicalCards");
            ACTIONS.setAccessible(true);
            PARAMS.setAccessible(true);
            CARDS.setAccessible(true);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                StringBuilder b = new StringBuilder("free-move-rule summary (removed options by kind):\n");
                REMOVED.forEach((k, n) -> b.append(k).append('\t').append(n).append('\n'));
                System.err.print(b);
            }));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private static final class State {
        String relocationKey = "";
        final Set<Integer> relocated = new HashSet<>();
        final Set<Integer> deployTried = new HashSet<>();  // location cards whose deploy was chosen this phase
        String phaseKey = "";
        final Map<Integer, Set<String>> visited = new HashMap<>();
        PhysicalCard pendingEmbark;
        PhysicalCard pendingDisembark;
        boolean pendingDockedTransfer;  // pendingDisembark is a transfer between docked starships
        PhysicalCard pendingTransfer;
        PhysicalCard pendingDock;
        final Map<Integer, Set<Integer>> docked = new HashMap<>();  // starship -> starships it ship-docked with this phase
    }

    // Plan free-move-churn (user 2026-10-14, permanent): A, a move menu whose every action is a return is emptied down to
    // Pass; B, Embark is filtered at the action menu from all its possible targets; C, Disembark predicts every possible
    // destination (the site or a vehicle / starship) and its target choice is filtered.
    private static final Field DESTINATION_EMBARK, DESTINATION_DISEMBARK, TARGET_FILTERS, TARGET_OVERRIDES, TRANSFER_DESTINATIONS;
    // Fix F (user 2026-10-14): a transfer between docked starships is filtered like Embark / Disembark.
    private static final Set<String> TARGETED = Set.of("EmbarkAction", "DisembarkAction", "TransferBetweenDockedStarshipsAction");
    // Plan churn-and-drain-cancel: the moved card of a transfer / ship-dock (their action source is null).
    private static final Field TRANSFERRED, DOCKING, TARGETING;

    static {
        try {
            DESTINATION_EMBARK = com.gempukku.swccgo.cards.actions.EmbarkAction.class.getDeclaredField("_chooseDestinationEffect");
            DESTINATION_DISEMBARK = com.gempukku.swccgo.cards.actions.DisembarkAction.class.getDeclaredField("_chooseDestinationEffect");
            TARGET_FILTERS = com.gempukku.swccgo.logic.effects.choose.ChooseCardsOnTableEffect.class.getDeclaredField("_filters");
            TARGET_OVERRIDES = com.gempukku.swccgo.logic.effects.choose.ChooseCardsOnTableEffect.class.getDeclaredField("_spotOverrides");
            TRANSFER_DESTINATIONS = com.gempukku.swccgo.cards.actions.TransferBetweenDockedStarshipsAction.class.getDeclaredField("_possibleDestinations");
            TRANSFERRED = com.gempukku.swccgo.cards.actions.TransferDeviceOrWeaponAction.class.getDeclaredField("_cardToTransfer");
            DOCKING = com.gempukku.swccgo.cards.actions.ShipdockingAction.class.getDeclaredField("_starship1");
            TARGETING = com.gempukku.swccgo.logic.actions.AbstractAction.class.getDeclaredField("_targetingCosts");
            for (Field f : new Field[]{DESTINATION_EMBARK, DESTINATION_DISEMBARK, TARGET_FILTERS, TARGET_OVERRIDES, TRANSFER_DESTINATIONS,
                    TRANSFERRED, DOCKING, TARGETING})
                f.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /** The cards Gemp would offer as the destination of an Embark / Disembark / docked transfer (null if unreadable). */
    @SuppressWarnings("unchecked")
    private static java.util.Collection<PhysicalCard> targets(SwccgGame game, Action action) {
        try {
            if (action instanceof com.gempukku.swccgo.cards.actions.TransferBetweenDockedStarshipsAction)
                return new ArrayList<>((List<PhysicalCard>) TRANSFER_DESTINATIONS.get(action));
            Field destination = action instanceof com.gempukku.swccgo.cards.actions.EmbarkAction ? DESTINATION_EMBARK
                : action instanceof com.gempukku.swccgo.cards.actions.DisembarkAction ? DESTINATION_DISEMBARK : null;
            if (destination == null) return null;
            Object effect = destination.get(action);
            if (!(effect instanceof com.gempukku.swccgo.logic.effects.choose.ChooseCardsOnTableEffect)) return null;
            com.gempukku.swccgo.common.Filterable filters = (com.gempukku.swccgo.common.Filterable) TARGET_FILTERS.get(effect);
            Map<com.gempukku.swccgo.common.InactiveReason, Boolean> overrides =
                (Map<com.gempukku.swccgo.common.InactiveReason, Boolean>) TARGET_OVERRIDES.get(effect);
            return com.gempukku.swccgo.filters.Filters.filterActive(game, action.getActionSource(), false, overrides, filters);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Every position a card may end in when moving onto / off to {@code target}: a location gives "on it, plain"; a
     * vehicle or starship gives each capacity slot that has room for the card. Null when unknown (no prediction: keep).
     */
    private static Set<String> results(SwccgGame game, PhysicalCard card, PhysicalCard target) {
        GameState gs = game.getGameState();
        com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying mq = game.getModifiersQuerying();
        Set<String> out = new HashSet<>();
        if (com.gempukku.swccgo.filters.Filters.location.accepts(gs, mq, target)) {
            out.add(position(target.getCardId(), 0, "plain"));
            return out;
        }
        if (!com.gempukku.swccgo.filters.Filters.character.accepts(gs, mq, card)) return null;  // vehicles in cargo bays etc.
        int loc = location(game, target);
        if (com.gempukku.swccgo.filters.Filters.hasAvailablePilotCapacity(card).accepts(gs, mq, target)) out.add(position(loc, target.getCardId(), "pilot"));
        if (com.gempukku.swccgo.filters.Filters.hasAvailablePassengerCapacity(card).accepts(gs, mq, target)) out.add(position(loc, target.getCardId(), "passenger"));
        return out.isEmpty() ? null : out;
    }

    /** All possible results of an Embark / Disembark action over all its targets; null when any is unknown. */
    private static Set<String> allResults(SwccgGame game, Action action) {
        java.util.Collection<PhysicalCard> targets = targets(game, action);
        if (targets == null || targets.isEmpty()) return null;
        Set<String> all = new HashSet<>();
        for (PhysicalCard target : targets) {
            Set<String> r = results(game, action.getActionAttachedToCard(), target);
            if (r == null) return null;
            all.addAll(r);
        }
        return all;
    }

    private FreeMoveRule() {
    }

    /** The phases with free moves: the move phase, and the deploy phase for the capacity-slot move. */
    private static boolean tracked(Phase phase) {
        return phase == Phase.MOVE || phase == Phase.DEPLOY;
    }

    private static State state(SwccgGame game) {
        GameState gs = game.getGameState();
        String key = tracked(gs.getCurrentPhase())
            ? gs.getCurrentPlayerId() + "#" + gs.getPlayersLatestTurnNumber(gs.getCurrentPlayerId()) + "#" + gs.getCurrentPhase() : "";
        State s = STATES.computeIfAbsent(game, g -> new State());
        if (!s.phaseKey.equals(key)) {
            s.phaseKey = key;
            s.visited.clear();
            s.pendingEmbark = null;
            s.pendingTransfer = null;
            s.pendingDock = null;
            s.docked.clear();
        }
        return s;
    }

    /** The weapon / device a transfer moves, or the starship a ship-dock starts from; null for other actions. */
    private static PhysicalCard movedBy(Action action) {
        try {
            if (action instanceof com.gempukku.swccgo.cards.actions.TransferDeviceOrWeaponAction) return (PhysicalCard) TRANSFERRED.get(action);
            if (action instanceof com.gempukku.swccgo.cards.actions.ShipdockingAction) return (PhysicalCard) DOCKING.get(action);
            return null;
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    private static boolean isCraft(SwccgGame game, PhysicalCard card) {
        GameState gs = game.getGameState();
        com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying mq = game.getModifiersQuerying();
        return com.gempukku.swccgo.filters.Filters.starship.accepts(gs, mq, card) || com.gempukku.swccgo.filters.Filters.vehicle.accepts(gs, mq, card);
    }

    /** Whether the card was aboard (pilot, passenger or cargo of) this starship / vehicle earlier in the phase. */
    private static boolean wasAboard(State s, PhysicalCard card, PhysicalCard holder) {
        Set<String> seen = s.visited.getOrDefault(card.getCardId(), Set.of());
        String id = String.valueOf(holder.getCardId());
        return seen.stream().anyMatch(p -> p.split("\\|")[1].equals(id)
            && (p.endsWith("|pilot") || p.endsWith("|passenger") || p.endsWith("|cargo")));
    }

    private static boolean isItem(SwccgGame game, PhysicalCard card) {
        GameState gs = game.getGameState();
        com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying mq = game.getModifiersQuerying();
        return com.gempukku.swccgo.filters.Filters.weapon.accepts(gs, mq, card) || com.gempukku.swccgo.filters.Filters.device.accepts(gs, mq, card);
    }

    /** The cards a weapon / device may be transferred to: Gemp's own check per card at its location, cost ignored. */
    private static List<PhysicalCard> transferTargets(SwccgGame game, PhysicalCard item) {
        List<PhysicalCard> out = new ArrayList<>();
        for (PhysicalCard t : around(game.getModifiersQuerying().getLocationThatCardIsAt(game.getGameState(), item)))
            if (t != item.getAttachedTo() && t != item && item.getBlueprint().getTransferDeviceOrWeaponAction(item.getOwner(), game,
                    item, true, false, com.gempukku.swccgo.filters.Filters.sameCardId(t)) != null)
                out.add(t);
        return out;
    }

    /**
     * A ship-dock's possible partners, from the action's own partner choice (its filter, as Gemp would offer it); when
     * that is unreadable, a superset: the owner's other starships at its location.
     */
    @SuppressWarnings("unchecked")
    private static List<PhysicalCard> dockPartners(SwccgGame game, Action action, PhysicalCard ship) {
        try {
            for (Object effect : (java.util.List<Object>) TARGETING.get(action)) {
                if (!(effect instanceof com.gempukku.swccgo.logic.effects.choose.ChooseCardsOnTableEffect)) continue;
                com.gempukku.swccgo.common.Filterable filters = (com.gempukku.swccgo.common.Filterable) TARGET_FILTERS.get(effect);
                Map<com.gempukku.swccgo.common.InactiveReason, Boolean> overrides =
                    (Map<com.gempukku.swccgo.common.InactiveReason, Boolean>) TARGET_OVERRIDES.get(effect);
                return new ArrayList<>(com.gempukku.swccgo.filters.Filters.filterActive(game, ship, false, overrides, filters));
            }
        } catch (IllegalAccessException | RuntimeException e) {
            // unreadable: fall back to the superset
        }
        return dockPartners(game, ship);
    }

    /** A superset of a starship's ship-dock partners: the owner's other starships at its location. */
    private static List<PhysicalCard> dockPartners(SwccgGame game, PhysicalCard ship) {
        GameState gs = game.getGameState();
        com.gempukku.swccgo.logic.modifiers.querying.ModifiersQuerying mq = game.getModifiersQuerying();
        List<PhysicalCard> out = new ArrayList<>();
        for (PhysicalCard t : around(mq.getLocationThatCardIsAt(gs, ship)))
            if (t != ship && ship.getOwner().equals(t.getOwner()) && com.gempukku.swccgo.filters.Filters.starship.accepts(gs, mq, t))
                out.add(t);
        return out;
    }

    /** Remove the targets {@code back} accepts from a target choice, unless that would empty it. */
    @SuppressWarnings("unchecked")
    private static void filterTargets(AwaitingDecision decision, java.util.function.Predicate<PhysicalCard> back, String kind)
            throws IllegalAccessException {
        List<PhysicalCard> targets = (List<PhysicalCard>) CARDS.get(decision);
        List<PhysicalCard> kept = new ArrayList<>();
        for (PhysicalCard target : targets) {
            if (back.test(target)) REMOVED.computeIfAbsent(kind, k -> new AtomicLong()).incrementAndGet();
            else kept.add(target);
        }
        if (!kept.isEmpty() && kept.size() < targets.size()) {
            targets.retainAll(kept);
            String[] ids = kept.stream().map(c -> String.valueOf(c.getCardId())).toArray(String[]::new);
            ((Map<String, String[]>) PARAMS.get(decision)).put("cardId", ids);
        }
    }

    /** A card's own top-level game-text relocation ("Relocate to your spy", "Relocate to your character", ...). */
    static boolean isRelocation(Action action) {
        if (action == null || !action.getClass().getSimpleName().equals("TopLevelGameTextAction")) return false;
        String text = com.gempukku.swccgo.game.knowledge.NameMarks.strip(action.getText());
        return text != null && text.startsWith("Relocate");
    }

    /** The location card a location deploy plays; null for other actions. */
    static PhysicalCard deployedLocation(Action action) {
        if (action == null || !action.getClass().getSimpleName().equals("PlayLocationAction")
                || !(action instanceof com.gempukku.swccgo.logic.actions.AbstractPlayCardAction)) return null;
        return ((com.gempukku.swccgo.logic.actions.AbstractPlayCardAction) action).getPlayedCard();
    }

    /** The relocation state of the current phase (any phase; reset when the player, turn or phase changes). */
    private static State relocationState(SwccgGame game) {
        GameState gs = game.getGameState();
        String key = gs.getCurrentPlayerId() + "#" + gs.getPlayersLatestTurnNumber(gs.getCurrentPlayerId()) + "#" + gs.getCurrentPhase();
        State s = STATES.computeIfAbsent(game, g -> new State());
        if (!s.relocationKey.equals(key)) {
            s.relocationKey = key;
            s.relocated.clear();
            s.deployTried.clear();
        }
        return s;
    }

    @SuppressWarnings("unchecked")
    private static void restrictRelocations(SwccgGame game, AwaitingDecision decision) throws IllegalAccessException {
        if (!(decision instanceof CardActionSelectionDecision)) return;
        State s = relocationState(game);
        if (s.relocated.isEmpty() && s.deployTried.isEmpty()) return;
        List<Action> actions = (List<Action>) ACTIONS.get(decision);
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            Action action = actions.get(i);
            PhysicalCard card = action.getActionSource();
            PhysicalCard location = deployedLocation(action);
            if (isRelocation(action) && card != null && s.relocated.contains(card.getCardId())) {
                REMOVED.computeIfAbsent("Relocate(game text)", k -> new AtomicLong()).incrementAndGet();
            } else if (location != null && s.deployTried.contains(location.getCardId())) {
                REMOVED.computeIfAbsent("PlayLocationAction(retry)", k -> new AtomicLong()).incrementAndGet();
            } else {
                keep.add(i);
            }
        }
        if (keep.size() < actions.size()) keepOnly(decision, actions, keep);
    }

    private static String role(PhysicalCard card) {
        if (card.isCaptive()) {
            if (card.isImprisoned()) return "imprisoned";
            return card.getEscort() == null ? "unattended" : "escorted";  // escorts hold captives via getCardsEscorting
        }
        if (card.isPilotOf()) return "pilot";
        if (card.isPassengerOf()) return "passenger";
        return "plain";
    }

    private static int location(SwccgGame game, PhysicalCard card) {
        PhysicalCard at = game.getModifiersQuerying().getLocationThatCardIsAt(game.getGameState(), card);
        return at == null ? 0 : at.getCardId();
    }

    private static String position(int location, int holder, String role) {
        return location + "|" + holder + "|" + role;
    }

    private static String current(SwccgGame game, PhysicalCard card) {
        String role = role(card);
        if (role.equals("plain") && card.getAttachedTo() != null && isItem(game, card))
            return position(location(game, card), card.getAttachedTo().getCardId(), "attached");
        if (role.equals("plain") && card.getAttachedTo() != null && isCraft(game, card))
            return position(location(game, card), card.getAttachedTo().getCardId(), "cargo");
        PhysicalCard holder = role.equals("escorted") ? card.getEscort()
            : role.equals("pilot") || role.equals("passenger") ? card.getAttachedTo() : null;
        return position(location(game, card), holder == null ? 0 : holder.getCardId(), role);
    }

    private static void remember(SwccgGame game, State s, PhysicalCard card) {
        s.visited.computeIfAbsent(card.getCardId(), k -> new HashSet<>()).add(current(game, card));
    }

    private static boolean visited(State s, PhysicalCard card, String position) {
        Set<String> seen = s.visited.get(card.getCardId());
        return seen != null && seen.contains(position);
    }

    /** Cards whose position a covered action changes, each with its predicted new position; null if not covered. */
    private static Map<PhysicalCard, String> outcome(SwccgGame game, Action action) {
        PhysicalCard card = action.getActionAttachedToCard();
        if (card == null) return null;
        GameState gs = game.getGameState();
        int loc = location(game, card);
        Map<PhysicalCard, String> result = new HashMap<>();
        switch (action.getClass().getSimpleName()) {
            case "DisembarkAction":
                return null;  // decided from all its possible destinations in restrict (fix C)
            case "MoveBetweenCapacitySlotsAction": {
                String text = com.gempukku.swccgo.game.knowledge.NameMarks.strip(action.getText());
                String role = text != null && text.contains("passenger") ? "passenger" : "pilot";
                result.put(card, position(loc, card.getAttachedTo() == null ? 0 : card.getAttachedTo().getCardId(), role));
                return result;
            }
            case "LeaveFrozenCaptiveUnattendedAction":
                for (PhysicalCard captive : gs.getCaptivesOfEscort(card))
                    if (captive.isFrozen()) result.put(captive, position(loc, 0, "unattended"));
                return result;
            case "DeliverCaptiveToPrisonAction":
                for (PhysicalCard captive : gs.getCaptivesOfEscort(card)) result.put(captive, position(loc, 0, "imprisoned"));
                return result;
            case "TakeUnattendedFrozenCaptiveIntoCustodyAction":
            case "TakeImprisonedCaptiveIntoCustodyAction": {
                boolean imprisoned = action.getClass().getSimpleName().startsWith("TakeImprisoned");
                PhysicalCard at = game.getModifiersQuerying().getLocationThatCardIsAt(gs, card);
                for (PhysicalCard other : around(at)) {
                    if (other.isCaptive() && location(game, other) == loc
                            && (imprisoned ? other.isImprisoned() : other.isFrozen() && other.getEscort() == null && !other.isImprisoned()))
                        result.put(other, position(loc, card.getCardId(), "escorted"));
                }
                return result;
            }
            default:
                return null;
        }
    }

    /** Every card at the location, including cards attached to them (escorted / imprisoned captives). */
    private static List<PhysicalCard> around(PhysicalCard location) {
        List<PhysicalCard> out = new ArrayList<>();
        if (location == null) return out;
        java.util.ArrayDeque<PhysicalCard> todo = new java.util.ArrayDeque<>(location.getCardsAtLocation());
        todo.addAll(location.getCardsAttached());
        Set<Integer> seen = new HashSet<>();
        while (!todo.isEmpty()) {
            PhysicalCard c = todo.poll();
            if (!seen.add(c.getCardId())) continue;
            out.add(c);
            todo.addAll(c.getCardsAttached());
        }
        return out;
    }

    /** Before any player decides: filter the decision in place. */
    @SuppressWarnings("unchecked")
    public static void restrict(SwccgGame game, AwaitingDecision decision) {
        try {
            restrictRelocations(game, decision);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        if (!tracked(game.getGameState().getCurrentPhase())) return;
        State s = state(game);
        try {
            if (decision instanceof CardActionSelectionDecision) {
                List<Action> actions = (List<Action>) ACTIONS.get(decision);
                List<Integer> keep = new ArrayList<>();
                for (int i = 0; i < actions.size(); i++) {
                    Action action = actions.get(i);
                    Map<PhysicalCard, String> out = outcome(game, action);
                    if (action.getActionAttachedToCard() != null && out != null) {
                        remember(game, s, action.getActionAttachedToCard());
                        for (PhysicalCard moved : out.keySet()) remember(game, s, moved);
                    }
                    boolean returns = out != null && !out.isEmpty()
                        && out.entrySet().stream().allMatch(e -> visited(s, e.getKey(), e.getValue()));
                    String kind = action.getClass().getSimpleName();
                    boolean targeted = TARGETED.contains(kind);
                    if (targeted) {
                        PhysicalCard card = action.getActionAttachedToCard();
                        remember(game, s, card);
                        if (kind.equals("TransferBetweenDockedStarshipsAction")) {
                            java.util.Collection<PhysicalCard> ships = targets(game, action);
                            returns = ships != null && !ships.isEmpty() && ships.stream().allMatch(t -> wasAboard(s, card, t));
                        } else {
                            Set<String> all = allResults(game, action);
                            returns = all != null && all.stream().allMatch(position -> visited(s, card, position));
                        }
                    }
                    if (kind.equals("TransferDeviceOrWeaponAction") && movedBy(action) != null) {
                        PhysicalCard item = movedBy(action);
                        remember(game, s, item);
                        int loc = location(game, item);
                        List<PhysicalCard> targets = transferTargets(game, item);
                        returns = !targets.isEmpty()
                            && targets.stream().allMatch(t -> visited(s, item, position(loc, t.getCardId(), "attached")));
                    } else if (kind.equals("ShipdockingAction") && movedBy(action) != null) {
                        PhysicalCard ship = movedBy(action);
                        Set<Integer> done = s.docked.getOrDefault(ship.getCardId(), Set.of());
                        List<PhysicalCard> partners = dockPartners(game, action, ship);
                        returns = !partners.isEmpty() && partners.stream().allMatch(p -> done.contains(p.getCardId()));
                    }
                    if (returns) REMOVED.computeIfAbsent(action.getClass().getSimpleName(), k -> new AtomicLong()).incrementAndGet();
                    else keep.add(i);
                }
                if (keep.size() < actions.size()) keepOnly(decision, actions, keep);  // fix A: may leave Pass only
            } else if (decision instanceof CardsSelectionDecision && s.pendingTransfer != null && decision.getText() != null
                    && decision.getText().startsWith("Choose where to transfer")) {
                PhysicalCard item = s.pendingTransfer;
                s.pendingTransfer = null;
                int loc = location(game, item);
                filterTargets(decision, t -> visited(s, item, position(loc, t.getCardId(), "attached")), "TransferDeviceOrWeaponAction(target)");
            } else if (decision instanceof CardsSelectionDecision && s.pendingDock != null && decision.getText() != null
                    && decision.getText().startsWith("Choose starship to ship-dock")) {
                Set<Integer> done = s.docked.getOrDefault(s.pendingDock.getCardId(), Set.of());
                filterTargets(decision, t -> done.contains(t.getCardId()), "ShipdockingAction(target)");
            } else if (decision instanceof CardsSelectionDecision && s.pendingDisembark != null && decision.getText() != null
                    && (decision.getText().startsWith("Choose where to disembark") || decision.getText().contains("Choose where to transfer"))) {
                PhysicalCard card = s.pendingDisembark;
                boolean docked = s.pendingDockedTransfer;
                s.pendingDisembark = null;
                s.pendingDockedTransfer = false;
                List<PhysicalCard> targets = (List<PhysicalCard>) CARDS.get(decision);
                List<PhysicalCard> kept = new ArrayList<>();
                for (PhysicalCard target : targets) {
                    Set<String> r = results(game, card, target);
                    boolean back = docked ? wasAboard(s, card, target)
                        : r != null && r.stream().allMatch(position -> visited(s, card, position));
                    if (back) REMOVED.computeIfAbsent("DisembarkAction(target)", k -> new AtomicLong()).incrementAndGet();
                    else kept.add(target);
                }
                if (!kept.isEmpty() && kept.size() < targets.size()) {
                    targets.retainAll(kept);
                    String[] ids = kept.stream().map(c -> String.valueOf(c.getCardId())).toArray(String[]::new);
                    ((Map<String, String[]>) PARAMS.get(decision)).put("cardId", ids);
                }
            } else if (decision instanceof CardsSelectionDecision && s.pendingEmbark != null && decision.getText() != null
                    && decision.getText().startsWith("Choose where to embark")) {
                PhysicalCard card = s.pendingEmbark;
                s.pendingEmbark = null;
                Set<String> seen = s.visited.getOrDefault(card.getCardId(), Set.of());
                List<PhysicalCard> targets = (List<PhysicalCard>) CARDS.get(decision);
                List<PhysicalCard> kept = new ArrayList<>();
                for (PhysicalCard target : targets) {
                    boolean wasAboard = seen.stream().anyMatch(p -> p.split("\\|")[1].equals(String.valueOf(target.getCardId()))
                        && (p.endsWith("|pilot") || p.endsWith("|passenger")));
                    if (wasAboard) REMOVED.computeIfAbsent("EmbarkAction(target)", k -> new AtomicLong()).incrementAndGet();
                    else kept.add(target);
                }
                if (!kept.isEmpty() && kept.size() < targets.size()) {
                    targets.retainAll(kept);
                    String[] ids = kept.stream().map(c -> String.valueOf(c.getCardId())).toArray(String[]::new);
                    ((Map<String, String[]>) PARAMS.get(decision)).put("cardId", ids);
                }
            }
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /** After a player decided (before the answer is applied): remember which card is embarking. */
    @SuppressWarnings("unchecked")
    public static void decided(SwccgGame game, AwaitingDecision decision, String answer) {
        if (decision instanceof CardsSelectionDecision && answer != null && decision.getText() != null
                && decision.getText().startsWith("Choose starship to ship-dock") && tracked(game.getGameState().getCurrentPhase())) {
            State s = state(game);
            if (s.pendingDock != null) {
                for (String id : answer.split(",")) {
                    try {
                        PhysicalCard partner = game.getGameState().findCardById(Integer.parseInt(id.trim()));
                        if (partner == null) continue;
                        s.docked.computeIfAbsent(s.pendingDock.getCardId(), k -> new HashSet<>()).add(partner.getCardId());
                        s.docked.computeIfAbsent(partner.getCardId(), k -> new HashSet<>()).add(s.pendingDock.getCardId());
                    } catch (NumberFormatException e) {
                        // not a card id: nothing to record
                    }
                }
                s.pendingDock = null;
            }
            return;
        }
        if (!(decision instanceof CardActionSelectionDecision) || answer == null || answer.isEmpty()) return;
        try {
            List<Action> chosen = (List<Action>) ACTIONS.get(decision);
            int at = Integer.parseInt(answer.trim());
            if (at >= 0 && at < chosen.size() && isRelocation(chosen.get(at)) && chosen.get(at).getActionSource() != null)
                relocationState(game).relocated.add(chosen.get(at).getActionSource().getCardId());
            if (at >= 0 && at < chosen.size() && deployedLocation(chosen.get(at)) != null)
                relocationState(game).deployTried.add(deployedLocation(chosen.get(at)).getCardId());
        } catch (IllegalAccessException | NumberFormatException e) {
            // not an action index (e.g. pass): nothing to record
        }
        if (!tracked(game.getGameState().getCurrentPhase())) return;
        try {
            List<Action> actions = (List<Action>) ACTIONS.get(decision);
            int index = Integer.parseInt(answer.trim());
            if (index >= 0 && index < actions.size()) {
                State s = state(game);
                Action chosen = actions.get(index);
                PhysicalCard moved = chosen.getActionAttachedToCard();
                String kind = chosen.getClass().getSimpleName();
                boolean covered = TARGETED.contains(kind) || outcome(game, chosen) != null;
                if (covered && moved != null) remember(game, s, moved);
                if (kind.equals("DisembarkAction") || kind.equals("TransferBetweenDockedStarshipsAction")) {
                    s.pendingDisembark = moved;
                    s.pendingDockedTransfer = kind.equals("TransferBetweenDockedStarshipsAction");
                }
                if (kind.equals("TransferDeviceOrWeaponAction") && movedBy(chosen) != null) {
                    remember(game, s, movedBy(chosen));
                    s.pendingTransfer = movedBy(chosen);
                }
                if (kind.equals("ShipdockingAction")) s.pendingDock = movedBy(chosen);
            }
            if (index >= 0 && index < actions.size() && actions.get(index).getClass().getSimpleName().equals("EmbarkAction")) {
                State s = state(game);
                PhysicalCard card = actions.get(index).getActionAttachedToCard();
                remember(game, s, card);
                s.pendingEmbark = card;
            }
        } catch (IllegalAccessException | NumberFormatException e) {
            // not an index answer: nothing to remember
        }
    }

    @SuppressWarnings("unchecked")
    private static void keepOnly(AwaitingDecision decision, List<Action> actions, List<Integer> keep) throws IllegalAccessException {
        List<Action> kept = new ArrayList<>();
        for (int i : keep) kept.add(actions.get(i));
        actions.clear();
        actions.addAll(kept);
        Map<String, String[]> params = (Map<String, String[]>) PARAMS.get(decision);
        for (String key : INDEXED) {
            String[] old = params.get(key);
            if (old == null) continue;
            String[] next = new String[keep.size()];
            for (int j = 0; j < keep.size(); j++) next[j] = key.equals("actionId") ? String.valueOf(j) : old[keep.get(j)];
            params.put(key, next);
        }
    }
}
