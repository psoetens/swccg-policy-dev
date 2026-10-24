# Knowledge tracker (design)

The per-player memory of hidden cards that the V3.1 model's knowledge tokens and grid are built from. It came with
the V2 generation (plan `knowledge-state-and-critic`, removed from HEAD with V2 / V3 by plan `remove-old-generations`;
see git history before that commit); this page keeps its design. Code: `engine/java/com/gempukku/swccgo/game/knowledge/`
(`GameKnowledge`, `PileKnowledge`; `KnowledgeGrid` turns it into the schema-4 grid; `ObservationV2` builds the
observation).

## Purpose and boundary

- The actor sees the current board plus everything its player has **legitimately learned or can deduce**: card
  identities, counts, pile membership, searches, verifications, known order. Deterministic bookkeeping does this; the
  network does not have to remember history.
- The training-only critic also sees the actual hidden state and both players' knowledge. The actor never does
  (`priv.*` rows and opponent-private grid cells are dropped for the actor; `InformationBoundaryCheck`,
  `ContextV5Check` and `HandOwnerCheck` test it).
- Explicit knowledge covers factual memory; it does not claim to be a sufficient statistic for every opponent strategy.

## Input-cost rule

Board state means **structural facts, not computed game statistics**: stored fields, pile sizes and ordered lists,
blueprint IDs, instance relationships, event-driven counts, static card features. Not allowed as inputs: modifier
queries for current power / ability / forfeit / deploy cost, aggregate power, simulated battles, re-evaluated legality.
Deductions run when relevant events happen, with bounded work; if one is not cheap, sound uncertainty stays.
(The schema-4 location state, ABI 5, is the one deliberate exception: Gemp's own location totals, computed lazily.)

## Events, not polling

- Knowledge changes only from **recipient-filtered observations** emitted at the event: observe / inspect / verify a
  pile, known and unknown moves, shuffles, known placements, reveals. The tracker never reads hidden `PhysicalCard`
  fields, and permanent card IDs do not let it follow a card through an unobserved shuffle or hidden transfer.
- Hooks sit on the shared operations: the pile-search effects (`ChooseCardsFromPileEffect`,
  `ChooseCardsFromMultiplePilesEffect`, `ChooseCardFromHandOrCardPilesEffect`), look / reveal effects, and the
  `GameState` move / stack / attach / shuffle / recirculate mutators (instrumented by `scripts/build_knowledge_engine.py`).
- Inspected contents are recorded before selectable-card filtering, trivial-decision fast paths and later moves; an
  observation is copied at event time. A path that is not covered must invalidate the affected certainty, never invent
  knowledge.

## What it maintains

- **Lost Piles:** identities recorded as cards enter and leave, so both Lost Piles stay complete exact multisets in
  normal play (not only the visible top).
- **Conservation:** multiplicities with lower / upper bounds per pile and joint constraints across piles (e.g. "one of
  Luke / Han is in Force, the other in Used" after an unknown card was spent). Our known deck minus everything
  accounted for gives exact remainders when only one pile is left. Bounds must be sound: incomplete is fine, false
  certainty is not. For the opponent: observed inventory and unknown counts; training decklists are never used.
- **Order:** known top / bottom segments; a shuffle drops positions within its scope but keeps composition; draws,
  activation, spending and recirculation transform the segments. Inspection gives composition, not physical order,
  unless the effect grants order.
- **Verification and disclosure:** a failed search shows the pile to each verifier, and we record what the opponent
  saw of our pile as disclosed knowledge, separate from their private deductions (which the actor never gets).
- **Persistence:** compact count maps and unresolved order segments, copied with engine snapshots (`GameKnowledge.copy`);
  a `shadow` copy can run the legacy rule side by side (plan critic-noise stage 0).

## Scopes in the observation

Each fact carries a scope: OUR VIEW, PUBLICLY DISCLOSED, OPPONENT PRIVATE (critic only). In schema 4 all facts of one
(blueprint, owner) become one grid row: 17 quantities (13 piles + 4 cross-pile masks) × 3 scopes × (lower, upper,
asserted); see `reports/knowledge-v3.1-contract.md`.
