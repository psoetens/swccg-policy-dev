# Context input architecture (V3.1 knowledge model, context version 5) — as of 2026-10-10

What the model's "context" input is, where its rows come from, how they become tokens and embedding-table lookups, and
what the actor may see versus the critic.

- **Plan:** `.claude/plans/context-overhaul.md`; results in `reports/context-overhaul.md`.
- **Token layout, edges, OOV buckets, text priors (shared by all context versions):**
  `reports/knowledge-v3.1-contract.md`, section "Context rows implemented in A/B". That section's key description
  (reflection paths) is context version 4.
- **Why `usage.*` was dropped:** `reports/context-usage-ablation.md`.

## 1. What context is

Cards, piles and knowledge cells say *what is on the table*. Context rows say *what is happening right now*: the action
resolving, the card being played and its targets, what a response window responds to, the effect results pending,
temporary modifiers, battle and drain state. Each row is one token of kind 7 in the same sequence as the card tokens.

## 2. Data flow

```
 ══════════════ BUILD TIME (once, when the vocabulary is made) ═══════════════

  engine enums                      StoredContextV5.keys()
  Action.Type  Zone  Effect.Type    ─────────────────────▶ context-v5-keys.json
  EffectResult.Type  ModifierType                              (795 keys)
  limit-counter fields ...                                         │
                                     + battle.* / drain.* (15) ────┤  shared with v4
                                                                   ▼
                                     build_v5_tables.py        810 keys, sorted
                                                                   │
            "mod.IMMUNE_TO_TITLE" ──readable_text──▶ "mod IMMUNE TO TITLE"
                                                                   │
                                          sentence model (bge-base-en-v1.5)
                                                                   ▼
   runs/knowledge-v3.1/tables-v5/                      768-float text vector
   ┌───────────────────────────────┐                   per key (fp16)
   │ vocab.json                    │◀──────────────────────────────┘
   │   "context": {                │
   │     "<oov:000>" : 0     ┐     │
   │      ...                ├ 512 OOV buckets (no text)
   │     "<oov:511>" : 511   ┘     │
   │     "battle.active" : 512 ┐   │
   │     "mod.IMMUNE_TO_TITLE": 660│
   │     "now.play.HAND"  : 1033 ├ 810 named keys
   │      ...                  ┘   │
   │   "context_version": 5        │
   │ template_priors.npz           │  row id → 768-float text vector
   └───────────────────────────────┘


 ═════════════ RUN TIME (every decision, in the Java engine) ═════════════════

  game state
  ├─ action stack ───────────┐
  ├─ temporary modifiers ────┤   StoredContextV5.append(player, privileged)
  └─ limit counters, card data (critic only)
                             ▼
   one ObservationV2 token per row (kind 7):
   ┌──────────────────────────────────────────────────────────────────────┐
   │ key  "now.play.HAND"   parent → token #8 (the card)   mine = 1         │
   │ a = 0 (stack position)  b = 1 (value)  c = -1  bp = "1_97" (Obi-Wan)  │
   └──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼   PolicyTensorsV4 (tensor builder)
          ┌──────────────────┴────────────────────────┐
          │ 1. contextRows: keep rows of this model's │  (v4 vs v5 matches:
          │    context_version (+ shared battle/drain)│   each model gets its
          │ 2. actor: drop every priv.* row           │   own rows)
          │ 3. key ──vocab.json──▶ row id             │
          │    "now.play.HAND" → 1033                 │
          │    unknown key → hashed to an OOV bucket 0..511
          └──────────────────┬────────────────────────┘
                             ▼
   tok  (int32, ids):     [ kind=7 | bp=<card-table row of 1_97> | zone | owner | slot=1033 | parent=9 | count | flags ]
   tokf (float32, values):[ a=0    | b=1 | c=-1 | d=-1 ]
     tokf columns are shared by all token kinds, hence float; v5 context fills them with whole numbers
     (indices and counts), read by the model as continuous inputs scaled by EfficientConfig.input_scaling:
     "legacy" (default; every actor so far) (index+1)/60 and b/60, "v2" (critic-v2 critics) (index+1)/3 and b/56


 ═══════════════════ MODEL (Python / TorchScript) ═════════════════════════════

   context embedding table  [1322 × d]   (TextPriorEmbedding)
   ┌───────────────────────────────────────────────────────────────┐
   │ rows 0..511    OOV:   learned vector only                      │
   │ rows 512..1321 named: Linear(768→d)(text vector) × scale        │
   │                       + learned per-row residual                │
   └───────────────────────────────────────────────────────────────┘
                 │ lookup by slot = 1033
                 ▼
   token vector =   table[1033]                      "what kind of row"
                  + MLP(scaled a, c, d, count)        where on the stack / which group
                  + Linear(scaled b)                  value (limit count, 0/1)
                  + flag / role embeddings
                  + owner + kind embeddings          mine vs theirs
                  + card_table[bp]                   which card (now.play only)
                 │
                 ▼
   relation message passing:  context token ◀──parent edge──▶ card token #9
                 │             (relation 12: the card's vector learns "I am being
                 │              played", "I am a target", "I am a modifier source")
                 ▼
   transformer over  [question, globals, cards…, context…, knowledge…]
                 │
          ┌──────┴──────┐
          ▼             ▼
       actor          critic   (critic input = the same rows + priv.*)
     (policy)        (value)
```

Reading the diagram:

- **Vocabulary rows are members, not occurrences.** Every enum member gets a table row when the tables are built
  (`mod.*` has 494 because `ModifierType` has 494 members). Per decision only the rows the engine emits become tokens,
  usually a handful; only those rows are looked up and receive gradient. In an 8-game AdvancedAI check, 13 of the 494
  `mod.*` and 70 of the 223 `now.result.*` members occurred. A row that never occurs keeps its text-prior start.
- **Text priors** place related keys near each other from the start ("mod MAY NOT MOVE" near "mod MAY NOT
  PARTICIPATE IN BATTLE").
- **OOV buckets** take any key missing from the vocabulary. With keys generated from closed enums this should not
  happen; the context-overhaul report counts it.
- **Card link, two paths:** the `parent` edge for every linked row, and the blueprint (`bp`) on `now.play` rows only,
  because the card being played often has no token of its own yet.

## 3. Families

| Family | Members | Seen by | Linked card | Fields |
|---|---|---|---|---|
| `now.action.<Action.Type>` | 8 | actor + critic | the action's source card | a = stack position (0 = resolving now) |
| `now.play.<Zone>` | 25 | actor + critic | the card being played | a = stack position; bp = played card (see §4) |
| `now.target` | 1 | actor + critic | an announced target on a visible card | a = stack position, c = target group |
| `now.respond.<Effect.Type>` | 28 | actor + critic | the card the response window is about | a = stack position |
| `now.result.<EffectResult.Type>` | 223 | actor + critic | none | a = stack position, owner = performing player |
| `mod.<ModifierType>` | 494 | actor + critic | the modifier's source card | a = duration container (21 `_untilEndOf*` / `_untilStartOf*` lists, sorted) |
| `battle.*` | 12 | actor + critic | battle location / participants | unchanged from v4 |
| `drain.*` | 3 | actor + critic | drain location | unchanged from v4 |
| `priv.target` | 1 | critic | a target the actor cannot see | b = 1 announced, 0 not yet |
| `priv.limit.<counter>` | 13 | critic | the card the counter belongs to (if any) | b = used count |
| `priv.card.whileInPlay` / `remainder` | 2 | critic | the card holding the stored data | — |

Total 810 named keys + 512 OOV buckets = 1,322 table rows (context version 4: 19,240).

Examples from a real game:

| Row | Situation |
|---|---|
| `now.action.PLAY_CARD` → Obi-Wan Kenobi, a=0 | "Choose characters to 'transport'": Obi-Wan's play is resolving |
| `now.play.HAND` → Obi-Wan Kenobi, bp shown | the same card is played from hand |
| `now.target` → Leia With Blaster Rifle, c=1 | an announced target, group 1 |
| `now.respond.PLAYING_CARD_EFFECT` → Squabbling Delegates | "Playing … — Optional responses" |
| `now.result.FORCE_DRAIN_INITIATED` | "Force drain initiated at … — Optional responses" |
| `mod.IMMUNE_TO_TITLE` → Abyssin Ornament (opponent's), a=3 | lasts until the end of the card play |
| `priv.target` → Tatooine: Obi-Wan's Hut, b=0 | the target is not announced yet; the actor has no such row |
| `priv.limit.turn` → I Will Make It Legal, b=1 | a once-per-turn ability, used once |

## 4. Information boundary (actor)

- A card the actor cannot see is never linked and never shows its blueprint. The row itself may stay ("a card is
  being played from the opponent's hand").
- "Visible" is `ObservationV2`'s rule: a public zone, or the viewer's own card in a zone its owner sees.
- The played card on `now.play` is shown to the actor once the play is announced (targets chosen), when it is visible,
  or when it is the actor's own card (also in a zone the rule does not show, such as the card being played, and with
  its true identity in a hidden pile, such as a location deployed from Reserve Deck).
- Targets: `now.target` only for announced targets on visible cards; everything else is `priv.target`.
- `priv.*` rows are built only for the critic view and are dropped again by the tensor builder for any actor input.
- Check: `ContextV5Check` (`-Dpolicy.checkContextV5=<summary.json>`) swaps an unseen opponent hand card with a Reserve
  card at every decision and requires the actor's v5 rows to stay byte-identical; it also checks that both-version
  output filtered to one version equals that version alone, and that every key is in the vocabulary.

## 5. What is read, and what is not

- Read: the action stack (private fields through reflection: `DefaultActionsEnvironment._actionStack`,
  `ActionStack._actionStack`, `TurnProcedure$PlayOutEffect._effect`, `PlayOutEffectResults._effectResults`), plain
  getters on actions, effects and results, the temporary modifier containers and the limit counters of
  `ModifiersLogic`, and stored card data. No rule evaluation.
- Not read: persistent "while in play" modifiers (`_modifiers`, always-on maps). They follow from the cards in play,
  which the model sees. Any modifier query that needs rule evaluation is excluded by the input-cost rule.

## 6. Context versions side by side

- The vocabulary carries `context_version` (absent = 4; `tables-v5` = 5). `v31_stage.py` and `evaluate_v31.py` read it
  and pass `-Dpolicy.context=4|5|both` to the JVM.
- `both` (a v4 model against a v5 model): the engine emits both versions' rows; `PolicyTensorsV4.contextRows` keeps,
  per model, the rows of its own version and remaps parents.
- Context version 4 (`usage.*`, `states.*`, `rules.*`, `card.*` reflection rows) stays only for playing against the
  finished v4 models.

## 7. Where to change things

| Change | File |
|---|---|
| Families, keys, link rules | `engine/java/com/gempukku/swccgo/game/knowledge/StoredContextV5.java` |
| Version selection per observation | `ObservationV2.java` (`CONTEXT`, `policy.context`) |
| Version filter, actor guard, key → id | `engine/java/com/gempukku/swccgo/bench/PolicyTensorsV4.java` |
| Vocabulary / tables | `scripts/build_v5_tables.py` (keys from `StoredContextV5.main`) |
| Embedding of a context token | `src/swccg_policy/efficient.py` (`encode`), `src/swccg_policy/text_priors.py` |
| Python privileged-row guards | `src/swccg_policy/v4_data.py`, `src/swccg_policy/soft_vocabulary.py` |

A new key needs new tables (a new vocabulary row), so existing v5 checkpoints cannot read it.
