# Knowledge V3.1 contract — implementation in progress

Reference: `5be2bc2a34ad11e7b0faa62809a2958a27812c3e`. The approved plan and
Amendments 1–5 are in `.claude/plans/knowledge-v3.1-efficient-knowledge.md`.
The previous overflow proposal was rejected; its historical document is preserved
in `reports/data/knowledge-v3.1/pre-amendment-knowledge-v3.1-contract.md`.

**Candidate implementation, not training-ready.** The grid encoder, Python model,
owned tables/checkpoints, SWTRJ003, Java/JNI and ONNX CPU paths are implemented and
have fixture evidence. Gate B and full Gate C/D remain incomplete. Vocabulary corpus
records are explicitly pre-grid and must never be used as final ABI-4 observations.
Schema 3 (V3) and its branches were removed from HEAD by plan remove-old-generations (2026-10-24); its contract is
in git history before that plan's merge.

## Context rows implemented in A/B

The keys below are context version 4. Context version 5 (typed keys, actor/critic split) and a
data-flow overview: `reports/architecture-context.md`.

The textual token remains `[kind, blueprint, zone, owner, key_or_slot, parent,
count, flags, a, b, c, d]`. Context kind is 7. Blueprint defaults to `<pad>`.
Relative owner is 1 for the viewer, 0 otherwise. A textual parent
is a zero-based token index or −1; tensor transport adds one, so zero
means absent. IDs used to find a source are transport lookups, never numeric model
features.

| Context field | Meaning |
|---|---|
| key | Normalized reflection path; each ordered container contributes `.N` |
| a, c, d, count | Up to four list/array indices, in traversal order; absent suffix padded −1 |
| b | Original scalar value; a nonfinite or ±unbounded value is stored as 0 with flag 1 / 2 / 4 (below) |
| parent | Source entity, except alias/reference companion rows described below |
| blueprint | Resolved title reference, when present |

Model handling: the four indices use `(index+1)/60` in a
separate `context_index` projection and are excluded from generic numeric inputs.
Scalar context values use `b/60`. Nothing in
A/B clips values or silently truncates an index tuple. A fifth index level throws
with its path and accumulated tuple.

Flags are independent bits: 1 nonfinite scalar, 2 positive unbounded sentinel,
4 negative unbounded sentinel, 8 alias companion, 16 structural object anchor,
32 canonical title resolution, 64 additional title reference. Scalar sentinels
store b=0. Structural anchors replace meaningless container type names only when
needed to retain alias topology.

A normal context parent derives directed relation 12 from the context row to its
source entity. The context row receives the edge-binding contribution. To retain
both a source-card relation and an alias relation on one stored object, an alias
uses an immediately following context companion (flag 8). Its parent points to
the previously visited object's anchor. It derives relation 14 from the preceding
context row to that target; binding also goes to the preceding row. There are no
separate kind-8 rows for either context relation. Physical kind-8 rows are kept.
The sparse-edge helper has exact topology/binding/padding tests; integration into
the schema-4 model remains pending.

Empty maps, collections, arrays and nulls emit nothing. `.type=` is restricted to
meaningful Action/Effect/Filter/state/evaluator/condition classes, and is removed
without retained descendants, except when A6 omitted default-valued fields. Such
objects retain their type even when all their fields are defaults. Opaque stored objects remain represented. Executable
callbacks, game state, RNG and infrastructure backreferences are never evaluated.
Anonymous/local names become named enclosing class plus enclosing method;
lambdas/synthetic names use `#lambda`. Coverage counts every collapsed name.

Integer map keys resolving to a card/permanent ID become source links and the
literal segment `card`; all other numeric keys remain literal and are counted.
Maps do not consume list-index slots. Debug-only omission and map-key provenance
supported the schema-3 leaf multiset comparison (removed with V3) and is not serialized.

A6 skips zero numeric and false Boolean fields only in the reflected engine-field
walk. It does not skip map/list members, public battle/drain values or sentinels.
The preservation checker restores each omission using the owning class's reflected
field list and verifies that its stable type token remains present. Audit-only
provenance is not part of the observation payload.

A7 removes public `usage.` counters only from the critic, after title normalization.
The public and reflected rows must refer to the same stored LimitCounter object and
match family, action, blueprint, source, used value and flags. An exact-twin check
runs before removal; unmatched rows stay. Context pointers are remapped after
removal. The checker verifies that each removed row still has its `rules.` twin
and restored it for the schema-3 leaf multiset comparison (removed with V3). Actor counters follow the public-counter
rules of `StoredContextV4.appendPublicCounters` and are never deduplicated.

## Title normalization implemented in B (pre-Amendment 2)

The frozen card table supplies blueprint/title pairs. Matching is longest first,
with word boundaries: prefixed `•`/`••`/`<>` titles always match; unprefixed titles
need at least two words or surrounding quotes. Explicit cardHint markup preserves
its stated blueprint. Matches become `<c>` with ordered references. The literal
rules notation `(•)` is a uniqueness symbol and remains text; the audit lists it
separately rather than falsely treating it as a card title.

Question and option references retain kinds 9 and 10, ordinal and option routing.
Flag 1 means canonical title resolution. Ambiguity is resolved only if exactly one
matching blueprint belongs to the supplied permitted set: actor-visible entities
plus the actor's own knowledge. Otherwise the lexicographically smallest blueprint
is canonical. The matcher has no hidden-state access.

A title in a LimitCounters key moves into the blueprint field; only family/action
remain in the key. Other stored strings normalize titles in place, preserving the
first reference on the leaf. Additional references use context companion rows
(flag 64), key `context.titleReference`, b=ordinal, parent=base leaf. These derive
relation 15 from the companion to the base leaf. No title reference is dropped.
The tensor relation convention is relation+1, therefore embeddings have
entries 0 through 16. Canonical resolution uses context flag 32.

## Current append-only vocabulary and cards

Each family q/o/context reserves IDs 0–511 as `<oov:000>` through `<oov:511>`.
Known keys are UTF-16-sorted within each append batch, at minimum frequency one.
Unknown keys use FNV-1a over UTF-8: unsigned initial `0x811c9dc5`, xor each byte,
multiply `16777619` modulo 2^32, then modulo 512. **Buckets collide and are not identities.**
Unknown lookup totals and bounded 100,000-key telemetry remain separate from bucket IDs.

Table format 2 fingerprints the entire payload except `sha`: recursively UTF-16-sorted
object keys, compact JSON separators, ASCII escapes, SHA-256 of UTF-8 bytes. It includes
schema/ABI/normalizer/prior-model metadata, prior-array hashes, append batches and the
parent/ancestor chain. The earlier line-based frozen-vocabulary fingerprint is superseded.
Card-array hashes include dtype, shape and content; the bundle fingerprints both tables.

Known q/o/context rows are offline float16 BAAI/bge-base-en-v1.5 embeddings, projected
by Linear(768→d), scaled and added to seeded identity residuals under Amendment 9 below.
Older checkpoints retain zero-known-residual initialization. Context text replaces `_` and `.` with
spaces. OOV rows are learned-only. No runtime text encoder runs. Fixed 64-row projection
blocks keep old-row GEMM dimensions unchanged on growth. Checkpoints embed their tables;
loading another table requires verified exact prefix content and ancestry. Stage growth
is manual, including model residual expansion; in-place exports and native swaps reject
changed table SHAs.

Initial **Decipher-only** cards are sets <200 from the unchanged local Gemp export,
with its SHA and commit recorded in the table. `tables-decipher` has 2,540 cards plus
four specials. The manual virtual extension adds 1,156 cards from sets 200–226 and 301;
set 227 and Gemp Legacy set 601 are excluded. `225_4` has no implementation in this
engine and was removed upstream. Unknown cards, including virtual cards presented to
the initial table, fail explicitly. The original V3 table remains untouched.

The normalizer version is `knowledge-v3.1-amendment3-v1`. Seed audits reject residual
titles, markers except uniqueness notation, variant suffixes, standalone integers and
anonymous-class numbers in every family. This filters candidates, not observation facts.
Rejected observed keys remain OOV lookups and coverage denominators.

## Grid layout and tensor ABI 4

One kind-2 token per `(blueprint,owner)` binds exactly one `uint8[K,153]` grid row.
Textual slot is zero-based; int32 tensor slot is one-based. Kind-2 zone, parent,
count, flags and all floats are zero after tensor conversion. Kind 4 is absent.
The 153 bytes are 17 quantities × 3 scopes × `(lower,upper,asserted)` in row-major
order. Quantities 0–12 are pile enum indices; TABLE at index 6 is zero. Quantities
13–16 are cross-pile masks 3,5,6,7. Scope order is OUR, PUBLIC, OPP_PRIVATE.
Actor OPP_PRIVATE cells are zero. Bounds remain unclipped uint8; overflow throws.

Each cell is the intersection of the tracker's emitted constraint rows (per pile or cross-pile mask and scope) for
that key, or the
stored completeness/default value when no row exists. Asserted records an emitted
constraint or completeness fact. Defaults preserve implied zero exclusions. A Transit-
bearing group cannot be merged while Transit is nonempty; empty intersections throw.
ORDER and PILE retain their non-grid roles. Hidden-fact actor counterfactual checks pass.

Final actor JSON has 12 elements: existing fields 0–10 plus grid at 11. Training JSON
has 15: actor grid 11, critic tokens 12, knowledge sequence 13, critic grid 14.
Pre-grid schema-4 records and all schema-3 records are rejected by the final parser.

Native arguments are `tok int32[B,T,8]`, `tokf float32[B,T,4]`, `tmask bool[B,T]`,
`opt int32[B,O,6]`, `omask bool[B,O]`, `head int32[B,22]`, `kgrid uint8[B,K,153]`.
Each packed field begins on a 16-byte boundary. All ABI-4 padding is zero; masks mark
valid rows. `forward_packed` additionally takes three int64 sparse-row vectors with
−1 padding: physical/linked-context rows, question rows and option rows. Unlinked
context leaves are not edges. No option/token tail is truncated.

SWTRJ003 file header is magic(8), codec(1), acting table SHA(32). Each game frame uses
big-endian metadata-size uint32, raw-size uint64, stored-size uint64, CRC32 uint32,
then JSON metadata and raw/gzip bytes. Metadata repeats schema 4 and acting SHA.
Step header is big-endian `10 int32, float64, int64`: nt,no,nc,nk,nkc,selected-count,
excluded-count (−1 absent), policy-version,repeats,k,behavior-logp,knowledge-sequence.
Payload is little-endian head, actor tok/tokf/opt/grid, critic tok/tokf/grid, followed
by big-endian selected/excluded int32 arrays. Older magic is rejected. Loader prefix
compatibility requires the actual acting-table bundle, not merely a claimed hash.
Explicit privileged payloads, rules context, private actor scope and private grid cells
are rejected at actor ingestion. Production browser tensor dispatch is still pending.

## Model scales and candidate ablations

All generic counts, option numeric values, grid lower/upper bounds, context scalars,
list indices after +1, location/reference positions, global numeric inputs, reference
counts/overlap and in/out degrees use `/60`. Grid assertion bits, masks and bit flags
remain 0/1. Grid, context-value and context-index projections are separate. IDs and
source indices route lookups; they are not numeric features. Phase uses its existing
categorical embedding. Printed feature normalization remains the unchanged `cards.py`
feature schema. No new count/value clipping is applied.

Amendment 5 adds model-only indicators per cell: `(lower>0, upper==0, lower==upper)`,
multiplied by `asserted`. Concatenate these 153 unit indicators after the existing
153 scaled values; the grid projection is Linear(306→d)→GELU→Linear(d→d). The optional
post-norm fact-value path still consumes the 153 scaled values. The ABI and Java grid
are unchanged. New configurations default `grid_indicators=true`; checkpoint loading
interprets an absent flag as the pre-amendment 153-input projection, without relabeling
or altering the earlier checkpoint.

The scope probe now uses known offline-prior questions 826/827 in a disposable
Decipher-only append bundle. Prior probe IDs 3/6 were OOV rows with zero priors.
Amendment 7 supersedes the earlier diagnostic conclusions and per-scope prototype.
One grid token per blueprint/owner remains the production representation. Gate B
requires 6,000 steps for the small probes, LR 3e-4, batch 40 and 2,000 evaluations per
task on three seeds. Amendment 8 defines the separate padded budget below. Printed comparison
uses all 944 Decipher cards with printed power, split into blueprint-disjoint halves
by SHA256 blueprint order. Ties are excluded; missing-power comparisons use the old
zero fallback only in a separately reported informational sub-case. Padded probes
require a passing small report for the same configuration, task version and tables.

Degrees are computed once, added before the first graph round, and reused by the
optional post-norm degree path. Optional fact-value uses the scaled grid through
Linear+LayerNorm and a learned gain initialized to 0.1. New configurations use
`post_degree=True`, `fact_value=False`, with D1b indicators enabled. The fact-value
switch remains until the gate passes. The two option blocks initialize blueprint
bias heads 0–3 to zero and heads 4–7 to four. Actor/critic depth is independently stored;
critic default remains six. Padded probes preserve real state rows near 400 tokens,
replace prior decision reference rows with the probe decision, and preserve alias-anchor
adjacency while permuting rows and remapping source/edge indices.

`embedding_init_std=0.02` is saved separately in actor and critic configurations.
It initializes only kind, zone, owner, fact_flag, sentinel, relation, graph direction,
dtype and phase embedding tables. Card deltas and text-prior residuals retain their
existing initialization. Checkpoints missing this setting are interpreted as the
historical std 1.0 configuration; saved weights always load unchanged. Explicit
saved settings are preserved. The balanced-init-off ablation uses std 1.0.

## Amendment 8 padded learning curves

The padded gate uses the same five tasks, blueprint splits and configuration as the
passing Amendment 7 small gate. Each seed runs 18,000 updates at LR 3e-4, batch 40;
fixed held-out sets of 2,000 cases per task are evaluated every 2,000 updates, including
the final update. Evaluation has an independent RNG and does not advance the training
stream. Reports include first measured crossing of 99% and whether subsequent measured
checkpoints remain at or above it. Every task on all three seeds must reach 99% at the
final evaluation. No earlier padded diagnostic determines this gate.

The informational control (removed with V3 by plan remove-old-generations) loaded the frozen V3 architecture at
`5be2bc2`, including
its original fact-value path and initialization. Source instances, training streams,
evaluation cases and budgets are paired. Background grid conversion reconstructs
schema-3 rows and checks their decoded grids against the original bytes. The selected
backgrounds each have 11 grids whose unasserted default bounds cannot be recovered
losslessly from those rows and pile summaries. Amendment 8 permits omitting these
rows from the control; their original grids, mismatch coordinates and reasons are
recorded in `data/knowledge-v3.1/amendment8-control-conversion.json` and the curves
report. Consequently this control is explicitly non-equivalent. All retained grids
round-trip exactly. The 64 context leaves in each background convert with explicit
source edges and no context omissions.

There are two eligible real backgrounds from one game: one training state and one
held-out state. They contain 381 schema-4 rows each after removing the old decision;
the converted control backgrounds contain 447 rows. This limited background diversity
and the differing token counts constrain interpretation of the control. The required
outputs are `data/knowledge-v3.1/padded-curves.json` and `padded-curves.png`; resumable
checkpoints and measurement-time source snapshots live under
`runs/knowledge-v3.1/amendment8-curves/`. A final failure stops work for review without
extending the budget or tuning.

## Amendment 9 template identity and training step

Current actor/critic configurations save `template_prior_scale=33.0`,
`template_identity_std=1.0` and `template_identity_seed` (the model seed). Known
q/o/context rows equal the scaled prior projection plus their identity residual.
Each known row uses a CPU generator seeded by the low 63 bits of the first eight
SHA256 bytes (little-endian) of JSON `[model_seed,family,"key",key]`. OOV rows have
no prior and use JSON `[family,"bucket",bucket_index]`, so their initial identities
are stable across model seeds. JSON uses ASCII escaping and compact separators.
No global RNG is advanced while drawing identity rows. Width and std specify the
normal draw; checkpoint weights remain the authority after training.

Append growth generates only new residual rows by the same rule. Existing priors,
residuals, projected embeddings and old-observation outputs remain unchanged.
Checkpoint loading defaults missing scale/std to 1.0/0.0 and restores all saved
weights; this retains the historical zero-known-residual semantics and learned OOV
rows. The template change adds no tensor input, schema or ABI field. Offline priors
and identity residuals are combined into the existing export buffers.

`train_step.py` owns optimizer construction, LR, precision/autocast, loss invocation,
gradient clipping at 1.0 and the optimizer update for BC and all current probe
runners. The step configuration is recorded. Probe runs use Adam at 3e-4, FP32,
zero weight decay and label smoothing 0.05; smoothing is a probe-only confidence
control standing in for PPO entropy/KL, not a BC/RL setting. Existing BC retains its
AdamW groups, bf16 and scheduler through the shared step with smoothing zero.
Probe measurements without this shared step are diagnostics only.

A9 runs fresh small and padded gates with three seeds. The user subsequently accepted
two passing seeds for this stage; all three are still run and retained. All three A9
V3.1 seeds actually met the final threshold. The padded budget and
conversion limitations above still apply. Subsequent Gate E compares paired-schema
AdvancedAI BC on at least 5,000 training games, held-out games and held-out decks.
The reviewer isolated the q-family benefit; the o/context change follows by design
and its real-data benefit remains unverified until measured. The user's later
authorization permits proceeding to the 30-minute RL pilot when the implementation
is ready, then directly to eight-hour training if the pilot works, without another
confirmation. Technical gate failures still stop execution.

## Amendment 9 native ownership and PPO publication

Each ABI-4 native model slot owns its tensor maps. An optional `<model.ts>.bundle`
directory supplies a historical model's maps; the native table fingerprint must
match that bundle. Without a sidecar, the run's current bundle is used. Loading a
historical ABI-4 model with its original tables is supported; mixing observation
schemas within a run is not implemented. Inference errors on ABI 3/4 fail closed.

Actor paths accept exactly the twelve-field public record, rejecting the fifteen-field
privileged record. The browser ABI-4 header is the little-endian SWT4 magic, T/O/K
counts and 22 head integers, followed by int32 tokens, float32 token values, int32
options and uint8 knowledge-grid bytes. Its byte size is `104 + 48*T + 24*O + 153*K`.
Legacy browser formats remain separate.

Native ABI-4 options above 64 use the ordinary eager TorchScript forward method with
the JIT graph optimizer disabled locally. This avoids the measured sparse-tail
near-tie rank change and mixed-method dtype specialization failure. Other native
paths retain their existing graph/sparse execution. Each publication updates cached
TorchScript weights after the first trace; table growth requires a new owned model
and export, not an in-place table swap.

The PPO logical batch retains the original mean losses. Actor and critic split it
independently to 32,768 tokens; each microbatch contributes its sample fraction of
the logical loss. Optimizers step once per logical batch, with actor clipping 0.5
and critic clipping 1.0, preserving the actual existing learner settings. FP32 gradient
comparison uses atol 2e-5 / rtol 2e-4; BF16 uses atol 0.005 / rtol 0.03. No label
smoothing is used in PPO. Two value-warmup updates, two PPO epochs, terminal win/loss
rewards, gamma 1, lambda 0.95 and the existing opponent mixture remain unchanged.

SWTRJ003 files name their acting table identity. The loader accepts the current
bundle and explicitly registered ancestor bundles after validating the full prefix;
it rejects unregistered identities. No automatic table growth runs during training.
Temporary trajectory/checkpoint retention happens only in the stage-owned `/tmp`
workspace. Final models, logs, snapshots and selected trajectory evidence are copied
under `runs/knowledge-v3.1/`; nothing is deleted beneath `runs/`.

Gemp's option-blueprint field also admits the exact non-card markers `inPlay`
(CardActionSelectionDecision board source) and `rules` (ActionSelectionDecision
without an attached card). Both map to existing special card ID 1, preserving the
frozen ABI-3 convention. This exception is limited to option sources; unknown real
card IDs and virtual cards absent from the owned table still raise errors.

Native snapshot retention stores the six most recent atomically published `.ts`
actor exports; it does not duplicate critic/optimizer `.pt` files for opponents.
Resumable learner checkpoints still contain both models, tables, and optimizer state.
The inherited resource limits are <28 GiB GPU, >=4 GiB RAM in sustained measurement,
with stop guards at 2 GiB available RAM and 20 GiB free disk. All measured resource
samples are retained, including startup and shutdown transitions.

## BL-20(a): Force icons on location tokens (plan objective-dojo, 2026-10-05)

- **Field:** a schema-4 location token (kind 1) carries its Force icons in the token float `a` (`tokf[...,0]`).
  It was 0 and unused for entity tokens before.
- **Packing:** `a = printedLight + 16·printedDark + 256·currentLight + 4096·currentDark`, each count in 0..15
  (the encoder throws otherwise). The value is exact in float32.
  - Printed counts come from the card blueprint (`getIconCount`).
  - Current counts come from `ModifiersQuerying.getIconCount`, so modifiers, blown-away rules and similar effects
    apply.
  - Source: `ObservationV2.forceIcons`.
- **ABI:** no tensor shape changes and no new field. Validators never constrained `a` on kind-1 tokens. Schema 2/3
  observations are unchanged (the field is set only for schema 4).
- **Model:** `EfficientConfig.location_icons` (default False, which every older checkpoint loads as).
  - When True, `unpack_location_icons` decodes the four digits and scales them `/60` (the contract's count rule).
  - They feed `Linear(4,d)→GELU→Linear(d,d)` on location tokens. The module is registered after every existing
    parameter, and its last layer starts at zero.
  - `scripts/extend_location_icons.py` turns a checkpoint into its extended twin (AdamW state remapped by name).
  - The twin computes what its parent did (n1: max |Δlogit| 7.6e-6, |Δvalue| 1.8e-6).
- **Not covered:** the production/browser encoder (`engine/production`) does not fill the field, so a browser
  bundle sees zeros there.

## Tensor ABI 5: location state in tokf columns 4–7 (plan objective-dojo Amendment 2, 2026-10-05)

- **Width:** under tensor ABI 5, schema-4 tensors carry `tokf[B,T,8]`. Columns 0–3 are the ABI-4 floats.
- **Location tokens (kind 1)** get columns 4–7: packed power, packed ability, drain us, drain them.
  - The packing is `h(us) + 512·h(them)` with `h(x) = round(2·clamp(x, 0, 255.5))`.
  - "Us" is the deciding player.
  - Power and ability come from gemp `getTotalPowerAtLocation` / `getTotalAbilityAtLocation`. Drain comes from
    `getForceDrainAmount` where that side occupies the location (else 0).
  - Every other token has zeros there.
- **Lazy computation:** `ObservationV2.locationState` is evaluated on first use while the game stands at the decision,
  so ABI-4 consumers never pay for it.
- **JSON records:** a location token may carry the four values as fields 12–15 (16 elements).
- **Where the width comes from:** `bundle.json` `tensor_abi` (4 or 5, from the checkpoint cfg) sets it for Java
  packing.
  - The table and vocabulary formats stay at version 4.
  - PolicyEngine refuses a slot whose module `tensor_abi` differs from its bundle (`NativePolicyV4.tensorAbi`).
  - The JNI library treats ABI 4 and 5 as one family, with the float width taken from the module.
- **SWTRJ003 under ABI 5:** each game's metadata has `float_width: 8`, and the reader takes the width from it.
- **Model:** `EfficientConfig.tensor_abi = 5` and `location_state = True` add `Linear(6,d)→GELU→Linear(d,d)` on
  location tokens (last layer zero, registered last).
  - All other parts read `tokf[..., :4]` only, so ABI-4 checkpoints run unchanged on width-8 batches.
- **n2's start:** `runs/dojo/n2-initial.pt` = n1 + `location_icons` + `location_state` (ABI 5). It computes what n1
  does (max |Δlogit| 2.9e-6 on real ABI-5 data).
