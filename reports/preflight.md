# Pre-flight fixes before the weekend run (Addendum 4)

**Plan:** `.claude/plans/weekend-modern-retrain.md`, Addendum 4 (approved 2026-09-25) · **Date:** 2026-09-25
**Status:**
- loop guard **done and meets its targets**;
- `publish()` **done**;
- CUDA graphs **done after Peter's option (b)**: bit-exact vs eager on the same shapes, 2.7×/1.7×/1.3× per forward. The end-to-end
  rollout rate is unchanged, because the Java side now saturates the CPU (§3b).

Nothing is committed.

## 1. Loop guard (`PolicyAi`, fork harness) — done

**Change:**
- **Fingerprint:** a situation = decision type + 64-bit FNV-1a hash of the exact encoded model input (`StateEncoder.Encoded.prefix`:
  state, prompt, options). Kept per game; only hashes are stored.
- **Threshold:** up to K = 10 occurrences (`-Dpolicy.loopGuardThreshold`) the policy decides normally.
- **Escape:** from occurrence 11, the policy's best-ranked answer not yet tried in that situation (remembered for the rest of the
  game). BeginnerAi only when every answer has been tried.
- **Rejected answers** (the same decision object re-sent): counted as tried, and the next untried answer is submitted.
- **Multi-card answers** are compared as sets: the same cards in another order are the same answer.
- **Server:** `rl.py server --guard-threshold` (default 10; steps with r > K are not recorded). This replaces the hard-coded `r < 3`.
- **Counters:** TrainGen's summary line adds `real-loops`, `rejected` and `decisions` (appended, so existing parsers still match).
  `-Dpolicy.loopLog=FILE` logs every override.
- **Files:** `PolicyAi.java` and `TrainGen.java` in the fork `local-dev` working tree and `build/gemp` (compiled);
  `src/swccg_policy/rl.py`.

**Validation** (`runs/guard-test/`, pilot checkpoint, v3 pool; old-guard figures from the thread A/B and the pilot evaluation):

| | old guard | new guard | target |
|---|---|---|---|
| greedy (pilot vs `rl-final`, 200 held-out games): overrides | ≈ 4.5 % | **0.28 %** | ≤ 1 % |
| greedy: BeginnerAi fallbacks | 0.1 % | **0.07 %** | ≤ 0.5 % |
| sampled (league, 200 games): overrides | ≈ 20 % | **0.02 %** | ≤ 1 % |
| sampled: BeginnerAi fallbacks | 9–18 % | **0.00 %** | ≤ 0.5 % |
| completion (greedy / sampled) | 99.8 % / ≈ 98 % | **100 % / 100 %** | ≥ 99 % |
| exceptions / rejected answers | — | 0 / 0 | 0 |

**Real loops:** with greedy play there are 143 situations repeated more than 10× in 200 games (about 0.7 per game), mostly on action
menus: "Choose Deploy action or Pass" 73, Activate 24, Control 20, weapons segment 10, Draw 6. Examples from the override log:
1. **"Choose Activate action or Pass"** (game gg2-170): many distinct situations, each repeated exactly 10×. The policy re-selects an
   action that gets aborted; the first escape (another action or pass) resolves each one.
2. **"Choose where to deploy Dark Jedi Lightsaber (V) as attached"** (game gg2-164, 5 targets): the same target 10×, then all 4
   others; none works, so from occurrence 15 BeginnerAi decides, also without success. It resolved only when the deploy menu one
   level up escaped its own loop, after 517 s of game time. This is probably a card/engine issue (targets offered that can't be
   attached to).
3. **The same card, game gg2-169** (2 targets): the same pattern.

**Found along the way (card/engine bugs, not fixed, outside Addendum 4):**
- **Any Methods Necessary** (`109_7`, DS starting interrupt: "Take into hand from Reserve Deck one prison and one bounty hunter (may
  also take a matching weapon and/or starship)"). Its step-by-step selection ("Choose a prison and a bounty hunter …") comes back
  unchanged forever, whatever the answer (policy or BeginnerAi): **8 of 200 games (4 %)** with the 75 decks containing the card hang
  until the 6,000-decision cap. **This produced most of the huge fallback counts in sampled rollouts** (≈ 6,000 fallbacks per stuck
  game). Suspected: the selection asks for exactly "previous selection + 1" cards, and an answer that drops earlier cards can never
  become valid. Needs an investigation like the NPEs.
- **Dark Jedi Lightsaber (V):** attach targets offered that don't work (loop example 2).

## 2. `publish()` (`tools/decks/screen.py`) — done

- **Change:** everything a publish replaces (`corpus/`, `rejected/`, `manifest.csv`, `unplayable_cards.csv`, …) is kept as
  `<name>.v<N-1>`, numbered further if that name is taken. `CORPUS_READY` lists every era (it skipped `retro`). `SCREEN_LIVE`
  allows testing on a copy.
- **Test** (scratch copy of `data/decks`, fake v4 staging): the old `rejected/` was kept (14 files); `manifest.v3.csv` equals the
  previous manifest; an existing `corpus.v3` wasn't touched (the old corpus went to `corpus.v3-2`); the live data was unchanged.
- **Earlier today (no code):** the 14 v2 deck files deleted by the v3 publish were restored from a deterministic rebuild (380/380
  byte-identical to `corpus.v2`). Corpus v2's manifest was also overwritten at that publish; only the rebuild's pre-screen manifest
  (`data/decks/v2rebuild/manifest.csv`) remains.

## 3. Fixed forward cost — stopped at a stop condition

**Findings:**
- **Cached card table** (`Policy.cache_card_table()`, used by every server model slot). The card MLP over about 3,700 cards ran on
  every forward. Caching saves only about 0.06 ms: the forward really is launch-bound (≈ 1.25 ms, flat from batch 1 to 6 on an idle
  GPU). Kept, since it's free.
- **CUDA graphs** (`GraphForward`, server flag `--graphs`, default off). One graph per padded bucket (batch 1–64 × tokens in steps of
  32 × options in steps of 8); larger shapes run eagerly. The server now launches all slots' forwards in a batch window before one
  synchronization (consolidation option a). This also applies without `--graphs`.

| batch | eager | graph (pad + copy + replay) | speed-up | argmax agreement | max rel. diff |
|---|---|---|---|---|---|
| 1 | 1.23 ms | 0.55 ms | 2.3× | 200/200 | 1.7e-3 |
| 6 | 1.30 ms | 0.89 ms | 1.5× | **1199/1200** | 4.5e-3 |
| 16 | 1.58 ms | 1.39 ms | 1.1× | **3198/3200** | 4.5e-3 |

Targets: ≥ 3× at batch 6; 100 % agreement; ≤ 1e-3.

**Root cause of the disagreement:** **TF32** (`allow_tf32 = True` in `rl.py`), not the graphs:

| eager, padded vs unpadded | argmax agreement | max rel. diff |
|---|---|---|
| TF32 on | 1798/1800 | 4.5e-3 |
| TF32 off | **1800/1800** | **2.9e-6** |

The current (eager) server is therefore already slightly batch-dependent: a request's padding depends on its batch neighbours, and
under TF32 near-ties can flip. The speed-up also falls short of 3×. The remaining cost at batch 6 is padding plus about 10 separate
host-to-device copies plus the larger padded compute.

**Options (not done; Peter's call):**
1. **TF32 off for server inference.** Deterministic regardless of batching, and the graphs then match eager. It costs little at these
   batch sizes (launch-bound); to be measured.
2. **More speed:**
   - pack all inputs into one pinned buffer (one copy);
   - finer token buckets (e.g. steps of 16);
   - consolidation option (b): one forward over stacked snapshot weights.
3. **Or accept today's gain** (2.3× at batch 1, 1.5× at batch 6) plus threads 10 → 20 (+28 %), and measure end to end.

## 3b. Option (b), Peter 2026-09-25: graphs v2 and end to end

- **Graphs v2:**
  - per bucket one pinned host buffer and one device buffer: the batch is written straight into pinned memory, then one copy up,
    the replay, and one copy of the logits back, all asynchronous, followed by one synchronization per batch window;
  - token buckets in steps of 16.
- **Equality:** graph vs eager on the same padded shapes: **argmax 100 %, max rel. difference 0.0 (bit-exact)** at batch 1/6/16.
  Against unpadded eager, only TF32's shape effects remain (≤ 4.5e-3, 1 argmax in 3,200), which the eager server has too.
- **Per forward** (idle GPU):

  | batch | eager | graph | speed-up |
  |---|---|---|---|
  | 1 | 1.28 ms | 0.47 ms | 2.7× |
  | 6 | 1.33 ms | 0.78 ms | 1.7× |
  | 16 | 1.67 ms | 1.29 ms | 1.3× |

  Target at batch 6 was 3×: **not met**. The parts at batch 6: filling 0.015 ms, copy up 0.009 ms, copy back 0.006 ms, **replay
  0.73 ms** (the GPU running the ≈ 150 small kernels one after another). Request preparation in Python (`parse_record` + `compact`)
  takes 0.28 ms per batch of 6. More would need kernel fusion (`torch.compile`) or bf16.
- **End to end** (1,500 games, 20 threads, v3 pool, league, no learner running):

  | | games/s | median wall time per game | Java CPU | Python server CPU | GPU |
  |---|---|---|---|---|---|
  | 10 threads, eager | 2.64 | 2.91 s | 4.2 cores | 71 % | 29 % |
  | 20 threads, eager | 3.39 | 5.24 s | 4.7 cores | 65 % | 25 % |
  | 20 threads, graphs | 3.35–3.40 | 4.28 s | **6.0 cores** | 52 % | 24 % |

  The server got faster and lighter, but **the Java side now saturates the CPU** (≈ 6 of 8 cores plus Python and the rest of the
  machine). It spends ≈ 1.1 ms of CPU per decision on the Gemp engine and `StateEncoder`. The next lever is on the Java side:
  profile `StateEncoder` and the engine per decision. Faster GPU inference alone no longer raises throughput on this machine.
- **Side note:** the 20-thread graphs run again showed ≈ 66,000 fallbacks: Any Methods Necessary games stuck until the cap (§5).

## 5. Any Methods Necessary — analysis (Peter asked for the deck and the cause)

- **Not deck-specific:** 11 stuck games involved 11 different decks, each stuck once in 2–6 games. The decks do contain a prison and
  bounty hunters.
- **Example:** game `amnx-7`, DS **26Gempc_Lingrell_DS_Eops** (learner). The prison is Cloud City: Security Tower; the bounty hunters
  are Boba Fett and 2× 4-LOM With Concussion Rifle; also Slave I, Symbol Of Fear.

  | step | required | chosen | state |
  |---|---|---|---|
  | #5 | 1 card | {Security Tower} | valid so far |
  | #6 | 2 cards | {Boba Fett, 4-LOM} | the Tower was dropped: two bounty hunters |
  | #7 | 3 cards | {Boba, 4-LOM, Tower} | invalid |
  | #8 onwards | 4 cards | {Boba, Tower, 4-LOM, Slave I} | repeated until the cap |
- **Cause (Gemp):**
  - while invalid, the selection requires exactly "previous + 1" cards, but it allows replacing earlier picks;
  - the allowed-card filter is computed from the *previous* selection only, so two bounty hunters were accepted in one step;
  - once invalid with nothing addable, min = max = the current selection, and cards can't be removed: an endless re-ask;
  - human-reachable (deselect + select two), policy-reachable easily (the harness answers top-k instead of "keep + add one").
- **Gemp fix: done** (Peter's choice; NPE-fixes A1.9): each answer is validated as a partial combination, and cards can be removed
  when stuck. 200 sampled games with these decks: **0 stuck** (before 8), 0 fallbacks (before 47,838). The harness fix ("keep + add
  one") was not done (not chosen).

## 6. Dark Jedi Lightsaber (V) — no implementation bug (Peter asked to fix it)

- **The loop:** Blaster Rack "Deploy stacked weapon" onto Vader, Emperor's Enforcer → Blaster Rack "Stack character weapon" (allowed
  "at any time") → repeat.
- **Why it's free:** character weapons state their deploy cost in the game text (Peter). The (V) version states none ("Deploy on
  Aurra Sing, Grievous, or your warrior of ability > 4"), so the rack transfer ("Force equal to the weapon's deploy cost") is free,
  and the state repeats exactly: a legal no-op cycle that the new guard breaks.
- **Card tests** (`Card_211_025_Tests`, 3 passing):
  - deploy from hand;
  - rack transfer with Force unchanged;
  - the Premiere version via the rack, charged exactly 7 − ability, so the rack path does charge textual costs.
- No code was changed. This corrects my earlier "targets that don't work" claim.

## 7. Not done / unverified

- **The end-to-end rollout A/B with graphs:** not run (stopped before it).
- **Consolidation options (b)/(c):** not implemented.
- **The new guard in a real training run:** not yet exercised; the server default K = 10 must match the harness default.
- **The two card/engine bugs above:** listed only.
