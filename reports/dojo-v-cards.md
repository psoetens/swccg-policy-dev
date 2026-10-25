# Dojo rewards for the V-card decks, and the first V-card training run

Plan: `.claude/plans/dojo-v-cards.md` (approved 2026-10-25, Amendments 1–4). Branch `dojo-v-cards` (worktree
`~/src/swccg-policy-dojov`). Data: `reports/data/dojo-v-cards/`. Hardware: AMD 9800X3D + RTX 5090.

## Result in one line

40 new Dojo items cover every objective, Epic Event and Jedi Test of the 927 V-card decks. The first V-card run
(shc095-vcards → vct-decay, ~19,200 + 2,000 games, 4 h + 26 min) **beats its starting checkpoint on the 184 held-out V
decks: 56.8% (95% CI 54.8–58.8%) over 2,400 paired games**. Against AdvancedAI both score 99.25%, the ceiling.

## §1 end-goal

> 1. `scripts/dojo_coverage.py` on `data/schedules/v-cards/{gather-ds,gather-ls,heldout}.tsv` … reports **0 missing**
>    items; on the existing Decipher pools it is unchanged (v3b: 0 missing).

Met: 0 missing for gather-ds (739 decks), gather-ls (738), heldout (184), the training pool
`rl_pool_vcards_gathering.txt` (743), and unchanged on rl_pool_decipher_v3b (753), eval_decipher_heldout (29),
schedule-v3b (666).

> 2. `efficient_java_check.sh DojoRegistryCheck` passes …

Met: "dojo items: 91", exit 0 (after every table change).

> 3. **Games** … every new item with ≥ 4 decks gets ≥ 128 games on its decks: necessity (0 violations, or each one
>    explained as an unmodelled path and fixed in the table); no free rewards; per-item rates reported.

Met, with two table fixes and one decision:
- 7,168 games (shc095-vcards vs AdvancedAI and vs itself, greedy, V title table), 128 per item (64 for the two items
  with 2–3 decks). Readout `item-readout.md`.
- Necessity: 652 violations at first → 0 after two fixes in the table (flags only, payouts unchanged, Amendment 2):
  - Endor Operations 8_167 (existing item): Establish Secret Base (V) 207_25 deploys on the Bunker when you control
    it, so the "control 2/3 Endor sites" milestones are not necessary in the 51 V decks that play it (643);
  - Restore Freedom To The Galaxy 208_17: the destiny alone can exceed 5 without a snub fighter (9).
- Free rewards: They Have No Idea We're Coming 209_29 flipped before own turn 3 in 90.6% of games vs AdvancedAI
  (Stardust controls the Data Vault from the start; a first-turn deployment adds a second Scarif location).
  **Peter: success 0** (Amendment 2). No passive item pays.

> 4. Existing Dojo items unchanged: … byte-identical before / after.

Met: 32/32 deterministic games (D vs AdvancedAI, 1 thread, seed 919001), base 923066e vs the branch: winner and every
per-game Dojo record identical (`regression.txt`).

> End-goal addition (Amendment 1): the run finishes (or is stopped at 8 h) with 0 crashes, a `final.pt`, and the
> held-out readout above; strength claims only with intervals, and "not shown" if inside them.

Met: 0 crashes and no restarts; `runs/dojo/vct/final.pt` and `runs/dojo/vct-decay/final.pt`; readout below.

## Training (Amendments 1, 3, 4)

shc recipe (λ 0.95, damage shaping 0.05, critic warm-up 8), the V title table, Dojo on, 12 threads, 743 gathering
decks as learner and opponent pool. The budget was cut to 4 h (Amendment 3); the trunk ran slower than the shc runs
(77–89 vs ~114 games/min: about a third larger observations, the learner is the bottleneck), so its 230-min cap
came at **19,191 games**, before the planned cosine decay. Amendment 4 (Peter, option 2): a separate 2,000-game
decayed branch from the trunk's final (learning rate 5e-5 → 2.5e-6, 26 min) is the evaluated model.

| Trunk quarter | Games | vs AdvancedAI | vs Rando | vs snapshot | Explained variance | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,972 | 99.7% (312) | 99.5% (183) | 50.5% (771) | 0.862 | 0.489 |
| 2 | 4,958 | 99.7% (291) | 98.6% (141) | 47.1% (766) | 0.885 | 0.527 |
| 3 | 4,655 | 99.6% (280) | 99.4% (155) | 48.0% (659) | 0.898 | 0.525 |
| 4 | 4,606 | 98.9% (270) | 100% (122) | 50.5% (711) | 0.900 | 0.527 |

In-run win rates sit at the ceiling (AdvancedAI, Rando) or near 50% by construction (snapshot), so they show no
progress; the critic's fit rose. Collapse check: "warn" only (raw max logit above 50, capped at 30 in the policy).

## Held-out V decks (184 decks never trained on, 2,400 paired rows, greedy)

| Model | Opponent | Win rate | 95% CI |
|---|---|---:|---:|
| vct-decay (final) | **start (shc095-vcards)** | **56.8%** (1,363/2,400) | 54.8–58.8% |
| vct-decay (final) | AdvancedAI | 99.25% | 98.8–99.5% |
| start | AdvancedAI | 99.25% | 98.8–99.5% |
| vct-decay (final) | Rando Cal | 98.5% | 97.9–98.9% |

The gain over the start is outside the interval. Against AdvancedAI: **not shown** (identical, at the ceiling). Rando
Cal for the start was not run, so no before/after there.

### Dojo items before / after (held-out, vs AdvancedAI, the agent's side)

`heldout-items-compare.json` (Wilson 95% intervals per item; "up/down" only where they do not overlap).

| Item | Games | Success start → final | |
|---|---:|---|---|
| The First Order Reigns 225_32 | 103 | 45% → 76% | **up** |
| Restore Freedom To The Galaxy 208_17 | 16 | 44% → 81% | |
| Old Allies 204_32 | 23 | 48% → 74% | |
| Rescue The Princess (V) 215_17 | 66 | 35% → 53% | |
| A Great Tactician Creates Plans 219_1 | 21 | 38% → 52% | |
| They Have No Idea We're Coming 209_29 (pays 0) | 78 | 100% → 88% | **down** |
| This Deal Is Getting Worse All The Time (V) 226_12 | 63 | 98% → 81% | **down** |
| Imperial Entanglements 201_39 | 64 | 91% → 72% | |
| Shadow Collective 213_32 | 87 | 92% → 78% | |
| Hunt For The Droid General 221_67 (pays 0, after-flip milestones) | 15 | 100% → 67% | |

Never reached before or after: The Shield Will Be Down In Moments (222_14, 222_30), Diplomatic Mission To Alderaan
(tapes never delivered), On The Verge Of Greatness (Death Star never orbits Scarif), The Hidden Path, A Stunning Move,
It Is The Future You See (V). Mixed per-item moves while the head-to-head improved: the Dojo flips are not what the
gain comes from, as far as these sample sizes show.

## Skipped / unverified / observations

- Skipped vs the plan: training was 4 h + a 26-min decay branch, not 8 h (Amendments 3–4). The start was not played
  against Rando Cal.
- Items with < 4 decks got 64 games each (221_67, 210_25); O11, E14, E15 are thin in the held-out set (≤ 17 games).
- Never active in any deck (excluded by their objective everywhere): 225_34 Tracked Fleet, 222_13 Target The Main
  Generator (V), 215_2 A Power Loss — their predicates are registry-checked, not game-checked.
- Endor Operations in V decks still pays control 2/3 Endor sites, which Establish Secret Base (V) does not need
  (follow-up option: milestones that depend on the deck's ESB).
- Shadow Collective 213_32 succeeds in ~90% of games but pays ~0.12 per game: the flip seems to be paid and taken
  back (the objective flipping back). Not investigated.
- 209_29 at 0 reward: its flip rate fell from 100% to 88% — the policy no longer goes out of its way for it.

## What is where

- Code/data on the branch (not committed yet): `data/dojo/rewards.json`, `ObjectiveDojo.java`, `tests/test_dojo.py`,
  `scripts/{dojo_item_readout.py,dojo_v_item_games.sh,make_item_schedules.py,vct_chain.sh,vct_decay.sh}`,
  `scripts/v31_stage.py` (`--title-table`), the item schedules and the gathering pool under `data/schedules/v-cards/`.
- Checkpoints: `runs/dojo/vct/final.pt` (trunk), `runs/dojo/vct-decay/final.pt` (the evaluated model).
- Next (Peter's word each): merge `dojo-v-cards` into lab main; export to the dev repo (`export_from_lab.py --ref`),
  build the Brainiac release there; push; redeploy the beta with it.
