# shc-r1

shc-095 + 25,000 games under the docked-transfer churn fix (FreeMoveRule), lab report shc-r1, 2026-10-24.

| | |
|---|---|
| File | `models/shc-r1/policy.pt`, 345 MB, full (weights + embedded tables + optimizer state) |
| sha256 | `e9e3b2f5c95f2dffc2a51224042617b8f5efbde32066b6ce189ff356e7dd7a48` |
| Lab source sha256 | `260ee721860af68e0aadebaa25716a809b67299d5621ead71c809e9350910eaa` |
| RL games so far (`rl_meta.games`) | 551,906 |
| Table SHA (cards, vocabulary, priors; embedded) | `6df090e02c530d34…` (identical for all four models) |
| Format | V3.1: schema 4, tensor ABI 5, context version 5 |
| Brainiac release | `models/brainiac/releases/shc-r1-2026-10-24/` |

**Lineage / strength.** vs D 0.496 [0.435, 0.557] (256) and 0.502 [0.459, 0.545] (512); AdvancedAI sweep 0.973 (1,110); head to head vs shc-095 0.475 [0.432, 0.518] (512): the plateau holds.

**Use.** The only FULL checkpoint (Adam moments included): resume exactly, or train LR-decayed branches (`scripts/branch_decay.sh`).

Win rates are the agent's share of finished games with Wilson 95% intervals, on the Decipher held-out schedules with
the engine of the date given; they are not comparable across engine versions.
