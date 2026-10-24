# shc-095

Consistent-shaping arm (lab plan shaping-consistent, 2026-10-21): reward shaping on the rewards, beta 0.05, lambda 0.95.

| | |
|---|---|
| File | `models/shc-095/policy.pt`, 172 MB, slim (weights + embedded tables; optimizer state dropped, RL restarts Adam moments) |
| sha256 | `0d22449e24e7319314f65e00e8a57b22c2e2c210eec65487f19dab33cc539676` |
| Lab source sha256 | `dfd601c3d93c2aab9f18b94648ec913db5778561ad3241d2faba138197c25d8e` |
| RL games so far (`rl_meta.games`) | 526,905 |
| Table SHA (cards, vocabulary, priors; embedded) | `6df090e02c530d34…` (identical for all four models) |
| Format | V3.1: schema 4, tensor ABI 5, context version 5 |
| Brainiac release | `models/brainiac/releases/shc-095-2026-10-24/` |

**Lineage / strength.** ~24k RL games from D. 0.531 vs cn-ctl; 0.520 / 0.549 vs D (256 / 512 games, rules before 2026-10-24).

**Use.** Parent of shc-r1. A v-cards variant of it is what the lab's Brainiac server ran until 2026-10-24.

Win rates are the agent's share of finished games with Wilson 95% intervals, on the Decipher held-out schedules with
the engine of the date given; they are not comparable across engine versions.
