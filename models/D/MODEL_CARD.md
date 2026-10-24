# D

Distilled reference policy (lab plan critic-noise, stage 0, 2026-10-18).

| | |
|---|---|
| File | `models/D/policy.pt`, 172 MB, slim (weights + embedded tables; optimizer state dropped, RL restarts Adam moments) |
| sha256 | `446be598f3116c827f519a6635487afe39a4f6bfa84e96f87ee761de9b210cd5` |
| Lab source sha256 | `379c9ab560598c587105e0aff56e3345999e402393d831b6d6c97dce6b750e3a` |
| RL games so far (`rl_meta.games`) | 502,872 |
| Table SHA (cards, vocabulary, priors; embedded) | `6df090e02c530d34…` (identical for all four models) |
| Format | V3.1: schema 4, tensor ABI 5, context version 5 |
| Brainiac release | `models/brainiac/releases/D-2026-10-24/` |

**Lineage / strength.** Student of trunk T-x distilled onto the current input view (KL 0.00016, greedy agreement 0.995 on held-out decisions); D vs T-x 0.491 (768 greedy games).

**Use.** The reference opponent of every lab evaluation (`--reference models/D/policy.pt`). cn-ctl, shc-095 and shc-r1 all descend from it.

Win rates are the agent's share of finished games with Wilson 95% intervals, on the Decipher held-out schedules with
the engine of the date given; they are not comparable across engine versions.
