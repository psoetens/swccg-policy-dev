# cn-ctl

Control arm trained from D (lab plan critic-noise, stage 2, 2026-10-19).

| | |
|---|---|
| File | `models/cn-ctl/policy.pt`, 172 MB, slim (weights + embedded tables; optimizer state dropped, RL restarts Adam moments) |
| sha256 | `ad069848038fe9923556871520b4df3d1375395edbde3767ee1e3daa08882eba` |
| Lab source sha256 | `5858d219e770ff837803b073dd1b3c73b881735d87123224ed87ebeb5a85c84d` |
| RL games so far (`rl_meta.games`) | 526,921 |
| Table SHA (cards, vocabulary, priors; embedded) | `6df090e02c530d34…` (identical for all four models) |
| Format | V3.1: schema 4, tensor ABI 5, context version 5 |
| Brainiac release | `models/brainiac/releases/cn-ctl-2026-10-21/` |

**Lineage / strength.** ~24k RL games from D with the standard recipe (pool v3b, 480-minute stages).

**Use.** Baseline for A/B arms: shc-095 scored 0.531 against it.

Win rates are the agent's share of finished games with Wilson 95% intervals, on the Decipher held-out schedules with
the engine of the date given; they are not comparable across engine versions.
