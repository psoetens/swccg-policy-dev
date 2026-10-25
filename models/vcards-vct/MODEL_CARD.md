# vcards-vct — after the first V-card training run

`models/shc095-vcards` trained on the 743 V-card gathering decks (lab plan dojo-v-cards, 2026-10-25; lab report
`reports/dojo-v-cards.md`, shipped here). The lab's `runs/dojo/vct-decay/final.pt`.

| | |
|---|---|
| File | `models/vcards-vct/policy.pt`, 218 MB, slim (no optimizer state) |
| sha256 | `09d069df7c401b5a3957c5116d40a29f2fea2391d5dd570405dcc30ee97bca3f` |
| Lab source sha256 | `0e4b8aab6300f91c61bf644608fafe7680305ff60be56645b3ead8280e9870b2` (`runs/dojo/vct-decay/final.pt`) |
| RL games (`rl_meta.games`) | 548,138 = shc095-vcards' 526,905 + 21,233 on V-card decks |
| Table SHA | `30d81bd4…` (the V-grown tables of shc095-vcards) |
| Title table | `models/vcards-vct/titles.tsv` (V-card display titles); pass it as `--title-table` |
| Brainiac release | `models/brainiac/releases/v-cards-vct-2026-10-25/` (also on the Brainiac beta server) |

**Training:** a 4-hour trunk (19,191 games, learning-rate warm-up then plateau; the time cap came before its decay)
and a 2,000-game decayed branch from its final (learning rate 5e-5 → 2.5e-6). shc recipe (λ 0.95, damage shaping
0.05, critic warm-up 8), V title table, Dojo rewards on (every V objective, Epic Event and Jedi Test has one), pool
`data/schedules/v-cards/rl_pool_vcards_gathering.txt` as learner and opponent pool, 12 threads, one GPU
(AMD 9800X3D + RTX 5090), 0 crashes.

**Held-out V decks** (`data/schedules/v-cards/heldout.tsv`, 184 decks never trained on, 2,400 paired rows, greedy):

| Opponent | Win rate | 95% CI |
|---|---:|---:|
| shc095-vcards (its start) | **56.8%** | 54.8–58.8% |
| AdvancedAI | 99.25% | 98.8–99.5% (shc095-vcards: the same) |
| Rando Cal | 98.5% | 97.9–98.9% |

The gain over the start is outside the interval; against AdvancedAI nothing is shown (both at the ceiling). Per-item
Dojo success moved both ways (The First Order Reigns 45% → 76%; This Deal Is Getting Worse All The Time 98% → 81%);
see the report.

**Evaluate:**

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_v31.py --checkpoint models/vcards-vct/policy.pt \
    --reference models/shc095-vcards/policy.pt --tag vct-vs-start --opponents initial --games 800 --threads 16 \
    --schedule data/schedules/v-cards/heldout-800.tsv --title-table models/vcards-vct/titles.tsv
```

Train further: `QUICKSTART.md` §5.
