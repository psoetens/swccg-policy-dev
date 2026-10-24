# shc095-vcards — NOT TRAINED ON V-CARD DECKS

shc-095 (see `models/shc-095/MODEL_CARD.md`) with its tables grown for Virtual cards (lab plan v-cards, 2026-10-23).
**It has never trained on a deck with V cards**: the 1,189 V-card rows and the 548 new question / option keys start
from text priors with their learned parts zeroed, and the weights are shc-095's. Use it to play or evaluate the V-card
decks (`data/schedules/v-cards/`), or as the start of the first V-card training run.

| | |
|---|---|
| File | `models/shc095-vcards/policy.pt`, 218 MB, slim (no optimizer state) |
| sha256 | `6e1c1c0a2ccc0c9ef326f777d08e78643b3d771ee42f9143c1219ca334cdcb02` |
| Lab source sha256 | `02abebb4e97364ac31d7f1da4203b2bd5fc1483349de0b55841d3fe5d7c46666` (`runs/v-cards/shc095-vcards.pt`) |
| RL games (`rl_meta.games`) | 526,905 — all on Decipher-era decks |
| Table SHA | `30d81bd4…` (3,733 card rows: Decipher 2,544 + 1,189 V cards of sets 200–227 and 301; **differs from the other models**) |
| Title table | `models/shc095-vcards/titles.tsv` (V-card display titles); pass it as `--title-table` |
| Brainiac release | `models/brainiac/releases/v-cards-shc095-2026-10-23/` |

**What is known (lab smoke test, 2,400 held-out V games, greedy, no training):** it plays V decks legally and to the
end; vs AdvancedAI 0.983, vs Rando Cal 0.980, vs itself 0.487 (2.9% unfinished). These opponents are weak: this says
nothing about how *well* it plays V cards. A Falcon-and-Han search question (Passage To The Alderaan System 223_42,
Beldon's Eye 211_31) repeats in some games; not traced.

**Evaluate** (Dojo coverage off: most V objectives have no Dojo reward yet):

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_v31.py --checkpoint models/shc095-vcards/policy.pt \
    --reference models/shc095-vcards/policy.pt --tag vcards-rando --opponents rando --games 800 --threads 16 \
    --schedule data/schedules/v-cards/heldout-800.tsv --title-table models/shc095-vcards/titles.tsv --no-dojo-coverage
```

**Training on V decks is not set up.** `v31_stage.py` refuses a pool whose objectives have no Dojo reward
(`scripts/dojo_coverage.py`; 40 of the 52 V objectives lack one), and the training rollouts use the Decipher title
table. Both are the open work item (lab BL-58: Dojo rewards for V objectives and a V training pool).
