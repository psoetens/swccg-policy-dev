# V-card decks — prepared, NOT TRAINED ON YET

927 tournament decks with Virtual cards (lab plan v-cards, 2026-10-23; selection record
`reports/data/v-cards/decks.json`): from `data/decks/v4/corpus`, passed the deck screen, modern format, 2025–2026 events
(Las Vegas GP 2025, GEMPC 2025 / 2026, Worlds 2025, US Nationals 2025 / 2026, San Diego, BWC, ...), at least one V card,
every card in the V-grown card table. 455 DS / 472 LS, median 47 V cards, 52 objectives.

| File | Rows | What |
|---|---:|---|
| `gather-ds.tsv`, `gather-ls.tsv` | 2,972 | the 743 gathering decks (80%), each 4 times on the policy's side, against AdvancedAI; used to find the V prompts the tables lacked |
| `heldout.tsv` | 2,400 | the 184 held-out decks (20% per side and objective, seeded; 44 objectives), paired seats |
| `heldout-800.tsv` | 800 | a fixed 800-row subset of the held-out pairings (before / after comparisons) |

**No policy in this repo has trained on these decks.** Only `models/shc095-vcards` can encode them (its tables hold the
V cards); the other models cannot. Training needs Dojo rewards for the V objectives first: `v31_stage.py` refuses pools
whose objectives have none (40 of 52 here). Evaluation works with `--no-dojo-coverage`; see
`models/shc095-vcards/MODEL_CARD.md`.

Which objectives, epic events and Jedi Tests the decks hold, and which still lack a Dojo reward: [CONTENTS.md](CONTENTS.md).
