# V-card decks

927 tournament decks with Virtual cards (lab plan v-cards, 2026-10-23; selection record
`reports/data/v-cards/decks.json`): from `data/decks/v4/corpus`, passed the deck screen, modern format, 2025–2026 events
(Las Vegas GP 2025, GEMPC 2025 / 2026, Worlds 2025, US Nationals 2025 / 2026, San Diego, BWC, ...), at least one V card,
every card in the V-grown card table. 455 DS / 472 LS, median 47 V cards, 52 objectives.

| File | Rows | What |
|---|---:|---|
| `gather-ds.tsv`, `gather-ls.tsv` | 2,972 | the 743 gathering decks (80%), each 4 times on the policy's side, against AdvancedAI; used to find the V prompts the tables lacked |
| `heldout.tsv` | 2,400 | the 184 held-out decks (20% per side and objective, seeded; 44 objectives), paired seats |
| `heldout-800.tsv` | 800 | a fixed 800-row subset of the held-out pairings (before / after comparisons) |
| `rl_pool_vcards_gathering.txt` | 743 | the gathering decks as a training pool (365 DS, 378 LS) |
| `items-{ds,ls}.tsv`, `items-small-{ds,ls}.tsv`, `items*.json` | 3,584 | Dojo item checks: 128 games per objective / Epic Event / Jedi Test on its decks (`scripts/make_item_schedules.py`) |

Only the V-card models can encode them (`models/shc095-vcards`, untrained on V; `models/vcards-vct`, after the first
V-card run on `rl_pool_vcards_gathering.txt`, lab plan dojo-v-cards, `reports/dojo-v-cards.md`); the other models cannot.
Every objective, Epic Event and Jedi Test here has a Dojo reward (lab plan dojo-v-cards), so the training gate passes.

Which objectives, epic events and Jedi Tests the decks hold, with their Dojo rewards: [CONTENTS.md](CONTENTS.md).
