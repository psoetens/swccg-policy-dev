# Engine on feat/brainiac: one Gemp source for playing, training and testing

Plan: `.claude/plans/engine-on-feat-brainiac.md` (decisions A and B by Peter). Date: 2026-10-11. Data:
`reports/data/engine-on-feat-brainiac/`.

**Result:**
- **One source:** training, evaluation, the Java checks and the Brainiac server now all build their Gemp engine from
  `~/src/gemp-swccg-private` `feat/brainiac` (local; nothing pushed).
- **The training layer lives in swccg-policy:**
  - the harness (`engine/harness/`);
  - the knowledge instrumentation and the snapshots switch (`scripts/build_knowledge_engine.py`);
  - the per-run overlay.
- **The new engine is 6% faster than the old one and plays the same games.**

## §1 End-goal, item by item

### 1. One source

**`feat/brainiac`:** upstream 2026-09-22 plus, in order:
1. PR #3388 (3 fixes) and PR #3390 (4 fixes);
2. Brainiac (2 commits);
3. Han's Dice / Jabba [untested];
4. Malastare;
5. RelocateBetweenLocationsEffect;
6. PR #3405 (2 commits; one conflict resolved, see below);
7. PR #3387 (1 commit applied; its other 2 were already contained);
8. the game-stats speed fix (new, below).

The full list is in `feat-brainiac-commits.txt`. HEAD is f7648c762.

**`scripts/build_knowledge_engine.py`:**
- `git archive` of `SWCCG_GEMP_BRANCH` (default `feat/brainiac`) → `.cache/gemp-base-v4`, Maven test-compile (plain
  branch with Gemp's own tests);
- → `.cache/gemp-knowledge-v4`:
  - the harness copied in;
  - the anchored edits (knowledge instrumentation; snapshots switch);
  - one javac of the instrumented files, the knowledge package, the per-run overlay set and every bench and
    production class into the server test-classes, which come first on every engine classpath.
- The tree is stamped with `GEMP_SOURCE` (repo, branch, commit), and every run's `configuration.json` carries it as
  `gemp_source`.

**Moved or changed:**
- **Harness moved** from the agent repo's `local-dev` into `engine/harness/` (13 files). `GempBench`, the engine
  benchmark, stays in the agent repo; the one method TrainGen used, `loadDeck`, is now `DeckLoader`.
- **Overlay:** the Malastare and Relocate patches are removed from `prepare_efficient_overlay.py`, since the branch
  has them.
- **Scripts on v4:** `efficient_java_check.sh`, `v31_engine.sh` (`SWCCG_ENGINE_TREE`, default v4),
  `prepare_efficient_overlay.py`, `machine_env.sh` (v4 classpath; Maven only where present), `collect_v31_bc.py`,
  `export_card_database.py`, `audit_knowledge_v31_capacity.py`.

**Still naming the old trees:**
- retired pipelines (schema-2/3 relational chain, `traingen*.sh`, `npe_repro.sh`, `knowledge_main.sh`,
  `rl_pilot_v3.sh`, `stage_relational_production.py`);
- `night_check.sh` (it lists the agent repo's build folder);
- `export_card_database.py`'s card-database classpath (`GEMP.parent/build/cp.txt`).
- None of them is used by training, evaluation or the checks.

**Engine changes on the way:**
1. **PR #3405 cherry-pick:** one conflict in `ModifiersLogic`. #3405's side carried PR #3387's
   affects-card-first hunk, merged upstream after the branch's base. The PR's side was taken.
2. **Game stats:** the new engine was first 43% slower than the old one (301 against 526 decisions per game-second).
   The cause was a `local-dev` speed fix that never went upstream: game stats (the UI's stats panel) were recomputed
   after every engine step, also in headless games. The new commit on `feat/brainiac` (Peter's decision) computes them
   only while a game-state listener is attached, and refreshes them on attach.
3. **Snapshots switch:** `local-dev`'s training-only "no revert snapshots" switch is now an edit in the training
   layer, not a Gemp commit.

### 2. Checks on the new engine

| Check | Result |
|---|---|
| pytest | 196 passed (`test_machine_env` updated for the v4 classpath) |
| Java checks | 9 of 9 |
| `check_v31_abi.py` on cI-n2 | Java bytes exact; 53 of 53 records choose the same option, all 4 native configurations (`abi-check.json`) |
| Name-marks identity (pinned, 10 games) | two runs byte-identical; no marker visible (`identity-v4.json`) |
| 2-minute training dry run (local, from cI-n2) | 4 learner updates, 3 rollout chunks |

### 3. Speed

100 seeded AdvancedAI games, 4 threads, the same machine and load, run one after the other.

| Engine | Wall time | Decisions per game-second |
|---|---|---|
| old (v3, `local-dev`) | 91.0 s | 526.5 |
| v4 before the game-stats commit | 160.4 s | 300.9 |
| **v4** | **78.3 s** | **556.3 (+6%)** |

### 4. Behaviour

- **Seeded games, old vs new engine, pinned identity hash:**
  - 50 held-out games: the same winner in 50 of 50, the same decision count in 50 of 50;
  - 10 identity games: 10 of 10 and 10 of 10, and their decision dump is byte-identical.
- None of the fixes the new engine adds happened to trigger in these games.

### 5. Server

**Setup:**
- `.cache/gemp-knowledge-v4` copied (rsync of the built tree; stamp f7648c762).
- The 37 third-party jars in `~/opt/swccg-engine-deps-v4/`, with a `cp.txt` for the server's paths. The server's
  `.env` points `SWCCG_ENGINE_DEPS` there; the old `.env` is saved as `~/opt/swccg-policy.env.pre-v4`, and the old
  deps folder stays.

**Checks:**
- the overlay compiles;
- the 2-game identity check gives decisions byte-identical to this machine's;
- a 2-minute dry run from cI-n2 (tag `v4-dry2`) gives 4 updates, and its `configuration.json` records `gemp_source`;
- cI-n2 against AdvancedAI wins 107 of 128 (83.6% [76.2, 89.0]), against 83.2% before.

**On the way:** the first server attempt failed in every engine script. `machine_env.sh`'s Maven lookup aborted under
`set -euo pipefail` on a machine without Maven. Fixed.
- That attempt also left a work folder `/tmp/swccg-v31-v4-dry` on the server. It is not deleted; the rerun used a new
  tag.

## Not done / open

- **Commits:** none in swccg-policy yet.
- **Pin:** swccg-policy does not pin the `feat/brainiac` commit it expects; the build takes the branch's HEAD and
  records it. A pin file was proposed.
- **Brainiac's production code** exists both on `feat/brainiac` (the PR version) and in `engine/production/` (the v6
  version both overlays use).
- **Kept, not deleted:** `.cache/gemp-knowledge-v3`, `build/gemp` / `local-dev`, the old server deps folder.
