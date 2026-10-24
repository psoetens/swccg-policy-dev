# dev-distribution: a developer edition of swccg-policy

Plan: lab repo `.claude/plans/dev-distribution.md` (approved 2026-10-24: new repo, private collaborator; policies D,
shc-r1, shc-095, cn-ctl; shc-r1 full, the others slim; Gemp `feat/brainiac` to be force-pushed). Evidence:
`reports/dev-distribution/`.

## Completion summary

End-goal (plan §1), item by item. The acceptance ran in a container (`tools/acceptance/`: Ubuntu 24.04, user uid
1000 with an empty home, no JDK / Maven / `~/.m2` / Gemp rig / lab paths; GPU passed through), from a fresh
`git clone` of this repo.

| # | Criterion | Result |
|---|---|---|
| 1 | clone + `git lfs pull` ≤ 1.2 GB, pack ≤ 25 MB, `check_lfs_policy.py` passes | **pack 1.77 MiB** (lab: 184 MB), LFS 1.2 GB (27 unique objects; ~1.26 GB counted per file): **at the 1.2 GB bound, not under it** (slim checkpoints are 172 MB, not the 141 MB the plan estimated). LFS policy: pass (`tests/test_lfs_policy.py`) |
| 2 | `scripts/setup.sh` exits 0; `pytest` 0 failures | setup **ok** (424 s from scratch: LFS, uv sync, JDK + Maven download with pinned checksums, Gemp clone, engine build, D's TorchScript); pytest **243 passed, 27 skipped, 0 failed** |
| 3 | `evaluate_v31.py` shc-r1 vs Rando / AdvancedAI, 32 games, 0 errors | **ok**: Rando Cal 32/32, AdvancedAI 32/32 (0 unfinished) |
| 4 | 1,000-game training from shc-r1 → `final.pt` loads and evaluates (16 vs Rando, 0 errors) | **ok**: `train_arm.sh` 1,006 games in 22 iterations, 9.4 min, 0 stale steps, `games_complete`; rl_meta 551,906 → 552,912; final vs Rando 15/16 (1 unfinished, counted as not won) |
| 5 | Brainiac serves the release; a browser game reaches turn 3 (manual) | **server part ok**: sync (Gemp + v6 overlay), DB from Gemp's Dockerfile, hall operational with the stock `asdf` admin, the hall offers `shc-r1-2026-10-24` with the manifest's model sha256, the served `.onnx` hashes to it, runtime served. **Browser game: not done** (manual check for Peter) |

Not met as written: item 1's size bound (on it, not under), item 5's browser game (unverified). Everything else met.

## What was built

- **Repo** `~/src/swccg-policy-dev` (branch `main`, local only, 6 commits, not pushed): exported from the lab at
  356ed36 **plus its uncommitted working tree** (the s24 changes to `v31_stage.py`, `lr2_arm.sh`, the server-built
  `libswccgpolicy.so`, and the untracked `branch_decay.sh`, `fresh_like.py`). `tools/export_from_lab.py` re-exports;
  files edited here are listed in `tools/export-local.txt` and never overwritten.
- **Kept:** `src/`, `engine/`, `native/` (source + prebuilt library), 142 script files (core chain, data pipeline, analysis,
  probes; 7 one-offs kept because kept code imports them), 58 Python test files + 31 Java checks, V3.1 pools / schedules
  (v3b, v3, v2, v1, held-out, h2h, sweep v3/v3b) and the **805 deck files** they use (2.7 MB), Dojo rewards, 9
  architecture docs, 6 model-input files from `reports/data` (combo map, counter map, card-database export, ...).
- **Left out** (`tools/export-dropped.txt`): `reports/data/**` (542 MB of results), 231 one-off / lab-only script files, 90 result
  reports, 83 plans, HANDOFF / TASKS, V1 corpora and schedules, 15k unused deck files incl. the wiki / decktech scrapes,
  leaderboard, host sync, server sandbox.
- **Models** (`models/models.tsv`): D, cn-ctl, shc-095 slim (172 MB: optimizer state dropped, weights checked
  tensor-equal to the source), shc-r1 full (345 MB). Brainiac releases for all four (`models/brainiac/releases/`; D,
  shc-095, shc-r1 built today with the ONNX export check, cn-ctl copied) + onnxruntime-web 1.30.0. One model card each.
- **Frozen inputs:** title tables, Decipher tables + probe tables + fixtures, `gates.json`, card DB export.
- **Portability changes:** JDK 21.0.12.1+1 / Maven 3.9.9 into `.cache/tools` (`machine_env.sh` falls back to them),
  Gemp clone in `.cache/gemp-swccg`, `brainiac_server.sh` rebuilt for any machine (`db` from Gemp's own MariaDB
  Dockerfile, credentials from `.env`, `asdf` admin), `train_arm.sh` (documented wrapper of `lr2_arm.sh` with the shc
  recipe and pool v3b), `branch_decay.sh` standalone (reference `models/D`), `export_torchscript.py`,
  `check_lfs_policy.py`, `tests/test_portable_paths.py`, Python pinned to 3.12, pytest as a dev dependency.

## Bugs the acceptance found (also present in the lab)

1. **Engine and Brainiac builds needed Gemp in `~/.m2`.** `build_knowledge_engine.py` and `brainiac_server.sh sync`
   ran `dependency:build-classpath` in a second Maven call without `-am`, so the sibling modules resolved from
   `~/.m2`, which the lab machine has from an old `mvn install` (`~/.m2/repository/com/gempukku/swccgo/`). This also
   means the lab's third-party classpath comes from those **stale POMs**. Fix here: one reactor call
   (`test-compile dependency:build-classpath -am`), reactor module folders dropped from `cp.txt`.
2. **`brainiac_overlay.py` crashes at its end** (`engine.PATCHES`, removed from `build_knowledge_engine.py` in lab
   commit 4393bae). The overlay classes are compiled before the crash, so a lab `sync` looks half-successful. Fix
   here: the patch-record write is gone.
3. Not a bug but a requirement: `torch.compile` (one test, `learner_speed`) needs a C compiler; documented.

Lab-side fixes for 1 and 2 are not applied (lab tree is shared with the running s24 work); see "Next".

## Unverified / caveats

- The browser game against Brainiac (item 5) — server side fully checked, client side not.
- Gemp came from a depth-1 mirror of the local `feat/brainiac` (28a27be02), not from GitHub: the fork's
  `feat/brainiac` is still 401777098 until the force-push. `setup.sh` clones `https://github.com/psoetens/gemp-swccg.git`
  by default; that URL path is untested.
- The 27 skipped tests skip on missing lab run data (`runs/dojo/...`) or similar; not reviewed one by one.
- Only Linux x86-64 with an RTX 5090 was tried; the acceptance shares this machine's NVIDIA driver.
- Strength numbers in the model cards are the lab's, measured before 2026-10-24's rules / engine; only the
  acceptance's own games were played on this build.
- Docs cite lab reports that are not shipped (dangling links are documented in README).

## Done after the report (Peter, 2026-10-24: "yes force push", "create the github repo and push")

1. Gemp `feat/brainiac` force-pushed to `psoetens/gemp-swccg` (with lease on 401777098): now 28a27be02, the engine
   this repo was tested with. `scripts/setup.sh`'s default clone URL points there.
2. This repo pushed to the private GitHub repo `psoetens/swccg-policy-dev` (`main`, 27 LFS objects, 1.2 GB); one LFS
   object fetched back from GitHub and hash-checked.
3. The two lab bugs are written up for the lab agent in the lab's `reports/dev-distribution-lab-fixes.md`.
4. The acceptance work directories and their containers / DB volumes were deleted.

Still open: one manual browser game against `shc-r1-2026-10-24` (QUICKSTART §2); a setup run that clones Gemp from
GitHub (the acceptance used a local mirror of the same commit).
