# swccg-policy (developer edition)

Neural policies for Star Wars CCG that play in the [Gemp](https://github.com/PlayersCommittee/gemp-swccg) engine,
trained with PPO self-play. This repo is the developer distribution of the lab repo: code, reusable tooling, four
trained policies on Decipher-era decks, two V-card policies (one untrained on V decks, one after the first V-card run) and the deck pools, without the lab's benchmark and probe history.

Start with **[QUICKSTART.md](QUICKSTART.md)**: set up, play against a policy in the browser, run bot-vs-bot
evaluations, and train a checkpoint further.

## What is here

| Path | What |
|---|---|
| `models/{D,cn-ctl,shc-095,shc-r1}/` | V3.1 policies (`policy.pt`, `MODEL_CARD.md`); `models/models.tsv` has sha256s and lineage |
| `models/shc095-vcards/` | shc-095 with tables grown for V cards, not trained on V-card decks (see its model card) |
| `models/vcards-vct/` | shc095-vcards after the first V-card training run (21,233 games on the V gathering decks); 56.8% vs shc095-vcards on the held-out V decks |
| `models/brainiac/` | Browser releases of those policies (ONNX + tables) and the onnxruntime-web runtime, for Brainiac |
| `src/swccg_policy/` | Model (`efficient*.py`), PPO learner (`v4_rl.py`, `ppo_step.py`), tables, trajectories, checkpoints. The V1 imitation code (`rl.py`, `model.py`, `train.py`, ...) stays because V3.1 imports helpers from it |
| `engine/` | Java added to Gemp at build time: knowledge tracker, training harness (TrainGen, PolicyAi), v6 Brainiac + web worker |
| `scripts/` | Tooling (below); `scripts/setup.sh` sets up a fresh clone |
| `native/` | JNI policy library (`build-v3.1/libswccgpolicy.so`, prebuilt; `native/build.sh` rebuilds it) |
| `data/schedules/` | Deck pools and evaluation schedules: `rl_pool_decipher_v3b.txt` (training pool v3b), v3, v2, v1; `eval_decipher_heldout.tsv`, `h2h_pool_v3_only.tsv`; `reports/data/dojo-sweep/schedule-v3b.tsv` |
| `data/schedules/v-cards/` | 927 V-card tournament decks (2025–2026) for V-card training: gathering, held-out, training pool and Dojo item-check schedules ([README](data/schedules/v-cards/README.md)) |
| `data/decks/` | The 1,732 deck files those schedules use (Decipher-era corpus, objective-coverage decks, V-card decks) |
| `data/cards/` | Gemp card database export (the Dojo coverage gate reads it) |
| `runs/knowledge-v3.1/` | Frozen inputs: title table, Decipher tables (`tables-decipher`, for new models), probe tables and fixtures, `gates.json` |
| `reports/` | Architecture docs: [RL loop](reports/architecture-rl-loop.md), [context input](reports/architecture-context.md), [how the network works](reports/how-the-policy-network-works.md), [training configuration](reports/training-configuration.md), [V3.1 contract](reports/knowledge-v3.1-contract.md), [Brainiac v6](reports/brainiac-v6.md), [engine on feat/brainiac](reports/engine-on-feat-brainiac.md) |
| `tools/` | How this repo was exported from the lab (`export_from_lab.py`, manifests, `package_models.py`) |

Lab reports cited in the docs (`reports/<name>.md`, `reports/data/...`) are not part of this distribution; a dangling
link means "lab result, not shipped".

## Tooling map (`scripts/`)

- **Setup / engine:** `setup.sh`, `machine_env.sh`, `build_knowledge_engine.py` (+ `prepare_efficient_overlay.py`),
  `efficient_java_check.sh` (per-run Java overlay + checks), `v31_engine.sh`, `native/{setup,build}.sh`.
- **Training:** `train_arm.sh` (games-budget run with warm-up / decay / crash restarts; wraps `lr2_arm.sh`),
  `v31_stage.py` (the stage supervisor all training goes through), `branch_decay.sh` (LR-decayed branch of a trunk +
  checks), `v31_collect.py` (games with a fixed policy, no learner), `arm_pause.sh`.
- **Evaluation:** `evaluate_v31.py` (paired-seat evaluation vs built-in AIs / a reference), `round_robin_v31.py`,
  `churn_scan.py`, `game_length_metrics.py`, `dojo_*` (objective "Dojo" coverage, stats, reports).
- **Checkpoints:** `export_torchscript.py`, `fresh_like.py` (fresh weights, same network), `reinit_critic.py`,
  `grow_*`, `extend_*` (table / input surgery with optimizer-state remapping).
- **Releases:** `brainiac_release.py` (checkpoint -> browser release), `brainiac_overlay.py`, `brainiac_server.sh`.
- **Probes:** `probe_tasks.py`, `v31_probe_*.py`, `probe_efficient.py`, `night_probe.py` (generic probe runner),
  `prepare_v31_probe_tables.py`, readouts (`probe_readout.py`, `critic_*`, `advantage_readout.py`, ...).
- **Data pipeline** (new tables, decks, card DB): `export_card_database.py`, `build_v*_tables.py`,
  `build_v4_vocabulary.py`, `prepare_*`, `make_*pool*.py`, `deck_*.py`, `stage_objective_decks.py`, `vc_*.py`.
- **Checks:** `check_v31_*.py`, `native_check.py`, `parity_fixture.py`, `check_lfs_policy.py`.

Some analysis scripts were written for one lab experiment; their docstrings name the lab plan. They are kept because
they are reusable, not because every default path in them exists here.

## Git and Git LFS

Git holds code and small configuration only; models, tables, the native library, the browser runtime and every file
over 512 KB are Git LFS (`.gitattributes`, enforced by `scripts/check_lfs_policy.py` / `tests/test_lfs_policy.py`).
Run output goes to `runs/` (git-ignored). Clone with `git lfs` installed, or run `git lfs pull` afterwards.
