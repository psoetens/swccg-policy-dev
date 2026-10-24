# Quickstart

Linux x86-64 with an NVIDIA GPU (training and evaluation; tested on RTX 4090 / 5090), ~30 GB free disk, 32 GB RAM.
Needs `git`, `git-lfs`, `curl`, `rsync`, `gcc` (torch.compile, native rebuilds), [`uv`](https://docs.astral.sh/uv/); `docker` for the
browser server.

## 1. Set up

```bash
git clone git@github.com:psoetens/swccg-policy-dev.git swccg-policy && cd swccg-policy
scripts/setup.sh          # LFS objects, uv sync, JDK 21 + Maven into .cache/tools, Gemp feat/brainiac into
                          # .cache/gemp-swccg, the instrumented engine (.cache/gemp-knowledge-v5), D's TorchScript
uv run pytest -q tests    # all pass (tests that need lab-only data are skipped)
```

Machine-specific settings (your own JDK or Gemp clone, ports, passwords) go in `.env`; see `.env.example`.
Everything below runs from the repo root; Python scripts need `PYTHONPATH=src:scripts`.

## 2. Play against a policy in the browser (Brainiac)

The policy runs in your browser (onnxruntime-web, WebGPU or WASM); the Gemp server only encodes the bot's view.

```bash
scripts/brainiac_server.sh sync                     # Gemp feat/brainiac + the v6 Brainiac overlay -> .cache/brainiac-build
RELEASE=models/brainiac/releases/shc-r1-2026-10-24 scripts/brainiac_server.sh start
#   -> http://localhost:17101/gemp-swccg/   log in as asdf / asdf, Hall -> "vs Bot" -> Brainiac (shc-r1-2026-10-24)
scripts/brainiac_server.sh stop
```

`start` also starts a MariaDB container (`swccg-brainiac-db`, built from Gemp's own `docker/gemp_db.Dockerfile`, with
Gemp's seeded users). One server serves one release; for another policy, stop and start with another `RELEASE`, or run
a second server with `PORT=17102 BUILD=...`. Releases: `D-`, `cn-ctl-`, `shc-095-`, `shc-r1-` and `v-cards-shc095-` (not trained on V decks) in
`models/brainiac/releases/`. Build one from your own checkpoint:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/brainiac_release.py --checkpoint runs/dojo/my-arm/final.pt --name my-arm-2026-11-01
```

## 3. Bot-vs-bot evaluation

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_v31.py \
    --checkpoint models/shc-r1/policy.pt --reference models/D/policy.pt --tag shc-r1-check \
    --opponents rando,advanced,initial --games 64 --threads 8
#   rando = Gemp's Rando Cal, advanced = AdvancedAI, random = random legal moves, initial = the --reference policy
#   results: runs/knowledge-v3.1/evaluations/shc-r1-check/results.json (win rate, Wilson 95% CI, game lengths)
```

Games are paired (each pairing played from both seats) on the Decipher held-out schedule
(`data/schedules/eval_decipher_heldout.tsv`; `--schedule reports/data/dojo-sweep/schedule-v3b.tsv` for the 1,110-game
pool-v3b sweep). `--tag` must be new. Head to head between several checkpoints: `scripts/round_robin_v31.py`.

## 4. Train a checkpoint further

```bash
NAME=my-arm START=models/shc-r1/policy.pt GAMES=20000 GPU=0 THREADS=12 scripts/train_arm.sh
#   unattended: setsid nohup scripts/train_arm.sh > runs/dojo/my-arm.nohup 2>&1 < /dev/null &   (same env vars)
tail -f runs/dojo/my-arm.log                        # progress; runs/dojo/my-arm/learner.jsonl per iteration
```

Defaults are the recipe of the shipped shc models: pool v3b, opponents 50% the learner, 20% its snapshots, 15% D,
10% Rando Cal, 5% random; PPO lr 5e-5 / value lr 3e-4, entropy 0.01, GAE λ 0.95, damage shaping (potential form,
weight 0.05). Throughput: about 6.4k games/hour at 12 threads, learner and rollouts on one GPU (AMD 9800X3D + RTX 5090;
1,006 games in 9.4 minutes including start-up), about 8k games/hour at 16 threads with learner and rollouts on two GPUs.
`models/shc-r1` carries its optimizer state (exact resume); the slim models restart Adam's moments.

Decay a trunk to its best greedy strength and check it against D and Rando Cal:

```bash
bash scripts/branch_decay.sh my-arm 0 runs/dojo/my-arm/milestones/trunk.pt   # or models/shc-r1/policy.pt
```

A fresh model with the same network: `scripts/fresh_like.py models/D/policy.pt runs/fresh.pt --seed 1`, then train
from it (`START=runs/fresh.pt`).

## 5. V-card decks (prepared, not trained on yet)

`data/schedules/v-cards/` holds 927 tournament decks with Virtual cards. **No policy here has trained on them.**
`models/shc095-vcards` (shc-095 with V-card tables, untrained on V) is the only model that can play them:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_v31.py --checkpoint models/shc095-vcards/policy.pt \
    --reference models/shc095-vcards/policy.pt --tag vcards-rando --opponents rando --games 800 --threads 16 \
    --schedule data/schedules/v-cards/heldout-800.tsv --title-table models/shc095-vcards/titles.tsv --no-dojo-coverage
RELEASE=models/brainiac/releases/v-cards-shc095-2026-10-23 scripts/brainiac_server.sh start   # in the browser
```

Training on them needs Dojo rewards for the V objectives first (`v31_stage.py` refuses the pool otherwise); see
`data/schedules/v-cards/README.md`.

## 6. Your own probes

`scripts/probe_tasks.py` / `v31_probe_tasks.py` define supervised probe tasks over recorded decisions;
`scripts/night_probe.py` runs one task against a checkpoint; `probe_readout.py` and the `critic_*` readouts analyse
critics. Start from their docstrings; the architecture docs in `reports/` describe the inputs the probes read.

## Troubleshooting

- `uv run pytest` fails in `test_lfs_policy.py::test_lfs_objects_are_pulled`: run `git lfs pull`.
- The engine build fails in Maven: check `java -version` (21) after `source scripts/machine_env.sh`; Maven needs network
  on the first build (`~/.m2`).
- `v31_stage.py` refuses to start: read the last lines of `runs/dojo/<tag>.supervisor.log`; it checks free disk
  (20 GiB), RAM, the Dojo coverage of the pool and the run duration rules (see `reports/preflight.md`).
