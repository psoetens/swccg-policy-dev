# RL training configuration (native rollouts + fast learner) — as of 2026-09-26

> **For the current V3.1 loop** (processes, CPU/GPU split, critic → actor, throughput balance, precision), see
> [`architecture-rl-loop.md`](architecture-rl-loop.md). This file describes the v4 configuration.

The configuration v4 was to be resumed with, and the recommended starting point for the next (from-scratch) run. Everything
below is measured in `reports/rollout-throughput.md` and `reports/learner-throughput.md`; the reference driver is
`scripts/rl_pilot_v4n.sh` (a resume driver; a from-scratch run needs the same pieces with START = the new BC checkpoint).

## Components and settings

| part | setting | why |
|---|---|---|
| GPU sharing | **CUDA MPS** control daemon (user level): `CUDA_MPS_PIPE_DIRECTORY=/tmp/swccg-mps/pipe` exported to every process (path < 108 chars) | without it the learner falls behind the rollouts (GPU time-slicing) |
| rollouts | TrainGen JVM with the in-JVM policy (`PolicyEngine`, `native/libswccgpolicy.so`): `-Dpolicy.native.run=RUN -Dpolicy.native.bundle=RUN/native-bundle -Dpolicy.native.bc=checkpoints/bc-v1/model.ts -Dpolicy.native.lib=native/build/libswccgpolicy.so -Dpolicy.native.lfBeta=0.2 -Dpolicy.native.waitMs=0.5`; **24 threads**; chunks of 2,000 games | ≈ 7 games/s with the learner running (≈ 8.9 alone) |
| rollout JVM memory | `TRAINGEN_HEAP=6g`, default G1 GC (peak RSS ≈ 5.6–7 GB incl. libtorch) | `-Xmx12g` + ParallelGC grew to 12.7 GB and exhausted RAM (reboot) |
| rollout inference precision | fp32 weights, TF32 matmuls (as the Python server); CUDA graphs per bucket | bf16 inference changes 1 in 400 decisions; not needed |
| learner | `rl.py learn --batch-steps 150000 --bytes-per-step-estimate 169 --native-ts --cards CARDS --fast-batch --gpu-stats --skip-bc --fused-adam --load-workers 6 --bf16` (+ the run's own PPO flags, e.g. v4: `--kl-start 0 --kl-min 0 --value-warmup 2`) | 1,505 → 3,011 steps/s; `--skip-bc` only matters while `kl_coef == 0` |
| learner → rollouts | `--native-ts`: `ckpt/latest.ts` (+ `latest.ts.json`), `snaps/snap-N.ts`; the JVM swaps weights in place every 15 s | no restart, CUDA graphs stay valid |
| card table | `CARDS=data/cards/card_table_v2.npz` for server, learner and bundle (`python -m swccg_policy.native bundle CKPT RUN/native-bundle --cards CARDS`) | v1 lacked six cards; **rebuild with the fixed `cards.py` (unlimited capacity = 16) for a new run** |
| evaluation | Python RL server (`rl.py server --graphs --cards CARDS`) only for the learning curve and final evaluations | unchanged evaluation path |
| safety | `scripts/mem_guard.sh` (stops rollouts below 2 GB free RAM + swap), `scripts/rl_watchdog.sh`, `scripts/hourly_check.sh` | |
| disk | benchmark scripts symlink the last 6 snapshots (copying all 21 per run filled the disk) | |

Build prerequisites for the native library: `native/setup.sh` (local CUDA_HOME from the venv's nvidia wheels, CCCL headers),
`native/build.sh`; harness classes: fork `local-dev`, `mvn -o test-compile -pl gemp-swccg-server -am` in `build/gemp`.

## Expected numbers (RTX 5090, Ryzen 7 9800X3D, 8 cores, 30 GB RAM)

- ≈ 7 games/s ≈ 2,100 learner steps/s produced; learner iteration ≈ 65–75 s for 125–150k steps; batch size stable.
- Memory: rollout JVM ≈ 6–7 GB, learner ≈ 3.3 GB, Python eval server ≈ 1.3 GB; ≥ 10 GB free on this machine.
- GPU ≈ 80 % busy; CPU: the Java engine's rule code is the largest remaining cost (≈ 58 % of the JVM).

## Known issues for the next run

1. **Card features** (fixed in code, not in the existing tables): `pilot/passenger/vehicleCapacity` = Integer.MAX_VALUE for
   Executor, Flagship Executor, Supremacy, Home One gave embeddings of ≈ 8 × 10⁸ and logits in the tens of thousands; it caused
   the approx_kl_old spikes since bc-v1. `cards.py` now maps unlimited to 16 and rejects other values ≥ 1,000.
2. **INTEGER decisions** have learned logits of ≈ ±2,300 (option vectors of norm ≈ 400): deterministic, no exploration. Candidate:
   a bounded logit scale for the score head (e.g. 30 · tanh(z / 30)).
3. `approx_kl_old` is a mean of log-probability differences — one extreme step dominates it. Log a robust version as well
   (`(r − 1) − log r`, or clipped / median); `--kl-diag THRESHOLD` shows the culprit steps (costs a GPU sync per minibatch).
4. After a restart the first iterations train on stale trajectories (played by older weights): high approx_kl_old is expected
   there; move `traj/` aside at restart (the drivers do).
5. Six cards were "unknown" to the RL server with card_table v1; use one card table for BC, server, learner and bundle.
