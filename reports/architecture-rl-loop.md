# RL loop architecture (V3.1 knowledge model) — as of 2026-10-07

How a V3.1 training stage runs: what runs where (CPU / GPU, which process), how data flows between the rollouts and
the learner, and how the critic feeds the actor.

- **Driver:** `scripts/v31_stage.py`, used by the chains (`scripts/dojo_chain.sh`, `scripts/dojo_focus_chain2.sh`,
  `scripts/followup_chain.sh`).
- **Measurements behind the numbers:** `reports/precision-capacity.md`, `reports/rollout-throughput.md`,
  `reports/learner-throughput.md`.
- **Older v4 configuration (MPS, 24 threads):** `reports/training-configuration.md`.

Performance numbers below were measured on the reference platform **AMD 9800X3D + RTX 5090**. They are there to show
the balance between the parts, not as targets. The loop is GPU-limited, so CPU thread counts matter less than the GPU
work per decision and per update.

## 1. Processes and devices

```
CPU, Java (TrainGen JVM, rollout threads: --threads)  GPU, same JVM via JNI (native/build-v3.1/libswccgpolicy.so)
──────────────────────────────────────────────         ─────────────────────────────────────────────────────────
gemp game engine plays until a decision
encoder builds the actor's view (≈ 150 tokens)
  + the critic's privileged view (captureCritic;
    stored for training only, not evaluated)
request queued ──────────────────────────────────▶   batcher waits ≤ 0.5 ms (native.waitMs) for other threads,
                                                       copies the batch to pinned memory → GPU,
                                                       replays a CUDA graph of the actor
logits ◀─────────────────────────────────────────   (TorchScript latest.ts, fp32, TF32 off)
sample the action, continue the game
finished games ──▶ trajectory files (SWTRJ003, 2 games per file, /tmp spool: <run>/traj)

GPU, Python learner process (swccg_policy.v4_rl)
──────────────────────────────────────────────────
collect ≥ 12,000 decisions (batch-steps) from <run>/traj; drop games older than 3 policy versions (max-policy-lag)
critic: V(s) for every decision (privileged view)            ── bf16 autocast (fp32 with --critic-fp32)
GAE (γ 1, λ 0.95): advantages A (normalised), returns R
PPO, 1 epoch, minibatch 64 (token-packed, 32,768 tokens per microbatch):
   actor loss ← A (clip 0.2, entropy 0.01, logit soft cap 30)   ── bf16 autocast, packed attention
   critic loss ← (V − R)²                                       ── bf16 autocast (fp32 with --critic-fp32)
   weights, optimizer state (AdamW, lr 5e-5 actor / 3e-4 critic), log-probabilities, losses: fp32
publish: update the cached TorchScript in place → <run>/ckpt/latest.ts (+ .json iteration);
         a snapshot every 5 updates (6 kept) for the policy:snap opponents
           │
           └──────────▶ rollouts reload latest.ts: the next decisions use the new actor
```

- **CPU:** the game engine, card rules, AdvancedAI / random opponents, state encoding and action sampling, in the
  rollout threads (`--threads`, 16 by default).
- **GPU (one device, shared by both processes):**
  - the actor's inference in the rollouts, through the native library: batching, pinned memory, CUDA graphs, one mutex
    serialising its GPU work;
  - the learner, a separate process with the critic, GAE and both networks' updates.
- **CUDA MPS is off** for V3.1 stages: `CUDA_MPS_PIPE_DIRECTORY` points at a non-existent directory. Every comparison
  uses that setting.
- **Per decision, the data flow is CPU → GPU → CPU.** Removing the Python policy server, batching across threads and
  CUDA graphs are what made this fast (v4: 3.2 → 8.9 games/s on the reference platform).

## 2. How the critic feeds the actor

1. Rollouts record, for each decision:
   - the action taken;
   - its log-probability under the acting policy (`old_logp`);
   - both views (actor and privileged critic);
   - the rewards: the game result ±1 plus signed Dojo rewards at their decision index.
2. The **critic** estimates V(s), the expected rest-of-game reward, for every decision.
3. **GAE** turns rewards and values into a per-decision **advantage**:
   - δₜ = rₜ + V(sₜ₊₁) − V(sₜ);
   - Aₜ = δₜ + 0.95·δₜ₊₁ + 0.95²·δₜ₊₂ + …

   Aₜ means "better or worse than the critic expected after this decision".
4. The **actor** makes actions with Aₜ > 0 more likely and Aₜ < 0 less likely, limited by PPO's clip of ±20% against
   `old_logp`.
5. The **critic** learns towards the returns Rₜ = Aₜ + V(sₜ). These are partly its own values, so targets and
   predictions are close. That's why the GAE-target EV (≈ 0.98) is far above the EV against Monte Carlo reward-to-go
   (≈ 0.55 held out), and why bf16 rounding is significant for the critic.

**Asymmetric actor-critic:**
- The critic sees privileged information: hidden cards, ≈ 275 tokens against the actor's ≈ 150.
- That's safe because the critic only trains; it never plays.
- The richer view gives cleaner advantages.

## 3. Throughput balance

- **Rollouts and learner must keep pace with each other.**
  - If the learner is slower, games wait in the queue, their policy version ages, and games older than 3 versions are
    discarded as stale. That's wasted compute and more off-policy data.
  - If the rollouts are slower, the learner waits (`queue_wait_seconds` > 0). That's harmless.
- **Typical s2 numbers (bf16, reference platform):** an update every ≈ 23 s (PPO ≈ 16 s), the learner waiting ≈ 11 s
  per update, 0 stale discards, about 5,300 counted games per hour.
- **The fp32 critic** makes updates take ≈ 30–36 s on the reference platform:
  - with 16 threads, whole batches went stale;
  - with 10 threads, it was roughly balanced, but a standing backlog kept the lag at ≈ 2.

  Reducing rollout production (fewer `--threads`) is the lever; see `.claude/plans/critic-fp32.md`.
- **Queue limits in the native rollouts:** 36,000 steps or 1 GiB (`native.maxQueueSteps` / `maxQueueBytes`). Above
  that, rollouts pause.

## 4. Precision summary

Details: `reports/precision-capacity.md`.

| Where | Precision | Evidence |
|---|---|---|
| Rollout actor inference | fp32, TF32 off | — |
| Learner actor (forward and backward) | bf16 autocast, fp32 master weights and loss | gradient cosine vs fp32 0.999 (normal minibatch-to-minibatch cosine ≈ 0.07): negligible |
| Learner critic | bf16 autocast (default) | gradient cosine vs fp32 0.56–0.84; norm 1.7× larger: about the size of SGD sampling noise (cosine ≈ 0.41) |
| Learner critic with `--critic-fp32` | fp32, standard attention | gradient cosine 1.0000; ≈ 40% slower updates (A/B running, 2026-10-07) |
