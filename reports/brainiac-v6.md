# Brainiac v6: the latest policy in the local Brainiac server, on one code path

Plan: `.claude/plans/brainiac-v6.md`. Date: 2026-10-10. Data: `reports/data/brainiac-v6/`. Runs only on this
machine; weights are pulled from the training server.

**Result:**
- Brainiac serves v6 policies:
  - a release is built from a checkpoint with one command;
  - Java, ONNX and browser inference match the training code;
  - Brainiac won all 3 complete browser games played against the scripted random human, with 0 errors.
- The 20-game e2e run was stopped after 2 games at the user's request ("1 e2e game is good enough for now"), so
  end-goal item 3 is met for 3 games, not 20.
- Peter's own game (item 5) is still open.

## §1 End-goal, item by item

### 1. One command builds a release

```
.venv/bin/python scripts/brainiac_release.py --checkpoint swccg-server:runs/dojo/v6-n1/final.pt --name v6-n1-2026-10-10
```

- **Pulling:** "swccg-server:<path>" is copied read-only over `ssh swccg-server`, together with its
  configuration.json, to `runs/brainiac/pull/`. A local path is used as is.
- **Refused:** anything that is not tensor ABI 5 with context version 5.
- **Written:**
  - `releases-staging/releases/<name>/`: `model.onnx`, the table bundle and the frozen title table `titles.tsv`;
  - `manifest.json`: encoder 5, tensor ABI 5, table SHA, title-table SHA, model SHA, checkpoint source and SHA, and
    the export check.
- **Never overwritten:** an existing release is never replaced, and a failed build's `.partial` folder must be
  removed by hand.
- **Built:** `v6-n1-2026-10-10` from `v6-n1/final.pt`. The model is 45 MB. ONNX matches PyTorch on the 4 export
  records to 2.4e-6 (`release-manifest.json`).

### 2. Exactness

| Check | Result |
|---|---|
| ONNX (CPU) vs PyTorch, 200 recorded v6 decisions, v6 pilot | max 1.3e-7; 199 strict argmax + 1 tie of 7e-9 (`onnx-check-pilot.json`) |
| Production Java bytes vs training `PolicyTensorsV4`, 200 decisions | identical (`BrowserV4Check`, in `browser-check.json`) |
| Browser vs PyTorch, final release, 200 decisions | all 400 runs close (max 3.6e-6) |
| — WebGPU | **pass** |
| — WASM | 199 pass; record 141 fails the strict "top two only" tie rule. Three or more of its 20 options tie exactly (gap 0); the browser's choice is 1.8e-7 below the best. |
| Brainiac encoder vs training encoder, record lines | **identical on 13,866 of 13,866** decisions, 10 AdvancedAI games, both seats (`encoder-check.json`, new `-Dpolicy.checkBrainiacEncoder`) |
| Training bytes after the shared ActorTensors class | ABI check exact; native within 6e-7; 53 of 53 records choose the same option (`abi-check-after-shared-class.json`) |

### 3. It plays

The e2e driver runs headless Chromium with a scripted random human, on Decipher-only decks.

| Game | Human deck vs Brainiac deck | Result | Brainiac decisions | Human decisions | Time |
|---|---|---|---|---|---|
| smoke (dev release, 1 h model) | DO Dark Senate vs P-SE 1998 World Champion (L) | Brainiac wins (Life Force) | 707 | 2,626 | 410 s |
| 0 (final release) | see `e2e-games.jsonl` | Brainiac wins (Life Force) | 424 | 1,712 | 325 s |
| 1 (final release) | see `e2e-games.jsonl` | Brainiac wins (Life Force) | 159 | 818 | 152 s |

- **Errors:** 0 rejected answers, 0 BeginnerAi fallbacks, 0 server exceptions and 0 page errors.
- **Speed:** browser inference takes about 70 ms per decision (WASM).
- **Activation adapter:** 45 and 94 adapter actions in the two server logs. These are Brainiac's own activations,
  the allow-all answers to the human's activations, and the "not activated" pass checks.
- **Unknown keys:** the Brainiac server wrote no unknown-key file during these 3 games. Either no key was unknown,
  which is plausible for about 1,300 Brainiac decisions, or the report path was not exercised. **Unverified.**

### 4. One path

- **Release format:** `BrainiacRelease` accepts only encoder 5 / tensor ABI 5. The old `rl-v3n-2026-09-26` release is
  refused at load with "this server serves only encoder 5 / tensor ABI 5 (v6 policies)". A JVM without
  `-Dpolicy.knowledge=true -Dpolicy.context=5` is refused too (`refusals.txt`).
- **Encoder:** the production `StateEncoder` lost its schema-1 encoder, and `KnowledgeStateEncoder` is schema 4 only.
- **Input conversion:** observation-to-tensor conversion has one implementation, `game/knowledge/ActorTensors`.
  Training's `PolicyTensorsV4` delegates to it (it keeps the native batch packing), and Brainiac uses it directly.
- **Browser worker:** it decodes only the `SWT5` payload (8 floats per token).

### 5. Peter's game

Open: http://localhost:17101/gemp-swccg/, after
`RELEASE=$PWD/releases-staging/releases/v6-n1-2026-10-10 scripts/brainiac_server.sh start`.

## How the server is built

`scripts/brainiac_server.sh sync`:
1. Copies the private `feat/brainiac` source into `~/src/gemp-swccg-agent/build/brainiac-v6`.
2. Builds it with Maven, unchanged.
3. Runs `scripts/brainiac_overlay.py`, which compiles into `overlay-classes` (first on the server classpath), the same
   way the training engine is built:
   - the knowledge engine hooks, made by `build_knowledge_engine`'s anchored edits;
   - the knowledge package;
   - the v6 Brainiac classes.

It also copies the v6 worker over the build's web copy. The private clone is never changed.

- **Why an overlay:** the knowledge package cannot live in Gemp's logic module under Maven, because `ObservationV2`
  uses the cards module, which depends on logic.
- **`start` additions:**
  - sets `-Dpolicy.knowledge=true -Dpolicy.context=5`;
  - writes unknown keys to swccg-policy's `runs/unknown-keys`;
  - logs in as the test admin to switch the fresh hall to operational, without which no game can be created;
  - requires `RELEASE`.

## Changes from the plan

- **Activation adapter** (user decision, added to the plan): Brainiac's seat plays the training activation rule. The
  next plan, permanent-play-rules, adds the cancel and free-move rules the same way.
- **No unit test for the adapter:** it needs a live game, so the evidence is the adapter counts in the e2e logs.
- **Decision encoder not merged:** the production encoder carries GUI labels and a state argument the training one
  lacks. Instead of merging the two, the new line-equality check proves them identical.
- **Release contents:** the release now includes the title table, which training passes as `policy.titleTable`.
  Without it Brainiac failed on its first decision.
- **e2e driver:** `--decipher-only` uses the "Decipher Cards Only" format with Decipher-era sample decks. v6 knows
  only Decipher cards; a random deck with a virtual card was refused by the hall. Hall errors now record the hall
  console and chat.

## Deletion list (for approval; nothing deleted)

Obsolete once this is accepted:
- `engine/production/.../ai/brainiac/BrainiacTensors.java` and `BrainiacTensorsV4.java`: replaced by ActorTensors and
  no longer compiled.
- `scripts/make_release.py`: V1/V2 releases, replaced by `brainiac_release.py`.
- `scripts/stage_knowledge_production.py` and `engine/knowledge-production.patch`: V2 staging into the private clone.
- `scripts/brainiac_golden.py`: V1 golden fixtures.
- `scripts/browser_equality.py`: V1, except for the `PAGE` string that `check_v31_browser.py` imports. That string
  would move into `check_v31_browser.py`.
- `engine/java/.../bench/KnowledgeParityCheck.java` and `FrozenSchema3Check.java`: schema 1–3 production parity. Their
  production counterpart is gone. `ContextPreservationCheck.production` (schema-4 opt-in) calls `FrozenSchema3Check`
  and would fail if switched on, so it belongs to remove-v4.

## Unverified

- Only 3 e2e games.
- WebGPU in browsers other than headless Chromium.
- Speed on slower machines.
- The unknown-key file from a Brainiac server (above).
- How v6 plays against a human.
