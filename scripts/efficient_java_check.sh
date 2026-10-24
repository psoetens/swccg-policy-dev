#!/usr/bin/env bash
# Compile the schema-4 overlay without changing any frozen build tree.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/machine_env.sh
# Plan engine-on-feat-brainiac: the engine tree (default v5 since 2026-10-24, built from feat/brainiac; v4 = tag engine-v4); an older tree for comparisons.
SWCCG_ENGINE_TREE="${SWCCG_ENGINE_TREE:-$PWD/.cache/gemp-knowledge-v5}"
SWCCG_V31_DEPS=$(tr ':' '\n' < "$SWCCG_ENGINE_DEPS" | grep -v /com/gempukku/ | paste -sd:)
SWCCG_V31_CLASSES="${SWCCG_V31_CLASSES:-$PWD/runs/knowledge-v3.1/check-classes}"
SWCCG_V31_CP="$SWCCG_ENGINE_TREE/gemp-swccg-server/target/test-classes"
for module in server cards logic common; do
    SWCCG_V31_CP="$SWCCG_V31_CP:$SWCCG_ENGINE_TREE/gemp-swccg-$module/target/classes"
done
SWCCG_V31_CP="$SWCCG_V31_CP:$SWCCG_V31_DEPS"
mkdir -p "$SWCCG_V31_CLASSES"
SWCCG_V31_SOURCE="$PWD"
for argument in "$@"; do
    if [[ "$argument" == "-Dpolicy.native.run=$PWD/runs/knowledge-v3.1/vocabulary-corpus" ]]; then
        SWCCG_V31_SOURCE="$PWD/runs/knowledge-v3.1/pre-amendment2-source"
    fi
done
if [[ "$SWCCG_V31_SOURCE" == "$PWD" ]]; then
    .venv/bin/python scripts/prepare_efficient_overlay.py
else
    echo "Pinned pre-amendment-2 corpus sources: $SWCCG_V31_SOURCE/manifest.json"
fi
javac -proc:none -cp "$SWCCG_V31_CP" -d "$SWCCG_V31_CLASSES" \
    "$SWCCG_V31_SOURCE"/runs/knowledge-v3.1/overlay-source/StateEncoder.java \
    "$SWCCG_V31_SOURCE"/runs/knowledge-v3.1/overlay-source/TrainGen.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/AbstractSwccgCardBlueprint.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/TriggerConditions.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/PlayersPlayPhaseActionsInOrderGameProcess.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/GameUtils.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/AbstractAwaitingDecision.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/DefaultActionsEnvironment.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/DefaultSwccgGame.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/GameState.java \
    "$PWD"/runs/knowledge-v3.1/overlay-source/PolicyAi.java \
    "$PWD"/engine/java/com/gempukku/swccgo/cards/ActivateOnce.java \
    "$PWD"/engine/java/com/gempukku/swccgo/game/rules/VisibleCancel.java \
    "$PWD"/engine/java/com/gempukku/swccgo/game/rules/FreeMoveRule.java \
    "$SWCCG_V31_SOURCE"/runs/knowledge-v3.1/overlay-source/PolicyEngine.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/game/knowledge/*.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/KnowledgeStateEncoder.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/PolicyTensors.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/PolicyTensorsV4.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/NativePolicyV4.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/BinaryTrajectoryWriter.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/TrajectoryWriter.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/BinaryTrajectoryCheck.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/BinaryObservation.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/BrainiacEncoderCheck.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/NameMarkCheck.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/CorpusRandom.java \
    "$SWCCG_V31_SOURCE"/engine/java/com/gempukku/swccgo/bench/CorpusControl.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/GameStats.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/ObjectiveDojo.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/ReplayControl.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/CounterfactualProbe.java \
    "$PWD"/engine/java/com/gempukku/swccgo/bench/LocationProbe.java \
    "$SWCCG_V31_SOURCE"/engine/production/com/gempukku/swccgo/ai/brainiac/StateEncoder.java \
    "$SWCCG_V31_SOURCE"/engine/production/com/gempukku/swccgo/ai/brainiac/KnowledgeStateEncoder.java \
    "$PWD"/engine/production/com/gempukku/swccgo/ai/brainiac/ActivationAdapter.java \
    "$PWD"/engine/production/com/gempukku/swccgo/ai/brainiac/BrainiacRelease.java \
    "$SWCCG_V31_SOURCE"/tests/java/*.java
# Plan objective-dojo: the Dojo rewards travel with the classes (ObjectiveDojo loads them as a classpath resource).
mkdir -p "$SWCCG_V31_CLASSES/dojo"
cp "$PWD/data/dojo/rewards.json" "$SWCCG_V31_CLASSES/dojo/rewards.json"
exec java --add-opens java.base/java.util=ALL-UNNAMED -Xmx6g \
    -cp "$SWCCG_V31_CLASSES:$SWCCG_V31_CP" "$@"
