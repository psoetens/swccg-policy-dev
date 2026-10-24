"""Plan brainiac-v6: the overlay that turns a Maven build of the feat/brainiac Gemp tree (scripts/brainiac_server.sh sync)
into the v6 Brainiac server, as the training engine does: changed and added sources are compiled with one javac into
<tree>/overlay-classes, which the server puts first on its classpath. The private clone and the tree's own sources are
never changed.

- The knowledge hooks of the training engine, generated with build_knowledge_engine's anchored edits from the tree's
  own files (an anchor the branch changed fails visibly): GameState knowledge, ModifiersLogic stored context, the
  card-selection and pile-verification hooks, DefaultUserFeedback.
- The shared knowledge package (engine/java/.../game/knowledge: observation, context v5, vocabulary, ActorTensors).
  It needs the cards module (ObservationV2's location state), so it cannot live in the logic module of a Maven build.
- The permanent play rules (engine/java/.../game/rules: VisibleCancel, FreeMoveRule), which BrainiacAi applies to its
  own seat (plan permanent-play-rules).
- The v6 production Brainiac (engine/production/.../ai/brainiac), shadowing the branch's classes of the same names.
- The card-name marks of plan titles-and-learner-ablation (build_knowledge_engine.NAME_MARK_EDITS): invisible to clients
  and AIs; Brainiac's v1 release encodes exactly as before.
- The browser worker replaces the tree's copy in its web path (the tree is a build copy; sync restores it).
Sources land in <tree>/overlay-src; the edits are also written to runs/brainiac/engine-overlay.patch for review.

    .venv/bin/python scripts/brainiac_overlay.py .cache/brainiac-build
"""

import argparse
from pathlib import Path
import shutil
import subprocess

import build_knowledge_engine as engine

ROOT = engine.ROOT
LOGIC = engine.LOGIC
KNOWLEDGE = ROOT / "engine/java/com/gempukku/swccgo/game/knowledge"
RULES = ROOT / "engine/java/com/gempukku/swccgo/game/rules"  # plan permanent-play-rules: Brainiac's seat
PRODUCTION = ROOT / "engine/production/com/gempukku/swccgo/ai/brainiac"
PRODUCTION_FILES = ("ActivationAdapter.java", "BrainiacAi.java", "BrainiacRelease.java", "KnowledgeStateEncoder.java",
                    "StateEncoder.java")
MODULES = ("async", "server", "cards", "logic", "common")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tree", type=Path)
    tree = parser.parse_args().tree.resolve()
    source, classes = tree / "overlay-src", tree / "overlay-classes"
    if source.exists() or classes.exists():
        raise SystemExit("The tree already has an overlay; run brainiac_server.sh sync (it rebuilds the tree)")
    engine.BASE, engine.BUILD = tree, source
    edits = [(LOGIC + "game/state/GameState.java", engine.state),
             (LOGIC + "logic/modifiers/querying/ModifiersLogic.java", engine.modifier_context),
             (LOGIC + "logic/decisions/ArbitraryCardsSelectionDecision.java", engine.decision),
             (LOGIC + "logic/timing/DefaultUserFeedback.java", engine.feedback)]
    edits += [(LOGIC + "logic/effects/choose/" + name + ".java", engine.verification)
              for name in ("ChooseCardsFromPileEffect", "ChooseCardsFromMultiplePilesEffect",
                           "ChooseCardFromHandOrCardPilesEffect")]
    # Plan titles-and-learner-ablation A5: the card-name marks and their stripping; GameState gets
    # both its knowledge hooks and the message strip.
    marked = dict.fromkeys(engine.NAME_MARK_EDITS)
    edits = [(relative, (lambda function, relative: (lambda text: engine.name_marks(relative, function(text)))
                         if relative in marked else function)(function, relative)) for relative, function in edits]
    edits += [(relative, (lambda relative: lambda text: engine.name_marks(relative, text))(relative))
              for relative in engine.NAME_MARK_EDITS if relative not in dict(edits)]
    for relative, function in edits:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        engine.edit(relative, function)
    files = [source / relative for relative, _ in edits]
    for directory, base, names in ((KNOWLEDGE, ROOT / "engine/java", [path.name for path in KNOWLEDGE.glob("*.java")]),
                                   (RULES, ROOT / "engine/java", [path.name for path in RULES.glob("*.java")]),
                                   (PRODUCTION, ROOT / "engine/production", PRODUCTION_FILES)):
        package = source / directory.relative_to(base)
        package.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(directory / name, package / name)
            files.append(package / name)
    third_party = [entry for entry in (tree / "cp-async.txt").read_text().strip().split(":")
                   if "/com/gempukku/" not in entry]
    classpath = [str(tree / f"gemp-swccg-{module}/target/classes") for module in MODULES] + third_party
    classes.mkdir()
    subprocess.run(["javac", "-proc:none", "-nowarn", "-cp", ":".join(classpath), "-d", str(classes),
                    *map(str, files)], check=True)
    web = tree / "gemp-swccg-async/src/main/web/js/brainiac"
    shutil.copy2(ROOT / "engine/production/web/brainiacWorker.js", web / "brainiacWorker.js")
    patch = ROOT / "runs/brainiac/engine-overlay.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("".join(engine.PATCHES))
    print(f"{classes}: {len(files)} sources ({len(edits)} engine edits), worker in {web}")


if __name__ == "__main__":
    main()
