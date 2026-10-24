"""Run the unchanged upstream exporter against our exact engine tree, then audit every card entry."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GEMP = Path(os.environ.get("SWCCG_GEMP_REPO", ROOT / ".cache/gemp-swccg"))  # a Gemp clone (scripts/setup.sh)
EXPORTER = "gemp-swccg-server/src/main/java/com/gempukku/swccgo/util/CardDataExporter.java"
DATABASE = "gemp-swccg-cards/src/main/resources/card_blueprint_database.json"
MISSING = "1_245 2_16 3_42 3_77 3_119 3_159 4_38 4_56 4_122 5_60 6_36 6_100 7_23 7_252 8_1 8_110 12_124".split()
CARD = re.compile(r"Card(\d+)_(\d+)(?:_(.+))?\.java$")


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(GEMP), *arguments])


def deletion_history(commit, blueprint):
    """Plan v-cards: the upstream commits that deleted a card's class (any side folder), newest first."""
    set_number, number = blueprint.split("_")[:2]
    pattern = f"src/gemp-swccg-cards/src/main/java/com/gempukku/swccgo/cards/set{set_number}/*/Card{set_number}_{int(number):03d}*.java"
    return git("log", "--diff-filter=D", "--format=%H %s", commit, "--", pattern).decode().strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card_id(path):
    match = CARD.fullmatch(path.name)
    if match is None:
        return None
    set_number, number, suffix = match.groups()
    return f"{set_number}_{int(number)}" + (f"_{suffix}" if suffix else "")


def source_provenance(engine, commit):
    objects = {}
    for line in git("ls-tree", "-r", commit, "src").decode().splitlines():
        metadata, path = line.split("\t", 1)
        if path.endswith(".java"):
            objects[path.removeprefix("src/")] = metadata.split()[2]
    changes = {}
    digests = {}
    for path in sorted(engine.glob("gemp-*/src/**/*.java")):
        relative = str(path.relative_to(engine))
        data = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        digests[relative] = hashlib.sha256(data).hexdigest()
        if objects.get(relative) != digest:
            changes[relative] = {"upstream_git_blob": objects.get(relative), "applied_source_sha256": digests[relative]}
    return {"gemp_commit": commit, "engine_tree": str(engine), "java_source_sha256": digests,
            "applied_java_changes": changes,
            "our_patch_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                 sorted((ROOT / "engine").glob("*.patch"))},
            "v31_overlay_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                   sorted((ROOT / "engine").rglob("*.java"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / ".cache/gemp-knowledge-v5")
    # Plan v-cards: export from another Gemp checkout (the rebased feat/brainiac), into another work dir and report.
    # Defaults: the Amendment 4 export, unchanged.
    parser.add_argument("--gemp-repo", type=Path, default=GEMP)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--work", type=Path, default=ROOT / "runs/knowledge-v3.1/card-database-export")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data/knowledge-v3.1/card-database-export.json")
    parser.add_argument("--deps", type=Path, default=ROOT / ".cache/gemp-knowledge-v5/cp.txt",
                        help="engine dependency classpath file (written by build_knowledge_engine.py)")
    parser.add_argument("--java-env", type=Path, default=ROOT / "scripts/machine_env.sh", help="script that puts javac / java on PATH")
    parser.add_argument("--audit", choices=["amendment4", "general"], default="amendment4",
                        help="general: every removed card must have a deletion in Gemp history (no fixed list)")
    args = parser.parse_args()
    globals()["GEMP"] = args.gemp_repo
    engine = args.engine.resolve()
    commit = git("rev-parse", args.commit).decode().strip()
    upstream_exporter = git("show", f"{commit}:src/{EXPORTER}")
    if (engine / EXPORTER).read_bytes() != upstream_exporter:
        raise ValueError("The engine exporter differs from unchanged upstream source")
    output = ROOT / "data/cards" / f"card_blueprint_database-{commit}.json"
    work = args.work
    if output.exists():
        raise FileExistsError(output)
    work.mkdir(parents=True, exist_ok=False)
    classes = work / "classes"
    classes.mkdir()
    dependencies = [path for path in args.deps.read_text().strip().split(":")
                    if "/com/gempukku/" not in path]
    classpath = ":".join([str(engine / f"gemp-swccg-{module}/target/classes")
                          for module in ("server", "cards", "logic", "common")] + dependencies)
    compile_command = ["bash", "-c", 'source "$1"; exec javac -proc:none -cp "$2" -d "$3" "$4"',
                       "card-export-compile", str(args.java_env), classpath, str(classes), str(engine / EXPORTER)]
    subprocess.run(compile_command, check=True)
    command = ["bash", "-c", 'source "$1"; cd "$2"; exec java -cp "$3" '
               'com.gempukku.swccgo.util.CardDataExporter "$4"',
               "card-export", str(args.java_env), str(engine), str(classes) + ":" + classpath, str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    (work / "output.log").write_text(result.stdout + result.stderr)
    provenance = source_provenance(engine, commit)
    provenance.update(output_sha256=sha(output) if output.exists() else None, command=command,
                      compile_command=compile_command, exporter_sha256=sha(engine / EXPORTER),
                      source_sha256=sha(Path(__file__)))
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    original_rows = json.loads(git("show", f"{commit}:src/{DATABASE}"))
    fresh_rows = json.loads(output.read_text()) if output.exists() else []
    original = {row["cardId"]: row for row in original_rows}
    fresh = {row["cardId"]: row for row in fresh_rows}
    counts = Counter(row["cardId"] for row in fresh_rows)
    added = sorted(fresh.keys() - original.keys())
    removed = sorted(original.keys() - fresh.keys())
    changed = {}
    for blueprint in sorted(fresh.keys() & original.keys()):
        fields = {field: {"before": original[blueprint].get(field), "after": fresh[blueprint].get(field)}
                  for field in original[blueprint].keys() | fresh[blueprint].keys()
                  if original[blueprint].get(field) != fresh[blueprint].get(field)}
        if fields:
            changed[blueprint] = fields
    java_ids = {card_id(path) for path in engine.glob("gemp-swccg-cards/src/main/java/**/Card*.java")}
    java_ids.discard(None)
    unexported = sorted(java_ids - fresh.keys())
    by_set = defaultdict(lambda: {"added": [], "removed": [], "changed": []})
    for kind, blueprints in (("added", added), ("removed", removed), ("changed", changed)):
        for blueprint in blueprints:
            by_set[blueprint.split("_", 1)[0]][kind].append(blueprint)
    deletion_path = "src/gemp-swccg-cards/src/main/java/com/gempukku/swccgo/cards/set225/dark/Card225_004.java"
    removal_history = git("log", "--diff-filter=D", "--format=%H %s", commit, "--", deletion_path).decode().strip()
    explained = removed == ["225_4"] and "225_4" not in java_ids and bool(removal_history)
    removals = {blueprint: deletion_history(commit, blueprint) for blueprint in removed}
    if args.audit == "general":
        explained = all(removals.values()) and not (set(removed) & java_ids)
    failures = result.returncode != 0 or not re.search(r"Failed:\s+0\b", result.stdout)
    passed = (not failures and explained and all(counts[blueprint] == 1 for blueprint in MISSING)
              and all(blueprint.startswith("501_") for blueprint in unexported)
              and not any(count > 1 for count in counts.values()))
    report = {"passed": passed, "output": str(output), "output_sha256": provenance["output_sha256"],
              "gemp_commit": commit, "engine_tree": str(engine), "original_entries": len(original_rows),
              "exported_entries": len(fresh_rows), "load_failures": failures, "added": added, "removed": removed,
              "changed_fields": changed, "by_set": by_set,
              "required_seventeen_counts": {blueprint: counts[blueprint] for blueprint in MISSING},
              "java_classes_without_entry": unexported,
              "removal_225_4": {"title": original.get("225_4", {}).get("title"),
                                 "explanation": "Stale database row; implementation deleted in upstream history.",
                                 "deletion_history": removal_history, "source_path": deletion_path},
              "removals": removals, "audit": args.audit,
              "provenance_file": str(output.with_suffix(".provenance.json")),
              "source_sha256": sha(Path(__file__))}
    destination = args.report
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "exported_entries", "load_failures", "removed")}))
    print(json.dumps({"added": len(added), "changed": len(changed), "by_set": dict(by_set)}))
    if not passed:
        raise SystemExit("Card database export audit failed; stop and report per Amendment 4")


if __name__ == "__main__":
    main()
