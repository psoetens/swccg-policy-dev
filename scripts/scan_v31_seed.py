"""Bounded setText/counter seed extraction. No question extraction or general data-flow analysis."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


SOURCE = Path("/home/kaltan/src/gemp-swccg-agent/gemp-swccg-public/src")
LEXICAL = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/', re.S)
STRING = re.compile(r'"(?:\\.|[^"\\])*"\Z', re.S)
HELPERS = ("GameUtils.getFullName", "GameUtils.getCardLink", "GameUtils.getCardLinkForGenericLocation")


def without_comments(source):
    return LEXICAL.sub(lambda match: " " * len(match.group()) if match.group().startswith(("//", "/*"))
                       else match.group(), source)


def balanced_end(source, opening):
    depth = 0
    index = opening
    while index < len(source):
        if source[index] in "\"'":
            match = LEXICAL.match(source, index)
            if not match:
                raise ValueError("Unterminated Java literal")
            index = match.end()
            continue
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("Unbalanced Java call")


def pieces(expression):
    start = 0
    index = 0
    while index < len(expression):
        if expression[index] in "\"'":
            match = LEXICAL.match(expression, index)
            if not match:
                raise ValueError("Bad literal")
            index = match.end()
            continue
        if expression[index] == "(":
            index = balanced_end(expression, index) + 1
            continue
        if expression[index] == "+":
            yield expression[start:index].strip()
            start = index + 1
        index += 1
    yield expression[start:].strip()


def candidate(expression, integers):
    text = []
    for part in pieces(expression):
        if STRING.fullmatch(part):
            try:
                text.append(json.loads(part.replace("\\'", "'").replace("\\s", " ")))
            except json.JSONDecodeError:
                return None
        elif any(part.startswith(helper + "(") and balanced_end(part, len(helper)) == len(part) - 1
                 for helper in HELPERS):
            text.append("<c>")
        elif part.removeprefix("this.") in integers:
            text.append("<n>")
        elif re.fullmatch(r"\d+", part):
            text.append(part)
        else:
            return None
    return "".join(text)


def scan(source):
    source = without_comments(source)
    integers = set(re.findall(r"\b(?:int|Integer)\s+([A-Za-z_$][\w$]*)", source))
    for name in re.findall(r"\b(?:String|boolean|float|double|long|Object)\s+([A-Za-z_$][\w$]*)", source):
        integers.discard(name)
    for match in re.finditer(r"\bsetText\s*\(", source):
        end = balanced_end(source, match.end() - 1)
        expression = source[match.end():end]
        yield candidate(expression, integers), expression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("decipher", "virtual"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    options = Counter()
    actions_used = set()
    skipped = Counter()
    source_hashes = {}
    calls = 0
    for path in sorted(SOURCE.glob("gemp-swccg-*/src/main/java/**/*.java")):
        relative = str(path.relative_to(SOURCE))
        match = re.search(r"/cards/set(\d+)/", relative)
        if match:
            number = int(match.group(1))
            if (args.scope == "decipher") != (number < 200):
                continue
            if args.scope == "virtual" and not (200 <= number <= 226 or number == 301):
                continue
        elif args.scope == "virtual":
            continue
        text = path.read_text()
        selected = list(scan(text))
        identifiers = re.findall(r"\bGameTextActionId\.([A-Za-z_$][\w$]*)", without_comments(text))
        if selected or identifiers:
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        actions_used.update(identifiers)
        for key, expression in selected:
            calls += 1
            if key is None:
                skipped[expression] += 1
            else:
                options[key] += 1
    environment = {**os.environ, "SWCCG_V31_CLASSES": "runs/knowledge-v3.1/seed-scan-classes"}
    command = ["bash", "scripts/efficient_java_check.sh", "SeedCounterCatalog"]
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    catalog = json.loads(lines[-1])
    actions = sorted(actions_used & set(catalog["actions"]))
    context = Counter()
    for family in catalog["families"]:
        context["usage." + family] += 1
        prefixes = ["usage." + family]
        if family in ("_startOfPhaseLimitCounters", "_endOfPhaseLimitCounters"):
            prefixes = [prefixes[0] + "." + phase for phase in catalog["phases"]]
        elif family in ("_forceLossLimitCounters", "_captivityLimitCounters", "_raceTotalLimitCounters"):
            prefixes = [prefixes[0] + ".active"]
        for prefix in prefixes:
            for action in actions:
                context[prefix + "." + action] += 1
                if family in ("_gameLimitCounters", "_turnForCardTitleLimitCounters"):
                    context[prefix + "|" + action] += 1
    report = {"scope": args.scope, "families": {"q": {}, "o": options, "context": context},
              "setText_calls": calls, "accepted_calls": sum(options.values()),
              "unique_option_candidates": len(options), "unique_context_candidates": len(context),
              "skipped_expressions": skipped,
              "integer_rule": "File-declared int/Integer names with no conflicting type",
              "title_helpers": HELPERS, "reflection_catalog": catalog, "selected_action_ids": actions,
              "source_root": str(SOURCE), "source_sha256": source_hashes,
              "scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reflection_source_sha256": hashlib.sha256(Path("tests/java/SeedCounterCatalog.java").read_bytes())
                  .hexdigest(),
              "reflection_command": command, "reflection_stderr": result.stderr,
              "question_extraction": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("scope", "setText_calls", "accepted_calls",
                                                 "unique_option_candidates", "unique_context_candidates")}))


if __name__ == "__main__":
    main()
