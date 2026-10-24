"""Re-normalize seed keys with Java, rejecting audited residuals without changing observations."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from audit_v4_titles import INTEGER, residual_reasons
from swccg_policy.soft_vocabulary import NORMALIZER
from swccg_policy.v4_vocabulary import FAMILIES


def normalize(counts, titles, title_report, work):
    work.mkdir(parents=True, exist_ok=True)
    requests = [(family, key) for family in FAMILIES for key in sorted(counts[family])]
    request_text = "".join(json.dumps(request, ensure_ascii=False) + "\n" for request in requests)
    command = ["bash", "scripts/efficient_java_check.sh", f"-Dpolicy.titleTable={titles}", "NormalizeSeedKeys"]
    environment = {**os.environ, "SWCCG_V31_CLASSES": str(work / "classes")}
    result = subprocess.run(command, input=request_text, env=environment, capture_output=True, text=True, check=True)
    normalized = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("[") and not
                  re.match(r"\[[0-9.]+s\]\[warning\]", line)]
    if [(family, before) for family, before, after in normalized] != requests:
        raise ValueError("Java normalizer request/response mismatch")
    aliases = [line.split("\t", 1)[1] for line in titles.read_text().splitlines()]
    title_names = {re.sub(r"^(?:•+|<>|◇)+", "", name) for name in aliases if len(name.split()) >= 2}
    title_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(title_names, key=len, reverse=True)))
                               + r")(?!\w)")
    suffixes = json.loads(title_report.read_text())["display_suffixes"]
    accepted = {family: Counter() for family in FAMILIES}
    observed = {family: Counter() for family in FAMILIES}
    dropped = {family: [] for family in FAMILIES}
    mapping = {family: {} for family in FAMILIES}
    for family, before, after in normalized:
        reasons = residual_reasons(after, family, title_pattern, suffixes)
        if family == "context" and INTEGER.search(after):
            reasons.append("standalone_integer")
        count = counts[family][before]
        observed[family][after] += count
        mapping[family][before] = after
        if reasons:
            dropped[family].append({"original": before, "normalized": after, "count": count, "reasons": reasons})
        else:
            accepted[family][after] += count
    return {"families": accepted, "observed_families": observed, "dropped": dropped, "mapping": mapping,
            "normalizer_version": NORMALIZER,
            "command": command, "stderr": result.stderr,
            "titles_sha256": hashlib.sha256(titles.read_bytes()).hexdigest(),
            "normalizer_sha256": hashlib.sha256(Path(
                "engine/java/com/gempukku/swccgo/game/knowledge/TitleMatcher.java").read_bytes()).hexdigest(),
            "audit_rule": "All families: reject residual titles/markers/suffixes/standalone integers/anonymous numbers"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--titles", type=Path, required=True)
    parser.add_argument("--title-report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = {family: Counter() for family in FAMILIES}
    sources = {}
    for path in args.input:
        data = json.loads(path.read_text())
        for family in FAMILIES:
            counts[family].update(data["families"][family])
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = normalize(counts, args.titles, args.title_report, args.work)
    result["input_sha256"] = sources
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": {family: len(keys) for family, keys in result["families"].items()},
                      "dropped": {family: len(keys) for family, keys in result["dropped"].items()}}))


if __name__ == "__main__":
    main()
