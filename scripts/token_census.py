"""Count observation tokens by family/view, with per-decision distribution statistics."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import orjson


FAMILIES = ("ENTITY", "LOCATION", "KNOWN", "PILE", "CONSTRAINT", "ORDER", "RESERVED",
            "CONTEXT", "EDGE", "QUESTION", "OPTION")


def statistics(values):
    array = np.asarray(values)
    return {"mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": int(array.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    views = {"actor": [], "critic": []}
    sources = {}
    schemas = set()
    for path in args.input:
        with path.open("rb") as stream:
            sources[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            for line in stream:
                record = orjson.loads(line)
                schema = record[0]
                schemas.add(schema)
                if schema == 4:
                    if len(record) not in (12, 15) or any(len(row) != 153 for row in record[11]):
                        raise ValueError("Pre-grid schema-4 corpus is not a final observation fixture")
                    critic = record[12] if len(record) == 15 else None
                else:
                    critic = record[11] if len(record) >= 13 else None
                for view, tokens in (("actor", record[8]), ("critic", critic)):
                    if tokens is not None:
                        counts = Counter(FAMILIES[token[0]] for token in tokens)
                        counts["TOTAL"] = len(tokens)
                        views[view].append(counts)
    if len(schemas) != 1:
        raise ValueError("Provide one observation schema per census")
    report = {"schema": next(iter(schemas)), "views": {}, "inputs_sha256": sources,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for view, decisions in views.items():
        if decisions:
            report["views"][view] = {"decisions": len(decisions), "families": {
                family: statistics([counts[family] for counts in decisions]) for family in (*FAMILIES, "TOTAL")}}
            mean = report["views"][view]["families"]["TOTAL"]["mean"]
            report["views"][view]["target"] = 300 if view == "actor" else 650
            report["views"][view]["target_met"] = mean <= report["views"][view]["target"]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({view: value["families"]["TOTAL"] for view, value in report["views"].items()}))


if __name__ == "__main__":
    main()
