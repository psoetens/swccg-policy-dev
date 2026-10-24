"""Plan titles-and-learner-ablation A3: unknown question / option lookups per 1,000 decisions, today's encoding with
tables-v6 against the marked encoding with tables-v7, on the same held-out games (scripts/name_marks_check.py export,
--unmarked and marked, --pin-hash --threads 1, a seed tables-v7 was not built from).

    .venv/bin/python scripts/name_marks_oov.py --v1 runs/name-marks/heldout-v1 --v2 runs/name-marks/heldout-v2 \\
        --out reports/data/titles-and-learner-ablation/oov-heldout.json
"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def counts(folder, vocabulary):
    records, unknown, distinct = 0, Counter(), {"q": Counter(), "o": Counter()}
    for path in sorted((folder / "export").glob("shard-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                records += 1
                for family, key in [("q", record[5]), *(("o", option[0]) for option in record[9])]:
                    if key not in vocabulary[family]:
                        unknown[family] += 1
                        distinct[family][key] += 1
    return {"decisions": records, **{f"unknown_{family}": unknown[family] for family in ("q", "o")},
            **{f"unknown_{family}_per_1000": round(1000 * unknown[family] / records, 3) for family in ("q", "o")},
            "top_unknown_o": distinct["o"].most_common(15), "top_unknown_q": distinct["q"].most_common(10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--tables-v1", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-v6")
    parser.add_argument("--tables-v2", type=Path, default=ROOT / "runs/knowledge-v3.1/tables-v7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = lambda folder: [row[:13] + row[14:] for row in csv.reader((folder / "games.csv").open())]
    result = {"same_games": rows(args.v1) == rows(args.v2),
              "v1_tables_v6": counts(args.v1, json.loads((args.tables_v1 / "vocab.json").read_text())),
              "v2_tables_v7": counts(args.v2, json.loads((args.tables_v2 / "vocab.json").read_text())),
              "v2_tables_v6": counts(args.v2, json.loads((args.tables_v1 / "vocab.json").read_text()))}
    v1, v2 = result["v1_tables_v6"], result["v2_tables_v7"]
    result["option_ratio"] = round(v2["unknown_o"] / v1["unknown_o"], 3) if v1["unknown_o"] else None
    result["question_not_higher"] = v2["unknown_q"] <= v1["unknown_q"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, dict)}
                     | {key: {k: v for k, v in value.items() if not k.startswith("top")}
                        for key, value in result.items() if isinstance(value, dict)}, indent=1))


if __name__ == "__main__":
    main()
