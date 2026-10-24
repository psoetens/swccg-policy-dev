"""Schema-4 compositional probes and real-state attention dilution checks; no weights saved."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from swccg_policy.train_step import TrainingStep, probe_config, probe_loss

from probe_tasks import TASKS, sample as old_sample
from prepare_v31_probe_tables import SCOPE_QUESTIONS
from v31_probe_tasks import GATE_BUDGET, TASK_VERSION, gate_budget, printed_example, printed_split
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import new_models
from swccg_policy.v4_contract import validate
from swccg_policy.v4_data import parse, stack
from swccg_policy.rl import use_tf32


def backgrounds(directory, tables):
    result = []
    for path in sorted(directory.glob("*.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if not 380 <= len(record[8]) <= 430:
                    continue
                item = parse(record, tables)
                keep = ~np.isin(item["tok"][:, 0], [9, 10])
                mapping = np.zeros(len(keep) + 1, np.int32)
                mapping[1:][keep] = np.arange(1, int(keep.sum()) + 1)
                tokens = item["tok"][keep].copy()
                floats = item["tokf"][keep].copy()
                tokens[:, 5] = mapping[tokens[:, 5]]
                edges = tokens[:, 0] == 8
                floats[edges, 0] = mapping[floats[edges, 0].astype(int)]
                item.update(tok=tokens, tokf=floats, opt=np.empty((0, 6), np.int32))
                validate(item, tables.vocabulary, len(tables.cards.ids), actor=True)
                result.append(item)
    if not result:
        raise ValueError("No real states near 400 tokens; padded gate cannot run")
    return result


def sample(rng, task, heldout, power, tables, background=None, scope_priors=True,
           printed_choices=None, missing=None):
    feature = power[:, None]
    while True:
        if task == 4 and printed_choices is not None:
            item, winner = printed_example(rng, power, printed_choices, missing)
        else:
            item, winner = old_sample(rng, task, heldout, feature)
        if task != 4 or power[item["opt"][0, 3]] != power[item["opt"][1, 3]]:
            break
    tokens = item["tok"].astype(np.int32)
    floats = item["tokf"].copy()
    grids = []
    if task == 2:
        if scope_priors:
            item["head"][1] = tables.vocabulary.lookup("q", SCOPE_QUESTIONS[int(item["head"][1])])
        # Schema-3 scope probe rows become one grid per blueprint/owner.
        rows = []
        for blueprint in item["opt"][:, 3]:
            grid = np.zeros((17, 3, 3), np.uint8)
            for token, values in zip(tokens, floats, strict=True):
                if token[1] == blueprint:
                    scope = 0 if token[7] // 2 == 1 else 1
                    grid[2, scope] = [token[6], int(values[1]), 1]
            grids.append(grid.reshape(153))
            rows.append([2, int(blueprint), 0, 1, len(grids), 0, 0, 0])
        tokens = np.asarray(rows, np.int32)
        floats = np.zeros((len(rows), 4), np.float32)
    grid = np.asarray(grids, np.uint8).reshape(-1, 153)
    if background is not None:
        offset = len(background["tok"])
        parent = tokens[:, 5] > 0
        tokens[parent, 5] += offset
        floats[tokens[:, 0] == 8, 0] += offset
        tokens[tokens[:, 0] == 2, 4] += len(background["kgrid"])
        source = item["opt"][:, 2] > 0
        item["opt"][source, 2] += offset
        tokens = np.concatenate((background["tok"], tokens))
        floats = np.concatenate((background["tokf"], floats))
        grid = np.concatenate((background["kgrid"], grid))
        item["head"][4:] = background["head"][4:]
    # Alias companions encode their anchor as the preceding row; keep that contract in permutations.
    blocks = []
    for index, token in enumerate(tokens):
        if token[0] == 7 and token[7] & 8:
            if not blocks:
                raise ValueError("Alias companion lacks its preceding anchor")
            blocks[-1].append(index)
        else:
            blocks.append([index])
    permutation = np.asarray([index for block in rng.permutation(len(blocks)) for index in blocks[block]])
    inverse = np.argsort(permutation)
    tokens = tokens[permutation]
    floats = floats[permutation]
    linked = tokens[:, 5] > 0
    tokens[linked, 5] = inverse[tokens[linked, 5] - 1] + 1
    edges = tokens[:, 0] == 8
    floats[edges, 0] = inverse[floats[edges, 0].astype(int) - 1] + 1
    source = item["opt"][:, 2] > 0
    item["opt"][source, 2] = inverse[item["opt"][source, 2] - 1] + 1
    item.update(tok=tokens, tokf=floats, kgrid=grid, schema_version=4, table_sha=tables.sha, privileged=False)
    # Scope probe blueprints must not duplicate a real background grid binding.
    if len(set(map(tuple, tokens[tokens[:, 0] == 2][:, [1, 3]]))) != len(grid):
        return sample(rng, task, heldout, power, tables, background, scope_priors, printed_choices, missing)
    validate(item, tables.vocabulary, len(tables.cards.ids), actor=True)
    return item, winner


def main(argv=None):
    use_tf32()  # the probes' CUDA precision (plan remove-old-generations, amendment 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("runs/knowledge-v3.1/tables-decipher-probes"))
    parser.add_argument("--fixtures", type=Path, default=Path("runs/knowledge-v3.1/grid-decipher/fixtures"))
    parser.add_argument("--oov-scope-control", action="store_true", help="Reproduce old scope OOV IDs for diagnosis")
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--fact-value", action="store_true")
    parser.add_argument("--post-degree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-grid-indicators", action="store_true", help="Pre-D1b control without unit indicators")
    parser.add_argument("--embedding-init-std", type=float, default=0.02)
    parser.add_argument("--legacy-printed-control", action="store_true", help="Diagnostic former 62-card task")
    parser.add_argument("--small-report", type=Path, help="Passing matching small gate, required for padded runs")
    parser.add_argument("--steps", type=int, default=GATE_BUDGET["steps"])
    parser.add_argument("--batch", type=int, default=GATE_BUDGET["batch"])
    parser.add_argument("--eval", type=int, default=GATE_BUDGET["eval"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[883001, 883002, 883003])
    parser.add_argument("--lr", type=float, default=GATE_BUDGET["lr"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.padded:
        raise ValueError("Amendment 8 padded gates use scripts/probe_padded_curves.py, not the small-gate budget")
    batch_stack = stack
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for full configuration probe gate")
    tables = TableBundle.load(args.tables)
    question_rows = {old: tables.vocabulary.lookup("q", text) for old, text in SCOPE_QUESTIONS.items()}
    if not args.oov_scope_control and (min(question_rows.values()) < 512 or
            np.array_equal(*(tables.vocabulary.priors["q"][index] for index in question_rows.values()))):
        raise ValueError("Scope probe requires distinct known rows with distinct priors")
    power, printed_pools, missing_power, printed_report = printed_split(tables)
    qualifying = (gate_budget(vars(args)) and len(set(args.seeds)) >= 3
                  and not args.oov_scope_control
                  and not args.legacy_printed_control and not args.fact_value and args.post_degree
                  and not args.no_grid_indicators and args.embedding_init_std == 0.02)
    if args.padded:
        if args.small_report is None or not qualifying:
            raise ValueError("Padded gate requires the selected configuration and its passing small report")
        small = json.loads(args.small_report.read_text())
        if (small.get("passed") is not True or small["configuration"]["padded"]
                or small["table_sha"] != tables.sha or small.get("task_version") != TASK_VERSION):
            raise ValueError("Small gate has not passed on these tables and task version")
        for key in (*GATE_BUDGET, "seeds", "fact_value", "post_degree", "no_grid_indicators", "embedding_init_std"):
            if small["configuration"][key] != vars(args)[key]:
                raise ValueError(f"Padded configuration differs from passing small gate: {key}")
        for name in ("src/swccg_policy/efficient.py", "src/swccg_policy/text_priors.py", "scripts/v31_probe_tasks.py"):
            if small["source_sha256"][name] != hashlib.sha256(Path(name).read_bytes()).hexdigest():
                raise ValueError(f"Small gate used different model/task source: {name}")
    noise = backgrounds(args.fixtures, tables) if args.padded else []
    noise_hashes = [hashlib.sha256(b"".join(item[key].tobytes() for key in ("tok", "tokf", "kgrid"))
                                  + item["head"][4:].tobytes()).hexdigest() for item in noise]
    if args.padded and (len(noise) < 2 or set(noise_hashes[::2]) & set(noise_hashes[1::2])):
        raise ValueError("Padded probe requires distinct training and held-out background contents")
    start = time.monotonic()
    report = {"configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}, "table_sha": tables.sha, "seeds": [],
              "backgrounds": len(noise), "passed": None, "gate_eligible": qualifying,
              "background_details": {"train_hashes": noise_hashes[::2], "heldout_hashes": noise_hashes[1::2],
                  "distinct": len(set(noise_hashes)), "token_counts": [len(item["tok"]) for item in noise]},
              "gate_status": "incomplete" if qualifying else "diagnostic_only",
              "task_version": "legacy-62-cards" if args.legacy_printed_control else TASK_VERSION,
              "amendment": 9, "training_step": probe_config(args.lr).to_dict(),
              "template_configuration": {"template_prior_scale": 33.0, "template_identity_std": 1.0},
              "printed_task": printed_report, "scope_question_rows": question_rows,
              "background_note": "Real state rows retained; prior decision references replaced by probe decision.",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                [Path(__file__), Path("scripts/probe_tasks.py"),
                                 Path("scripts/v31_probe_tasks.py"), Path("src/swccg_policy/efficient.py"),
                                 Path("src/swccg_policy/text_priors.py"), Path("src/swccg_policy/train_step.py"),
                                 Path("src/swccg_policy/v4_checkpoint.py")]}}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        model, _ = new_models(tables, seed=seed, fact_value=args.fact_value, post_degree=args.post_degree,
                              grid_indicators=not args.no_grid_indicators,
                              embedding_init_std=args.embedding_init_std)
        model = model.cuda().train()
        training = TrainingStep(model, probe_config(args.lr))
        max_abs_logit = 0.0

        def examples(count, heldout, task=None, missing_case=False):
            rows = []
            for _ in range(count):
                background = None
                if noise:
                    # Disjoint background records in addition to the original compositional split.
                    choices = noise[1::2] if heldout else noise[::2]
                    background = choices[int(rng.integers(len(choices)))]
                item, label = sample(rng, int(rng.integers(5)) if task is None else task,
                                     heldout, power, tables, background, not args.oov_scope_control,
                                     (None if args.legacy_printed_control and not missing_case
                                      else printed_pools[heldout]),
                                     missing_power if missing_case else None)
                rows.append((item, label))
            return rows

        for step in range(args.steps):
            rows = examples(args.batch, False)
            labels = torch.tensor([label for _, label in rows], device="cuda")
            update = training(batch_stack([item for item, _ in rows], "cuda"), probe_loss(labels))
            loss = update["loss"]
            max_abs_logit = max(max_abs_logit, update["max_abs_logit"])
            if step % 100 == 0:
                print(json.dumps({"seed": seed, "step": step, "loss": loss.item(),
                                  "seconds": time.monotonic() - start,
                                  "max_abs_logit": update["max_abs_logit"]}), flush=True)
        model.eval()
        scores = {}
        errors = {}
        categories = {}
        with torch.no_grad():
            model.cache_card_table()
            for task, name in enumerate(TASKS):
                correct = 0
                errors[name] = []
                counts = Counter()
                for begin in range(0, args.eval, args.batch):
                    rows = examples(min(args.batch, args.eval - begin), True, task)
                    logits = model(batch_stack([item for item, _ in rows], "cuda")).cpu().numpy()
                    predicted = logits.argmax(1).tolist()
                    for values, prediction, (item, label) in zip(logits, predicted, rows, strict=True):
                        correct += prediction == label
                        details = {"logits": values.tolist()}
                        category = "all"
                        if task == 2:
                            category = "question_row_" + str(item["head"][1])
                            details["question_row"] = int(item["head"][1])
                            details["asserted_scope_bounds"] = item["kgrid"].reshape(-1, 17, 3, 3)[:, 2].tolist()
                        elif task == 4:
                            ids = item["opt"][:, 3]
                            powers = power[ids] * 5
                            gap = abs(float(powers[0] - powers[1]))
                            category = "power_gap_le_1" if gap <= 1.000001 else "power_gap_gt_1"
                            details.update(blueprints=[tables.cards.ids[index] for index in ids],
                                           titles=[str(tables.cards.cards["titles"][index]) for index in ids],
                                           printed_power=powers.tolist(), power_gap=gap)
                        counts[category + ":total"] += 1
                        counts[category + ":wrong"] += prediction != label
                        if prediction != label and len(errors[name]) < 5:
                            errors[name].append({"label": label, "prediction": prediction,
                                                 **details,
                                                 **{key: value.tolist() for key, value in item.items()
                                                    if isinstance(value, np.ndarray)}})
                scores[name] = correct / args.eval
                categories[name] = dict(counts)
        with torch.no_grad():
            missing_correct = 0
            for begin in range(0, args.eval, args.batch):
                rows = examples(min(args.batch, args.eval - begin), True, 4, missing_case=True)
                guesses = model(batch_stack([item for item, _ in rows], "cuda")).argmax(1).cpu().tolist()
                missing_correct += sum(guess == label for guess, (_, label) in zip(guesses, rows, strict=True))
        report["seeds"].append({"seed": seed, "heldout": scores, "errors": errors,
                                "max_abs_training_logit": max_abs_logit,
                                "missing_power_informational": {"accuracy": missing_correct / args.eval,
                                    "cases": args.eval, "gates": False,
                                    "convention": "One missing-power card uses zero fallback; no ties"},
                                "error_categories": categories,
                                "blueprint_bias": [block.blueprint_bias.detach().cpu().tolist()
                                                   for block in model.decoder]})
        report["seconds"] = time.monotonic() - start
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"seed": seed, "heldout": scores}), flush=True)
    report["threshold_passed"] = len(set(args.seeds)) >= 3 and all(
        score >= 0.99 for seed in report["seeds"] for score in seed["heldout"].values())
    if qualifying:
        report["passed"] = report["threshold_passed"]
        report["gate_status"] = "passed" if report["passed"] else "failed"
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
