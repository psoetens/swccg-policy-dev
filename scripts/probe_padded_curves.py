"""Amendment 9 model/training step with the bounded Amendment 8 paired curves."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from probe_efficient import backgrounds, sample
from probe_tasks import TASKS
from v31_probe_tasks import TASK_VERSION, printed_split
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import new_models
from swccg_policy.v4_data import stack
from swccg_policy.train_step import TrainingStep, probe_config, probe_loss
from swccg_policy.rl import use_tf32


STEPS = 18000
INTERVAL = 2000
BATCH = 40
EVALUATIONS = 2000
SEEDS = (883001, 883002, 883003)


def clear_cache(model):
    if hasattr(model, "clear_tables"):
        model.clear_tables()
    else:
        model._card_table = None


def digest(rows):
    result = hashlib.sha256()
    for item, label in rows:
        result.update(bytes([label]))
        for key in ("tok", "tokf", "opt", "head", "kgrid"):
            result.update(item[key].tobytes())
    return result.hexdigest()


def threshold_history(curve, task):
    hits = [index for index, point in enumerate(curve) if point["accuracy"][task] >= 0.99]
    first = hits[0] if hits else None
    values = [point["accuracy"][task] for point in curve]
    passed = values[-1] >= 0.99
    if passed:
        classification = "passed"
    elif max(values) <= 0.60 and min(values) >= 0.40:
        classification = "representation/attention failure (flat near chance)"
    elif len(values) >= 4 and np.mean(values[-2:]) > np.mean(values[-4:-2]) + 0.005:
        classification = "budget-limited (still rising)"
    else:
        classification = "failed; plateau or instability, cause unresolved"
    return {"first_step_at_99": None if first is None else curve[first]["step"],
            "stayed_at_99": first is not None and all(value >= 0.99 for value in values[first:]),
            "final_accuracy": values[-1] if curve[-1]["step"] == STEPS else None,
            "latest_accuracy": values[-1],
            "classification": classification if curve[-1]["step"] == STEPS else "pending_final_evaluation"}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for axis, task in zip(axes, TASKS, strict=True):
        for variant, seeds in report["runs"].items():
            for seed, result in seeds.items():
                curve = result["curve"]
                axis.plot([point["step"] for point in curve], [100 * point["accuracy"][task] for point in curve],
                          color={"883001": "#0072B2", "883002": "#D55E00", "883003": "#009E73"}[seed],
                          linestyle="-" if variant == "v31" else "--", label=f"{variant} {seed}")
        axis.axhline(99, color="black", linewidth=0.7, linestyle=":")
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("Training steps")
        axis.set_ylim(0, 101)
        axis.set_xlim(0, STEPS)
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Held-out accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6)
    figure.suptitle("A9 padded curves")
    figure.tight_layout(rect=(0, 0.10, 1, 0.91))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    use_tf32()  # the probes' CUDA precision (plan remove-old-generations, amendment 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=Path("reports/data/knowledge-v3.1/padded-curves.json"))
    parser.add_argument("--run", type=Path, default=Path("runs/knowledge-v3.1/amendment9-curves"))
    parser.add_argument("--small-report", type=Path,
                        default=Path("reports/data/knowledge-v3.1/amendment9-small.json"))
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(2)
    tables = TableBundle.load("runs/knowledge-v3.1/tables-decipher-probes")
    small = json.loads(args.small_report.read_text())
    if small["passed"] is not True or small["table_sha"] != tables.sha or small["task_version"] != TASK_VERSION:
        raise ValueError("Matching small gate must pass first")
    expected_training = json.loads(json.dumps(probe_config().to_dict()))
    if small.get("amendment") != 9 or small.get("training_step") != expected_training:
        raise ValueError("Padded gate requires the Amendment 9 shared training step")
    for name in ("src/swccg_policy/efficient.py", "src/swccg_policy/text_priors.py", "scripts/v31_probe_tasks.py",
                 "src/swccg_policy/train_step.py", "src/swccg_policy/v4_checkpoint.py"):
        if small["source_sha256"][name] != hashlib.sha256(Path(name).read_bytes()).hexdigest():
            raise ValueError(f"Model/task source changed after the small gate: {name}")
    fixtures = Path("runs/knowledge-v3.1/amendment7-padded-fixtures/fixtures")
    noise = backgrounds(fixtures, tables)
    power, pools, missing, printed = printed_split(tables)
    source_paths = [Path(__file__), Path("scripts/v31_padded_control.py"), Path("scripts/probe_efficient.py"),
                    Path("scripts/v31_probe_tasks.py"), Path("scripts/v31_probe_variants.py"),
                    Path("src/swccg_policy/efficient.py"), Path("src/swccg_policy/efficient_base.py"),
                    Path("src/swccg_policy/text_priors.py"), Path("src/swccg_policy/v4_checkpoint.py"),
                    Path("src/swccg_policy/train_step.py")]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    report = {"configuration": {"steps": STEPS, "interval": INTERVAL, "lr": 3e-4, "batch": BATCH,
                                 "eval": EVALUATIONS, "seeds": args.seeds, "fact_value": False, "post_degree": True,
                                 "grid_indicators": True, "embedding_init_std": 0.02, "per_scope_tokens": False},
              "table_sha": tables.sha, "task_version": TASK_VERSION, "printed_task": printed,
              "amendment": 9, "training_step": expected_training,
              "template_configuration": {"template_prior_scale": 33.0, "template_identity_std": 1.0},
              "source_sha256": hashes,
              "backgrounds": small.get("background_details"),
              "evaluation_protocol": "Fixed held-out cases per seed; separate RNG, reused at every checkpoint.",
              "classification_rule": "Final >=99 passes; all points 40–60% means flat/chance; late mean gain >0.5pp "
                                     "means still rising; other failures remain unresolved.",
              "runs": {"v31": {}}, "passed": None, "status": "running"}
    report["backgrounds"] = {"count": len(noise), "tokens": [len(item["tok"]) for item in noise],
                             "train_hash": digest([(noise[0], 0)]), "heldout_hash": digest([(noise[1], 0)]),
                             "limitation": "Two distinct states from one game; one train and one held-out background."}
    if report["backgrounds"]["train_hash"] == report["backgrounds"]["heldout_hash"]:
        raise ValueError("Background split overlaps")
    if args.resume:
        report = json.loads(args.out.read_text())
        if report["source_sha256"] != hashes or report["table_sha"] != tables.sha:
            raise ValueError("Resume requires identical source and table hashes")
    args.run.mkdir(parents=True, exist_ok=True)
    for path in source_paths:
        target = args.run / "source" / path.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    started = time.monotonic()

    def persist():
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        plot(report, args.out.with_suffix(".png"))

    persist()
    for seed in args.seeds:
        def examples(rng, count, heldout, task=None, missing_case=False):
            rows = []
            for _ in range(count):
                choices = noise[1::2] if heldout else noise[::2]
                background = choices[int(rng.integers(len(choices)))]
                rows.append(sample(rng, int(rng.integers(5)) if task is None else task, heldout, power, tables,
                                   background, printed_choices=pools[heldout],
                                   missing=missing if missing_case else None))
            return rows

        eval_rng = np.random.default_rng(seed + 1000000)
        evaluation = {task: examples(eval_rng, EVALUATIONS, True, task) for task in range(5)}
        missing_rows = examples(np.random.default_rng(seed + 2000000), EVALUATIONS, True, 4, True)
        for variant in ("v31",):
            key = str(seed)
            result = report["runs"][variant].setdefault(key, {"curve": [], "training_samples": {}})
            if result.get("complete"):
                continue
            train_rng = np.random.default_rng(seed)
            model, critic = new_models(tables, seed=seed)
            del critic
            batch_stack = stack
            eval_rows = evaluation
            info_rows = missing_rows
            result["evaluation_hashes"] = {TASKS[task]: digest(rows) for task, rows in evaluation.items()}
            model = model.cuda().train()
            training = TrainingStep(model, probe_config())
            optimizer = training.optimizer
            max_abs_logit = 0.0
            first_step = 0
            if result["curve"]:
                checkpoint = torch.load(result["checkpoint"], map_location="cuda", weights_only=False)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_rng.bit_generator.state = checkpoint["train_rng"]
                first_step = checkpoint["step"]
            for step in range(first_step + 1, STEPS + 1):
                rows = examples(train_rng, BATCH, False)
                if (step - 1) % INTERVAL == 0:
                    result["training_samples"][str(step)] = digest(rows)
                items = [item for item, _ in rows]
                labels = torch.tensor([label for _, label in rows], device="cuda")
                update = training(batch_stack(items, "cuda"), probe_loss(labels))
                loss = update["loss"]
                max_abs_logit = max(max_abs_logit, update["max_abs_logit"])
                if step % 500 == 0:
                    print(json.dumps({"variant": variant, "seed": seed, "step": step, "loss": loss.item(),
                                      "seconds": time.monotonic() - started,
                                      "max_abs_logit": update["max_abs_logit"]}), flush=True)
                if step % INTERVAL:
                    continue
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
                        for begin in range(0, EVALUATIONS, BATCH):
                            rows = eval_rows[task][begin:begin + BATCH]
                            values = model(batch_stack([item for item, _ in rows], "cuda")).cpu().numpy()
                            for offset, ((item, label), value) in enumerate(zip(rows, values, strict=True)):
                                prediction = int(value.argmax())
                                correct += prediction == label
                                category = str(item["head"][1]) if task == 2 else "all"
                                counts[category + ":total"] += 1
                                counts[category + ":wrong"] += prediction != label
                                if prediction != label and len(errors[name]) < 5:
                                    errors[name].append({"eval_index": begin + offset, "label": label,
                                                         "prediction": prediction, "logits": value.tolist(),
                                                         "question": int(item["head"][1]),
                                                         "option_blueprints": item["opt"][:, 3].tolist()})
                        scores[name] = correct / EVALUATIONS
                        categories[name] = dict(counts)
                    missing_correct = 0
                    for begin in range(0, EVALUATIONS, BATCH):
                        rows = info_rows[begin:begin + BATCH]
                        predictions = model(batch_stack([item for item, _ in rows], "cuda")).argmax(1).cpu().tolist()
                        missing_correct += sum(p == label for p, (_, label) in zip(predictions, rows, strict=True))
                clear_cache(model)
                model.train()
                point = {"step": step, "accuracy": scores, "errors": errors, "categories": categories,
                         "max_abs_training_logit_since_checkpoint": max_abs_logit,
                         "missing_power_informational": missing_correct / EVALUATIONS,
                         "blueprint_bias": [block.blueprint_bias.detach().cpu().tolist() for block in model.decoder]}
                result["curve"].append(point)
                checkpoint_path = args.run / f"{variant}-{seed}-{step:05}.pt"
                torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "train_rng": train_rng.bit_generator.state, "cfg": model.cfg.to_dict(),
                            "table_sha": tables.sha, "source_sha256": hashes}, checkpoint_path)
                result["checkpoint"] = str(checkpoint_path)
                result["thresholds"] = {task: threshold_history(result["curve"], task) for task in TASKS}
                result["complete"] = step == STEPS
                persist()
                max_abs_logit = 0.0
                print(json.dumps({"variant": variant, "seed": seed, "step": step, "heldout": scores}), flush=True)
            del model, optimizer, training, eval_rows, info_rows
            torch.cuda.empty_cache()
    report["local_threshold_passed"] = all(point["final_accuracy"] >= 0.99
                                           for result in report["runs"]["v31"].values()
                                           for point in result["thresholds"].values())
    if set(args.seeds) == set(SEEDS):
        report["passed"] = report["local_threshold_passed"]
        report["status"] = "passed" if report["passed"] else "failed"
    else:
        report["status"] = "worker_complete"
    persist()
    if report["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
