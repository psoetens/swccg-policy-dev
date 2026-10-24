"""D15 synthetic probe runner: padded/small probe curves for fresh variants or frozen RL encoders.

Reuses the Amendment 9 case generator (`probe_efficient.sample`), the same real-state backgrounds and the
shared `TrainingStep`. Unlike `probe_padded_curves.py` it is not an A9 gate: it screens hypotheses.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_efficient import backgrounds, sample
from probe_tasks import TASKS
from v31_probe_tasks import printed_split
from v31_stage import VARIANTS
from swccg_policy.table_bundle import TableBundle
from swccg_policy.train_step import TrainingStep, probe_config, probe_loss
from swccg_policy.v4_checkpoint import load, new_models
from swccg_policy.v4_data import stack
from swccg_policy.rl import use_tf32


PROBE_TABLES = Path("runs/knowledge-v3.1/tables-decipher-probes")
FIXTURES = Path("runs/knowledge-v3.1/amendment7-padded-fixtures/fixtures")
BATCH = 40
EVALUATIONS = 2000
# Frozen-encoder probe: only the option read-out learns (two cross-attention blocks, option base, pools, score).
HEAD_MODULES = ("decoder", "option", "option_pool", "score")


def freeze_to_head(model):
    trainable = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in HEAD_MODULES)
        trainable += parameter.numel() if parameter.requires_grad else 0
    return trainable


def build(args, tables):
    if args.init:
        actor, critic, grown, _ = load(args.init, "cpu", tables=tables)
        if grown.sha != tables.sha:
            raise ValueError("Checkpoint did not grow onto the probe tables")
    else:
        actor, critic = new_models(tables, seed=args.seed, **VARIANTS[args.variant])
    del critic
    trainable = freeze_to_head(actor) if args.trainable == "head" else sum(p.numel() for p in actor.parameters())
    return actor.cuda().train(), trainable


def main():
    use_tf32()  # the probes' CUDA precision (plan remove-old-generations, amendment 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="a10c")
    parser.add_argument("--init", type=Path, help="RL checkpoint whose actor is grown onto the probe tables")
    parser.add_argument("--trainable", choices=["all", "head"], default="all")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="cosine: decay to 0 over --steps (D15 H5)")
    parser.add_argument("--dataset", type=int, default=0, help="Fixed training pool size; 0 samples fresh cases")
    parser.add_argument("--padded", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=883001)
    parser.add_argument("--evaluations", type=int, default=EVALUATIONS, help="Held-out cases per task")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    tables = TableBundle.load(PROBE_TABLES)
    noise = backgrounds(FIXTURES, tables) if args.padded else None
    power, pools, _, _ = printed_split(tables)
    task_ids = [TASKS.index(task) for task in args.tasks]

    def examples(rng, count, heldout, task=None):
        rows = []
        for _ in range(count):
            background = None
            if noise is not None:
                choices = noise[1::2] if heldout else noise[::2]
                background = choices[int(rng.integers(len(choices)))]
            chosen = task_ids[int(rng.integers(len(task_ids)))] if task is None else task
            rows.append(sample(rng, chosen, heldout, power, tables, background, printed_choices=pools[heldout]))
        return rows

    # Same evaluation RNG convention as probe_padded_curves.py: held-out cases depend only on the seed.
    eval_rng = np.random.default_rng(args.seed + 1000000)
    evaluation = {task: examples(eval_rng, args.evaluations, True, task) for task in range(len(TASKS))}
    train_rng = np.random.default_rng(args.seed)
    pool = examples(train_rng, args.dataset, False) if args.dataset else None
    model, trainable = build(args, tables)
    training = TrainingStep(model, probe_config(args.lr))
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training.optimizer, T_max=args.steps, eta_min=0.0)
    report = {"configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "trainable_parameters": trainable, "table_sha": tables.sha,
              "model_config": {key: model.cfg.to_dict()[key] for key in
                               ("component_norm", "component_norm_scope", "component_label_gain", "logit_cap")},
              "source_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for path in (__file__, "scripts/probe_efficient.py", "src/swccg_policy/efficient.py")},
              "curve": [], "status": "running"}
    started = time.monotonic()
    max_abs_logit = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        if pool is not None:
            rows = [pool[index] for index in train_rng.integers(len(pool), size=BATCH)]
        else:
            rows = examples(train_rng, BATCH, False)
        labels = torch.tensor([label for _, label in rows], device="cuda")
        update = training(stack([item for item, _ in rows], "cuda"), probe_loss(labels))
        max_abs_logit = max(max_abs_logit, update["max_abs_logit"])
        if scheduler is not None:
            scheduler.step()
        if step % args.interval and step != args.steps:
            continue
        model.eval()
        scores = {}
        with torch.no_grad():
            model.cache_card_table()
            for task, name in enumerate(TASKS):
                correct = 0
                for begin in range(0, args.evaluations, BATCH):
                    batch_rows = evaluation[task][begin:begin + BATCH]
                    values = model(stack([item for item, _ in batch_rows], "cuda")).argmax(1).cpu().tolist()
                    correct += sum(value == label for value, (_, label) in zip(values, batch_rows, strict=True))
                scores[name] = correct / args.evaluations
            model.clear_tables()
        model.train()
        point = {"step": step, "accuracy": scores, "loss": float(update["loss"]),
                 "lr": training.optimizer.param_groups[0]["lr"],
                 "max_abs_training_logit": max_abs_logit, "seconds": time.monotonic() - started}
        report["curve"].append(point)
        max_abs_logit = 0.0
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"out": str(args.out), **point}), flush=True)
    report["status"] = "complete"
    report["final"] = report["curve"][-1]["accuracy"]
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
