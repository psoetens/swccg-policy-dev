"""Plan policy-diagnostics: what each policy learns, where it wins and loses (D1-D5). Runs on the machine that has the
runs and evaluations (the server); every subcommand writes one JSON for reports/policy-diagnostics.md.

    policy_diagnostics.py d1 --out d1.json   learning curves: diag-<run>-<checkpoint> vs AdvancedAI and ctl-n1's final
    policy_diagnostics.py d2 --out d2.json   per-game engine stats of the finals' games (wins / losses, seats, objectives)
    policy_diagnostics.py d3 --out d3.json   behaviour per decision family, policy seat vs AdvancedAI seat (exports)
    policy_diagnostics.py d4 --records DIR --out d4.json   confidence and agreement on one fixed decision set
    policy_diagnostics.py d5 --out d5.json   learning signal per decision family in each run's kept trajectories
    policy_diagnostics.py d6 --records DIR --out d6.json   conditioning scan: input component sizes (actor on the fixed
                                             decisions, critic on kept trajectories) and the option layer's parts

Decision families come from the question template and decision type (`family`); the table is part of the report.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import math
from pathlib import Path
import random
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOJO = ROOT / "runs/dojo"
EVALS = ROOT / "runs/knowledge-v3.1/evaluations"
RUNS = ("ctx5-n1", "ctl-n1", "v6-n1", "v7-n1", "cL-n1", "cI-n1", "cI-n2", "cI-lam98-n1", "cI-mb128-n1")

FAMILIES = (  # first match wins: (family, decision type or None, question regex)
    ("sabacc", None, r"sabacc|wild card"),
    ("setup", None, r"^Choose starting|^Select OK to start"),
    ("activate amount", "INTEGER", r"amount of Force to activate"),
    ("menu: activate", None, r"^Choose Activate action or Pass"),
    ("menu: draw", None, r"^Choose Draw action or Pass"),
    ("menu: deploy", None, r"^Choose Deploy action or Pass"),
    ("menu: control (drain)", None, r"^Choose Control action or Pass"),
    ("menu: move", None, r"^Choose Move action or Pass"),
    ("menu: battle", None, r"^Choose Battle action or Pass"),
    ("menu: weapons", None, r"weapons segment action"),
    ("response window", None, r"Optional responses|Required responses"),
    ("force loss / forfeit", None, r"Force to lose|to forfeit|forfeit"),
    ("destiny", None, r"destiny"),
    ("deploy target", None, r"deploy|On which side of"),
    ("move / transfer target", None, r"move|transfer|relocate|embark|land|capacity slot"),
    ("target / select", None, r"[Tt]arget|^Choose character|^Choose location|^Choose card from hand|^Choose Effect"),
    ("pile / search", None, r"Pile|Reserve Deck|take into hand|stack|Lost|Used"),
    ("integer other", "INTEGER", r""),
    ("yes / no", "MULTIPLE_CHOICE", r""),
)


def family(decision_type, question):
    """decision_type "" (trajectory steps carry none): type-specific rules match on the question alone, and the
    type-only rules (empty pattern) are skipped."""
    for name, kind, pattern in FAMILIES:
        if not decision_type and kind is not None and not pattern:
            continue
        if (kind is None or not decision_type or kind == decision_type) and re.search(pattern, question):
            return name
    return "other"


def wilson(wins, games):
    if not games:
        return None
    p, z = wins / games, 1.96
    centre, spread = p + z * z / (2 * games), z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    return [round((centre - spread) / (1 + z * z / games), 3), round((centre + spread) / (1 + z * z / games), 3)]


def evaluation(tag):
    path = EVALS / tag / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data["results"] if data.get("complete") else None


def games_trained(run, checkpoint):
    if checkpoint == "final":
        rows = (DOJO / run / "learner.jsonl").read_text().splitlines()
        return json.loads(rows[-1])["games_total"]
    meta = DOJO / run / "archive" / f"{checkpoint}.json"
    return json.loads(meta.read_text())["games"] if meta.exists() else None


def d1(args):
    result = {}
    for run in RUNS:
        points = []
        tags = sorted(path.name for path in EVALS.glob(f"diag-{run}-*"))
        for tag in tags:
            checkpoint = tag[len(f"diag-{run}-"):]
            results = evaluation(tag)
            if not results:
                continue
            point = {"checkpoint": checkpoint, "games": games_trained(run, checkpoint)}
            for opponent, label in (("advanced", "advanced"), ("initial", "ctl_final")):
                if opponent in results:
                    row = results[opponent]
                    point[label] = round(row["win_rate_all_games"], 3)
                    point[label + "_ci"] = [round(value, 3) for value in row["ci95_all_games"]]
                    point[label + "_n"] = row["games"]
            points.append(point)
        points = [point for point in points if point["games"] is not None]
        points.sort(key=lambda point: point["games"])
        slope = {}
        for label in ("advanced", "ctl_final"):
            usable = [point for point in points if label in point]
            if len(usable) >= 2:
                a, b = usable[-2], usable[-1]
                difference = b[label] - a[label]
                se = math.sqrt(a[label] * (1 - a[label]) / a[label + "_n"] + b[label] * (1 - b[label]) / b[label + "_n"])
                slope[label] = {"from_games": a["games"], "to_games": b["games"], "change": round(difference, 3),
                                "ci95": [round(difference - 1.96 * se, 3), round(difference + 1.96 * se, 3)]}
                if len(usable) >= 3 and len({point["games"] for point in usable[-3:]}) == 3:
                    x = np.array([point["games"] for point in usable[-3:]], float)
                    y = np.array([point[label] for point in usable[-3:]], float)
                    slope[label]["last3_per_10k_games"] = round(float(np.polyfit(x / 1e4, y, 1)[0]), 3)
        result[run] = {"points": points, "slope": slope}
    write(args.out, result)


def agent_seat(row):
    return "ds" if "agent.ts" in row["ds_ai"] else "ls" if "agent.ts" in row["ls_ai"] else None


def d2(args):
    result = {}
    for run in RUNS:
        folder = EVALS / f"diag-{run}-final"
        out = {}
        for opponent in ("advanced", "initial"):
            path = folder / f"{opponent}.csv"
            if not path.exists():
                continue
            groups = collections.defaultdict(list)
            objectives = collections.defaultdict(lambda: [0, 0])
            for row in csv.DictReader(path.open()):
                seat = agent_seat(row)
                if seat is None or row["outcome"] != "finished":
                    continue
                other = "ls" if seat == "ds" else "ds"
                stats = json.loads(row["stats"])
                me, them = stats[seat], stats[other]
                won = row["winner"] == seat
                turns = int(row["turns"])
                record = {"turns": turns, "own_drains": me["drains"], "opp_drains": them["drains"],
                          "own_lost_to_drains": me["force_lost_drain"], "own_lost_to_battle": me["force_lost_battle"],
                          "opp_lost_to_battle": them["force_lost_battle"], "own_forfeits": me["forfeits"],
                          "own_force_activated": me["force_activated"], "own_flips": me.get("flips", 0),
                          "own_life_force_end": me["life_force"], "opp_life_force_end": them["life_force"]}
                groups["win" if won else "loss"].append(record)
                groups[seat].append({**record, "won": won})
                objective = me.get("objective") or "none"
                objectives[objective][0] += won
                objectives[objective][1] += 1
            summary = {}
            for key, rows in groups.items():
                summary[key] = {"games": len(rows), **{field: round(float(np.mean([r[field] for r in rows])), 2)
                                                        for field in rows[0] if field != "won"}}
                if key in ("ds", "ls"):
                    summary[key]["win_rate"] = round(float(np.mean([r["won"] for r in rows])), 3)
            summary["by_objective"] = {name: {"wins": w, "games": g} for name, (w, g) in
                                       sorted(objectives.items(), key=lambda item: -item[1][1])}
            out[opponent] = summary
        result[run] = out
    # Do two policies win the same matchups? Per objective of the agent's deck against AdvancedAI.
    pairs = [("cI-n1", "cI-n2"), ("cI-n1", "ctl-n1"), ("cI-n2", "ctl-n1"), ("ctl-n1", "ctx5-n1"), ("v7-n1", "cI-n2")]
    agreement = {}
    for a, b in pairs:
        oa = result.get(a, {}).get("advanced", {}).get("by_objective", {})
        ob = result.get(b, {}).get("advanced", {}).get("by_objective", {})
        shared = [name for name in oa if name in ob and oa[name]["games"] >= 4 and ob[name]["games"] >= 4]
        if len(shared) >= 5:
            ra = rank([oa[name]["wins"] / oa[name]["games"] for name in shared])
            rb = rank([ob[name]["wins"] / ob[name]["games"] for name in shared])
            agreement[f"{a} vs {b}"] = {"objectives": len(shared), "spearman": round(float(np.corrcoef(ra, rb)[0, 1]), 3)}
    result["objective_rank_agreement"] = agreement
    write(args.out, result)


def rank(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def records(folder):
    for path in sorted(Path(folder).glob("shard-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                yield json.loads(line)


def chosen_template(record):
    picks = record[10]
    return record[9][picks[0]][0] if picks and picks[0] < len(record[9]) else "<none>"


def d3(args):
    result = {}
    for run in RUNS:
        folder = EVALS / f"diag-{run}-final"
        if not (folder / "advanced-export").exists():
            continue
        seats = {row["game"]: agent_seat(row) for row in csv.DictReader((folder / "advanced.csv").open())}
        games = collections.Counter()
        stats = collections.defaultdict(lambda: {"decisions": 0, "pass_or_none": 0, "chosen": collections.Counter(),
                                                 "activate_fraction": [], "activate_context": []})
        for record in records(folder / "advanced-export"):
            policy_seat = seats.get(record[1])
            if policy_seat is None:
                continue
            who = "policy" if record[3].lower() == policy_seat else "advanced"
            games[(who, record[1])] = 1
            name = family(record[4], record[5])
            entry = stats[(who, name)]
            entry["decisions"] += 1
            template = chosen_template(record)
            if template in ("<pass>", "<none>") or template.startswith("Pass"):
                entry["pass_or_none"] += 1
            entry["chosen"][template] += 1
            if name == "activate amount" and record[9] and record[10]:
                values = [option[5] for option in record[9] if isinstance(option[5], (int, float))]
                if values and max(values) > 0:
                    fraction = record[9][record[10][0]][5] / max(values)
                    entry["activate_fraction"].append(fraction)
                    # Globals (record[7]): turn us/them, phase, my turn, Life Force us/them, Reserve Deck us/them, ...
                    entry["activate_context"].append((fraction, record[7][6], record[7][0]))
        per_seat_games = collections.Counter(who for who, _ in games)
        out = {}
        for (who, name), entry in stats.items():
            out.setdefault(name, {})[who] = {
                "per_game": round(entry["decisions"] / max(per_seat_games[who], 1), 2),
                "pass_or_none_share": round(entry["pass_or_none"] / entry["decisions"], 3),
                "top_choices": [[t, round(n / entry["decisions"], 3)] for t, n in entry["chosen"].most_common(6)],
                **({"activate_fraction_mean": round(float(np.mean(entry["activate_fraction"])), 3)}
                   if entry["activate_fraction"] else {}),
                **(activation_context(entry["activate_context"]) if len(entry["activate_context"]) >= 10 else {})}
        result[run] = {"games": dict(per_seat_games), "families": out}
    write(args.out, result)


def activation_context(rows):
    """Does the activated share follow the Reserve Deck and the turn? Correlations and Reserve Deck buckets."""
    fraction, reserve, turn = (np.array(column, float) for column in zip(*rows))
    correlation = lambda a, b: round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() and b.std() else None
    buckets = {}
    for label, mask in (("reserve<=8", reserve <= 8), ("reserve 9-20", (reserve > 8) & (reserve <= 20)),
                        ("reserve>20", reserve > 20)):
        if mask.sum():
            buckets[label] = {"decisions": int(mask.sum()), "fraction": round(float(fraction[mask].mean()), 3)}
    return {"activate_vs_reserve_corr": correlation(fraction, reserve), "activate_vs_turn_corr": correlation(fraction, turn),
            "activate_by_reserve": buckets}


def d4(args):
    import torch
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import parse, stack
    pool = collections.defaultdict(list)
    for record in records(args.records):
        if len(record[9]) >= 2 and record[10]:
            pool[family(record[4], record[5])].append(record)
    rng = random.Random(5151)
    sample = []
    for name, rows in sorted(pool.items()):
        sample += rng.sample(rows, min(args.per_family, len(rows)))
    families = [family(r[4], r[5]) for r in sample]
    choices = {}
    summary = {"decisions": len(sample), "per_family": collections.Counter(families)}
    for run in RUNS:
        model, _, tables, _ = load(DOJO / run / "final.pt", "cuda")
        width = 8 if model.cfg.tensor_abi == 5 else 4
        model.cache_card_table()
        argmax, entropy, top = [], [], []
        with torch.no_grad():
            for record in sample:
                record = list(record)
                if model.cfg.tensor_abi == 4:
                    record[8] = [token[:12] for token in record[8]]
                logits = model(stack([parse(record, tables, width=width)], "cuda")).float().cpu().numpy()[0]
                logits = logits[:len(record[9])]
                p = np.exp(logits - logits.max())
                p /= p.sum()
                argmax.append(int(p.argmax()))
                entropy.append(float(-(p * np.log(p + 1e-12)).sum()))
                top.append(float(p.max()))
        choices[run] = argmax
        by = collections.defaultdict(lambda: {"entropy": [], "top": [], "matches_advanced": []})
        for name, a, e, t, record in zip(families, argmax, entropy, top, sample):
            by[name]["entropy"].append(e)
            by[name]["top"].append(t)
            by[name]["matches_advanced"].append(a == record[10][0])
        summary[run] = {name: {key: round(float(np.mean(values)), 3) for key, values in cell.items()}
                        for name, cell in by.items()}
        del model
        torch.cuda.empty_cache()
    agreement = {}
    for i, a in enumerate(RUNS):
        for b in RUNS[i + 1:]:
            same = collections.defaultdict(list)
            for name, x, y in zip(families, choices[a], choices[b]):
                same[name].append(x == y)
                same["all"].append(x == y)
            agreement[f"{a} vs {b}"] = {name: round(float(np.mean(values)), 3) for name, values in same.items()}
    summary["agreement"] = agreement
    write(args.out, summary)


def d5(args):
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_rl import predict_values
    from swccg_policy.v4_trajectory import load_binary
    result = {}
    for run in RUNS:
        segment = DOJO / run
        files = sorted(glob.glob(str(segment / "trajectories" / "*.traj")))
        if not files:
            continue
        lam = json.loads((segment / "configuration.json").read_text())["args"].get("lam") or 0.95
        actor, critic, tables, payload = load(segment / "final.pt", "cuda")
        vocab = torch.load(segment / "final.pt", map_location="cpu", weights_only=False)["vocab"]
        question = {index: text for text, index in vocab["q"].items()}
        games = [game for path in files for game in load_binary(path, tables)]
        items = [item for game in games for item in game["critic_items"]]
        values = predict_values(critic, items, PPOConfig(), 256)
        actor.cache_card_table()
        stats = collections.defaultdict(lambda: {"steps": 0, "advantage": [], "entropy": [], "options": []})
        position = 0
        with torch.no_grad():
            for game in games:
                count = len(game["items"])
                current = values[position:position + count]
                rewards = np.zeros(count)
                rewards[-1] = game["reward"]
                for entry in game.get("dojo_rewards", ()):
                    rewards[int(entry[0])] += entry[1]
                last, advantage = 0.0, np.zeros(count)
                for index in range(count - 1, -1, -1):
                    following = current[index + 1] if index + 1 < count else 0.0
                    last = rewards[index] + following - current[index] + lam * last
                    advantage[index] = last
                for index, item in enumerate(game["items"]):
                    name = family("", question.get(int(item["head"][1]), "?"))
                    cell = stats[name]
                    cell["steps"] += 1
                    cell["advantage"].append(advantage[index])
                    cell["options"].append(len(item["opt"]))
                    if cell["steps"] <= args.entropy_sample:
                        logits = actor(stack([item], "cuda")).float().cpu().numpy()[0][:len(item["opt"])]
                        p = np.exp(logits - logits.max())
                        p /= p.sum()
                        cell["entropy"].append(float(-(p * np.log(p + 1e-12)).sum()))
                position += count
        total = sum(cell["steps"] for cell in stats.values())
        out = {}
        for name, cell in stats.items():
            a = np.array(cell["advantage"])
            out[name] = {"step_share": round(cell["steps"] / total, 3), "mean_abs_advantage": round(float(np.abs(a).mean()), 4),
                         "advantage_std": round(float(a.std()), 4), "mean_options": round(float(np.mean(cell["options"])), 1),
                         "single_option_share": round(float(np.mean(np.array(cell["options"]) <= 1)), 3),
                         "final_entropy": round(float(np.mean(cell["entropy"])), 3) if cell["entropy"] else None}
        result[run] = {"games": len(games), "steps": total, "lam": lam, "families": out}
        del actor, critic
        torch.cuda.empty_cache()
    write(args.out, result)


OPTION_PARTS = ("template+card", "source", "pooled refs", "question", "question x pooled", "question x source",
                "value", "none")


def component_hooks(model, store):
    """Records the mean L2 norm of every named input component (EfficientPolicy.part) over its nonzero rows. Parts
    that the model multiplies by a mask afterwards (reference flags, position, grid, numeric, entity flags) are
    measured before the mask, so on all rows; card, kind, zone, owner and degree apply to every token."""
    original = model.part

    def part(name, value):
        out = original(name, value)
        norms = out.detach().float().norm(dim=-1).flatten()
        norms = norms[norms > 0]
        if len(norms):
            store[name].append(float(norms.mean()))
        return out
    model.part = part
    return original


def d6(args):
    import torch
    from swccg_policy.ppo_step import PPOConfig
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import parse, stack
    from swccg_policy.v4_trajectory import load_binary
    pool = collections.defaultdict(list)
    for record in records(args.records):
        if len(record[9]) >= 2 and record[10]:
            pool[family(record[4], record[5])].append(record)
    rng = random.Random(6161)
    sample = [row for name, rows in sorted(pool.items()) for row in rng.sample(rows, min(args.per_family // 3, len(rows)))]
    result = {}
    for run in RUNS:
        actor, critic, tables, _ = load(DOJO / run / "final.pt", "cuda")
        width = 8 if actor.cfg.tensor_abi == 5 else 4
        actor.cache_card_table()
        components = collections.defaultdict(list)
        captured = {}
        hooks = [actor.option[0].register_forward_hook(lambda module, inputs, output: captured.__setitem__("x", inputs[0])),
                 actor.reference_match.register_forward_hook(
                     lambda module, inputs, output: captured.__setitem__("ref", output))]
        component_hooks(actor, components)
        d = actor.cfg.d
        weight = actor.option[0].weight.detach().float()
        slices = [slice(0, d), slice(d, 2 * d), slice(2 * d, 3 * d), slice(3 * d, 4 * d), slice(4 * d, 5 * d),
                  slice(5 * d, 6 * d), slice(6 * d, 6 * d + 1), slice(6 * d + 1, 6 * d + 2)]
        spread = collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for record in sample:
                record = list(record)
                if actor.cfg.tensor_abi == 4:
                    record[8] = [token[:12] for token in record[8]]
                actor(stack([parse(record, tables, width=width)], "cuda"))
                x = captured["x"][0, :len(record[9])].float()
                name = family(record[4], record[5])
                for label, part in zip(OPTION_PARTS, slices):
                    contribution = x[:, part] @ weight[:, part].T
                    spread[name][label].append(float((contribution - contribution.mean(0)).norm(dim=-1).mean()))
                reference = captured["ref"][0, :len(record[9])].float()
                spread[name]["reference match (in pooled)"].append(
                    float((reference - reference.mean(0)).norm(dim=-1).mean()))
        for hook in hooks:
            hook.remove()
        option = {}
        for name, parts in spread.items():
            cells = {label: round(float(np.mean(values)), 4) for label, values in parts.items()}
            total = sum(value for label, value in cells.items() if label != "reference match (in pooled)") or 1.0
            option[name] = {"spread": cells, "value_share": round(cells["value"] / total, 4),
                            "decisions": len(parts["value"])}
        actor_components = {name: round(float(np.mean(values)), 3) for name, values in sorted(components.items())}
        critic_components = collections.defaultdict(list)
        files = sorted(glob.glob(str(DOJO / run / "trajectories" / "*.traj")))[:args.critic_files]
        if files:
            component_hooks(critic.encoder, critic_components)
            items = [item for path in files for game in load_binary(path, tables) for item in game["critic_items"]]
            with torch.no_grad():
                for start in range(0, min(len(items), args.critic_items), 64):
                    critic(stack(items[start:start + 64], "cuda"))
        result[run] = {"input_scaling": actor.cfg.input_scaling, "component_norm": actor.cfg.component_norm,
                       "actor_components": actor_components,
                       "critic_components": {name: round(float(np.mean(values)), 3)
                                             for name, values in sorted(critic_components.items())},
                       "option_parts": option}
        del actor, critic
        torch.cuda.empty_cache()
    write(args.out, result)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=lambda value: dict(value) if isinstance(value, collections.Counter)
                               else str(value)) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["d1", "d2", "d3", "d4", "d5", "d6"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=Path, help="d4: folder of exported AdvancedAI-game records")
    parser.add_argument("--per-family", type=int, default=150, help="d4: decisions per family")
    parser.add_argument("--entropy-sample", type=int, default=400, help="d5: steps per family scored by the actor")
    parser.add_argument("--runs", nargs="+", help="runs (runs/dojo/<run>, evaluations diag-<run>-*) instead of the nine")
    parser.add_argument("--critic-files", type=int, default=10, help="d6: kept trajectory files for the critic scan")
    parser.add_argument("--critic-items", type=int, default=2000, help="d6: critic decisions scanned")
    args = parser.parse_args()
    if args.runs:
        global RUNS
        RUNS = tuple(args.runs)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
