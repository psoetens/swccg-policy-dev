"""Plan multiselect-count (stages A + B): the one extension step for the next run's starting checkpoint.

    PYTHONPATH=scripts:src .venv/bin/python scripts/extend_selection.py SOURCE.pt OUT.pt

1. Tables: the option family grows by the Done template (scripts/add_done_template.py, text prior, exact prefix
   extension), unless the checkpoint's tables already hold it.
2. Actor: cfg.option_selected = True, i.e. the "already selected" flag of an option row (column 1) gets a projection
   into the option's base; zero-initialised and registered last.

The extended checkpoint computes exactly its parent's function on every decision without a Done row, whatever the flag
(tests/test_multiselect_count.py; scripts/multiselect_count/check_extension.py on real decisions). The critic is unchanged
(no option path; its tables grow with the actor's). AdamW states are kept, remapped by parameter name. The grown option
embedding's moments get zero rows for the Done row, and the new projection starts without moments.

Other extensions are separate steps (e.g. extend_dna.py); run this one last or first, it only touches the option table
and adds one parameter.
"""
import argparse
import json
from pathlib import Path

import torch

from add_done_template import DONE, grow_done, text_prior
from swccg_policy.efficient import EfficientConfig, EfficientPolicy
from swccg_policy.v4_checkpoint import load, save, unpack_tables


def extend_actor(actor, tables):
    """The actor rebuilt with option_selected, every existing weight kept (only option_selected.weight is new)."""
    model = EfficientPolicy(EfficientConfig(**{**actor.cfg.to_dict(), "option_selected": True}), tables)
    missing, unexpected = model.load_state_dict(actor.state_dict(), strict=False)
    if unexpected or missing != ["option_selected.weight"]:
        raise ValueError(f"unexpected layout: missing {missing}, unexpected {unexpected}")
    return model.eval()


def remap(old_names, new_model, state):
    """AdamW state of a one-group optimizer over old_names (positional), mapped by name onto new_model's parameters.
    A parameter that grew along dim 0 (an embedding table) keeps its moments, padded with zero rows."""
    if len(state["param_groups"]) != 1 or list(state["param_groups"][0]["params"]) != list(range(len(old_names))):
        raise ValueError("unexpected optimizer layout")
    by_name = {old_names[index]: value for index, value in state["state"].items()}
    parameters = dict(new_model.named_parameters())
    names = list(parameters)
    mapped = {}
    for position, name in enumerate(names):
        if name not in by_name:
            continue
        entry = dict(by_name[name])
        shape = parameters[name].shape
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry[key]
            if moment.shape != shape:
                if moment.shape[1:] != shape[1:] or moment.shape[0] > shape[0]:
                    raise ValueError(f"cannot map optimizer state of {name}: {tuple(moment.shape)} -> {tuple(shape)}")
                entry[key] = torch.cat((moment, moment.new_zeros((shape[0] - moment.shape[0], *shape[1:]))))
        mapped[position] = entry
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=state["param_groups"][0]["lr"], weight_decay=0.0)
    optimizer.load_state_dict({"state": mapped,
                               "param_groups": [{**state["param_groups"][0], "params": list(range(len(names)))}]})
    return optimizer, len(names) - len(mapped)


def extend(source, out, embed=text_prior):
    payload = torch.load(source, map_location="cpu", weights_only=False)
    parent = unpack_tables(payload["tables"])
    grown = parent if DONE in parent.vocabulary.payload["o"] else grow_done(parent, embed)
    old_actor, critic, tables, _ = load(source, payload=payload)
    old_names = [name for name, _ in old_actor.named_parameters()]
    old_critic_names = [name for name, _ in critic.named_parameters()]
    if old_actor.cfg.option_selected:
        raise ValueError("the checkpoint already has the option_selected input")
    actor, critic, tables, _ = load(source, tables=grown, payload=payload)
    actor = extend_actor(actor, tables)
    metadata = {**payload.get("rl_meta", {}), "extended_from": str(source),
                "extension": "multiselect-count: Done template + option_selected", "done_template_parent_sha": parent.sha}
    optimizers = None
    fresh = None
    if payload.get("optimizer_state"):
        actor_optimizer, fresh = remap(old_names, actor, payload["optimizer_state"][0])
        critic_optimizer, critic_fresh = remap(old_critic_names, critic, payload["optimizer_state"][1])
        if critic_fresh:
            raise ValueError("critic layout changed")
        optimizers = [actor_optimizer, critic_optimizer]
    save(out, actor, critic, tables, metadata, optimizers)
    return {"source": str(source), "out": str(out), "parent_sha": parent.sha, "sha": tables.sha,
            "done_id": tables.vocabulary.payload["o"][DONE], "o_rows": len(tables.vocabulary.payload["o"]),
            "optimizer_state": optimizers is not None, "fresh_actor_parameters": fresh}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    print(json.dumps(extend(args.source, args.out)))


if __name__ == "__main__":
    main()
