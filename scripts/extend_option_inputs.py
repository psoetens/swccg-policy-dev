"""Plan option-inputs-and-balance-probe: extend a V3.1 checkpoint's actor with the zero-initialised v2 option inputs.

    extend_option_inputs.py SOURCE.pt OUT.pt

The actor gets cfg.option_inputs = "v2" and the new projections (integer value, range-end flags, the pointed-at card
and location, log1p reference match), registered after every existing parameter, all starting at zero: the extended
actor computes exactly what its parent did. The critic is unchanged (it has no option path). The actor's AdamW state
is remapped by parameter name and extended with the new parameters (no moments yet); the critic's state is kept.
"""
import sys

import torch

from swccg_policy.efficient import EfficientConfig, EfficientPolicy
from swccg_policy.v4_checkpoint import load, save


def extend(cfg, tables, state):
    model = EfficientPolicy(EfficientConfig(**{**cfg, "option_inputs": "v2"}), tables)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("option_") or key.startswith(("option_pool", "option."))
                         for key in missing):
        raise ValueError(f"unexpected checkpoint layout: missing {missing}, unexpected {unexpected}")
    return model, len(missing)


def remap(old, new, state):
    old_names = [name for name, _ in old.named_parameters()]
    new_names = [name for name, _ in new.named_parameters()]
    position = {name: index for index, name in enumerate(new_names)}
    if len(state["param_groups"]) != 1 or list(state["param_groups"][0]["params"]) != list(range(len(old_names))):
        raise ValueError("unexpected optimizer layout")
    remapped = {"state": {position[old_names[index]]: value for index, value in state["state"].items()},
                "param_groups": [{**state["param_groups"][0], "params": list(range(len(new_names)))}]}
    optimizer = torch.optim.AdamW(new.parameters(), lr=state["param_groups"][0]["lr"], weight_decay=0.0)
    optimizer.load_state_dict(remapped)
    return optimizer


def main():
    source, out = sys.argv[1:3]
    actor, critic, tables, checkpoint = load(source, "cpu")
    if actor.cfg.option_inputs != "legacy":
        raise ValueError("checkpoint already has v2 option inputs")
    new_actor, added = extend(actor.cfg.to_dict(), tables, actor.state_dict())
    metadata = dict(checkpoint.get("rl_meta", {}))
    metadata["extended_from"] = str(source)
    metadata["extension"] = "option-inputs-and-balance-probe: option_inputs v2"
    optimizers = None
    if checkpoint.get("optimizer_state"):
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1.0, weight_decay=0.0)
        critic_optimizer.load_state_dict(checkpoint["optimizer_state"][1])
        optimizers = [remap(actor, new_actor, checkpoint["optimizer_state"][0]), critic_optimizer]
    save(out, new_actor, critic, tables, metadata, optimizers)
    print(f"extended {source} -> {out}: +{added} actor tensors")


if __name__ == "__main__":
    main()
