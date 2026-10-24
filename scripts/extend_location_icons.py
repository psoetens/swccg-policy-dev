"""Plan objective-dojo BL-20: extend a V3.1 checkpoint with the zero-initialised location inputs (tensor ABI 5).

BL-20(a) Force icons (`location_icons`) and, with Amendment 2, the BL-20(b) location state (`location_state`:
power, ability and drain for both sides, tokf columns 4-7) are added; the checkpoint becomes tensor ABI 5.

    extend_location_icons.py SOURCE.pt OUT.pt

Actor and critic get cfg.location_icons = True and the new projection (last layer zero), registered after every
existing parameter of the policy (inside the critic's encoder); all existing weights are copied. The AdamW states
are remapped by parameter name and extended with the new parameters (no moments yet), so a continued run keeps its
parent's optimizer state for everything else.
"""
import sys

import torch

from swccg_policy.efficient import EfficientConfig, EfficientCritic, EfficientPolicy
from swccg_policy.v4_checkpoint import load, save


def extend(model_cls, cfg, tables, state):
    model = model_cls(EfficientConfig(**{**cfg, 'location_icons': True, 'location_state': True, 'tensor_abi': 5}), tables)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any('location_icons' not in key and 'location_state' not in key for key in missing):
        raise ValueError(f'unexpected checkpoint layout: missing {missing}, unexpected {unexpected}')
    return model, len(missing)


def main():
    source, out = sys.argv[1:3]
    actor, critic, tables, checkpoint = load(source, 'cpu')
    if actor.cfg.location_icons or actor.cfg.location_state:
        raise ValueError('checkpoint already has location inputs')
    new_actor, added_actor = extend(EfficientPolicy, actor.cfg.to_dict(), tables, actor.state_dict())
    new_critic, added_critic = extend(EfficientCritic, critic.cfg.to_dict(), tables, critic.state_dict())
    metadata = dict(checkpoint.get('rl_meta', {}))
    metadata['extended_from'] = str(source)
    metadata['extension'] = 'objective-dojo BL-20 location_icons + location_state (tensor ABI 5)'
    optimizers = None
    if checkpoint.get('optimizer_state'):
        optimizers = []
        pairs = ((actor, new_actor, checkpoint['optimizer_state'][0]), (critic, new_critic, checkpoint['optimizer_state'][1]))
        for old, new, state in pairs:
            # AdamW state is keyed by parameter position; the critic's new encoder part sits before its value head,
            # so positions are remapped by parameter name. The new parameters start without moments.
            old_names = [name for name, _ in old.named_parameters()]
            new_names = [name for name, _ in new.named_parameters()]
            position = {name: index for index, name in enumerate(new_names)}
            if len(state['param_groups']) != 1 or list(state['param_groups'][0]['params']) != list(range(len(old_names))):
                raise ValueError('unexpected optimizer layout')
            remapped = {'state': {position[old_names[index]]: value for index, value in state['state'].items()},
                        'param_groups': [{**state['param_groups'][0], 'params': list(range(len(new_names)))}]}
            optimizer = torch.optim.AdamW(new.parameters(), lr=state['param_groups'][0]['lr'], weight_decay=0.0)
            optimizer.load_state_dict(remapped)
            optimizers.append(optimizer)
    save(out, new_actor, new_critic, tables, metadata, optimizers)
    print(f'extended {source} -> {out}: +{added_actor} actor and +{added_critic} critic tensors')


if __name__ == '__main__':
    main()
