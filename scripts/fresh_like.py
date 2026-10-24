"""Plan scratch-24h-ab: a freshly initialised model with exactly a reference checkpoint's network (tables, actor and
critic configuration), for training from scratch on today's architecture.

    PYTHONPATH=src:scripts .venv/bin/python scripts/fresh_like.py models/D/policy.pt OUT.pt --seed 990001

Like `v4_checkpoint.new_models` (seed, then actor, then critic), but the configurations are the reference's own
`cfg` / `critic_cfg` (new_models derives the critic from the actor's, which D's critic does not match: it has no
option inputs). Only `template_identity_seed` takes the new seed. The result is refused unless actor and critic
state_dict keys, shapes and dtypes equal the reference's, and it saves without optimizer state, rl_meta at 0 games.
"""
import argparse
from pathlib import Path

import torch

from swccg_policy.efficient import EfficientConfig, EfficientCritic, EfficientPolicy
from swccg_policy.v4_checkpoint import load, save, unpack_tables


def layout(state):
    return {key: (tuple(value.shape), value.dtype) for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference = torch.load(args.reference, map_location='cpu', weights_only=False)
    tables = unpack_tables(reference['tables'])
    actor_cfg = {**reference['cfg'], 'template_identity_seed': args.seed}
    critic_cfg = {**reference['critic_cfg'], 'template_identity_seed': args.seed}
    torch.manual_seed(args.seed)
    actor = EfficientPolicy(EfficientConfig(**actor_cfg), tables)
    critic = EfficientCritic(EfficientConfig(**critic_cfg), tables)
    for name, model, key in (('actor', actor, 'state'), ('critic', critic, 'value_state')):
        if layout(model.state_dict()) != layout(reference[key]):
            raise ValueError(f'{name} parameters differ from {args.reference}')
    for name, model, key in (('actor', actor, 'cfg'), ('critic', critic, 'critic_cfg')):
        cfg = model.cfg.to_dict()
        differ = sorted(k for k in set(cfg) | set(reference[key])
                        if k != 'template_identity_seed' and cfg.get(k) != reference[key].get(k))
        if differ:
            raise ValueError(f'{name} configuration differs from {args.reference}: {differ}')
    save(args.out, actor, critic, tables, {'iteration': 0, 'games': 0, 'steps': 0, 'fresh_seed': args.seed,
                                           'fresh_like': str(args.reference)})
    loaded, loaded_critic, _, checkpoint = load(args.out)  # the production loader accepts it
    same = all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), actor.state_dict().values()))
    if not same or checkpoint['table_sha'] != reference['table_sha']:
        raise ValueError('Saved checkpoint does not round-trip')
    print(f'{args.out}: fresh (seed {args.seed}) with the network of {args.reference}; table {tables.sha[:12]}, '
          f'{sum(p.numel() for p in actor.parameters()):,} actor / {sum(p.numel() for p in critic.parameters()):,} '
          f'critic parameters')


if __name__ == '__main__':
    main()
