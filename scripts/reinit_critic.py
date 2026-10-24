"""Plan critic-v2: keep a checkpoint's actor (weights and AdamW state) and give it a freshly initialised critic.

    reinit_critic.py SOURCE.pt OUT.pt [--seed 912001] [--set key=value ...]

The new critic is built from the source's critic_cfg with the --set overrides (values parsed as JSON, else kept as
strings). The default overrides are the critic-v2 package: component_norm_scope=all, input_scaling=v2, win_head=true.
The critic's optimizer state is dropped (the learner then starts it fresh); rl_meta is kept, so iteration and games
continue. The actor is copied bit for bit.
"""
import argparse
import json

import torch

from swccg_policy.efficient import EfficientConfig, EfficientCritic
from swccg_policy.v4_checkpoint import load, save

CRITIC_V2 = {'component_norm_scope': 'all', 'input_scaling': 'v2', 'win_head': True}


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition('=')
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def reinit(source, out, seed, overrides):
    actor, critic, tables, checkpoint = load(source, 'cpu')
    cfg = {**critic.cfg.to_dict(), **overrides}
    torch.manual_seed(seed)
    fresh = EfficientCritic(EfficientConfig(**cfg), tables)
    optimizers = None
    if checkpoint.get('optimizer_state'):
        state = checkpoint['optimizer_state'][0]
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=state['param_groups'][0]['lr'], weight_decay=0.0)
        actor_optimizer.load_state_dict(state)
        optimizers = [actor_optimizer]  # the learner loads states positionally: the critic's starts fresh
    metadata = {key: value for key, value in checkpoint.get('rl_meta', {}).items() if key != 'critic_warmup'}
    metadata['critic_reinit'] = {'source': str(source), 'seed': seed, 'overrides': overrides}
    save(out, actor, fresh, tables, metadata, optimizers)
    return actor, fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('out')
    parser.add_argument('--seed', type=int, default=912001)
    parser.add_argument('--set', nargs='*', default=None, metavar='KEY=VALUE',
                        help='critic_cfg overrides (default: the critic-v2 package)')
    args = parser.parse_args()
    overrides = CRITIC_V2 if args.set is None else parse_overrides(args.set)
    _, fresh = reinit(args.source, args.out, args.seed, overrides)
    print(f'{args.source} -> {args.out}: fresh critic ({sum(p.numel() for p in fresh.parameters())} parameters), '
          f'overrides {overrides}')


if __name__ == '__main__':
    main()
