"""Plan night-integer-fp32: --param-lr-scale (AdamW groups by name, state restored by name) and --actor-fp32."""
import sys

import pytest
import torch
from torch import nn

from swccg_policy import v4_rl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.option_value = nn.Embedding(5, 4)
        self.option_scalars = nn.Linear(3, 4, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, x, index, scalars):
        return self.head(self.body(x) + self.option_value(index) + self.option_scalars(scalars)).sum()


def batch():
    generator = torch.Generator().manual_seed(1)
    return torch.randn(8, 4, generator=generator), torch.arange(8) % 5, torch.randn(8, 3, generator=generator)


def train(model, optimizer, steps=3):
    for _ in range(steps):
        optimizer.zero_grad()
        model(*batch()).backward()
        optimizer.step()


SCALES = [('option_value', 10.0), ('option_scalars', 10.0)]


def test_parse_lr_scales():
    assert v4_rl.parse_lr_scales(['option_value=10', 'option_scalars=10']) == SCALES
    assert v4_rl.parse_lr_scales([]) == []
    for bad in ('option_value', '=10', 'option_value=0', 'option_value=-1'):
        with pytest.raises(ValueError):
            v4_rl.parse_lr_scales([bad])


def test_groups_scale_only_the_prefixed_parameters():
    model = Tiny()
    groups = v4_rl.actor_groups(model, 5e-5, SCALES)
    assert [(group['lr_scale'], group['lr']) for group in groups] == [(1.0, 5e-5), (10.0, 5e-4)]
    assert groups[1]['param_names'] == ['option_value.weight', 'option_scalars.weight']
    assert sum(len(group['params']) for group in groups) == len(list(model.parameters()))
    with pytest.raises(ValueError, match='match no actor parameter'):
        v4_rl.actor_groups(model, 5e-5, [('no_such_module', 10.0)])


def test_state_restored_by_name_from_one_plain_group():
    torch.manual_seed(0)
    model = Tiny()
    plain = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    train(model, plain)
    saved = plain.state_dict()
    grouped = torch.optim.AdamW(v4_rl.actor_groups(model, 5e-5, SCALES), lr=5e-5, weight_decay=0.0)
    assert v4_rl.load_by_name(grouped, saved, [name for name, _ in model.named_parameters()]) == 6
    by_parameter = {id(p): state for p, state in plain.state.items()}
    for parameter, state in grouped.state.items():
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            assert torch.equal(state[key], by_parameter[id(parameter)][key])
    # The current groups keep their learning rates (not the checkpoint's single rate).
    assert [group['lr'] for group in grouped.param_groups] == [5e-5, 5e-4]


def test_factor_one_parameters_follow_the_plain_optimizer():
    """With the state restored by name, the unscaled parameters step exactly as under the old single group; the
    scaled ones move further."""
    torch.manual_seed(0)
    reference, scaled = Tiny(), Tiny()
    scaled.load_state_dict(reference.state_dict())
    plain = torch.optim.AdamW(reference.parameters(), lr=5e-5, weight_decay=0.0)
    train(reference, plain)
    scaled.load_state_dict(reference.state_dict())
    grouped = torch.optim.AdamW(v4_rl.actor_groups(scaled, 5e-5, SCALES), lr=5e-5, weight_decay=0.0)
    v4_rl.load_by_name(grouped, plain.state_dict(), [name for name, _ in scaled.named_parameters()])
    before = {name: p.detach().clone() for name, p in reference.named_parameters()}
    train(reference, plain, 1)
    train(scaled, grouped, 1)
    for (name, a), (_, b) in zip(reference.named_parameters(), scaled.named_parameters()):
        if name.startswith('option_'):
            ratio = ((b - before[name]).norm() / (a - before[name]).norm()).item()
            assert 9.0 < float(ratio) < 11.0, (name, float(ratio))
        else:
            assert torch.allclose(a, b, atol=1e-7), name


def test_grouped_state_round_trips():
    torch.manual_seed(0)
    model = Tiny()
    grouped = torch.optim.AdamW(v4_rl.actor_groups(model, 5e-5, SCALES), lr=5e-5, weight_decay=0.0)
    train(model, grouped)
    again = torch.optim.AdamW(v4_rl.actor_groups(model, 5e-5, SCALES), lr=5e-5, weight_decay=0.0)
    assert v4_rl.load_by_name(again, grouped.state_dict(), []) == 6
    # A plain single-group state whose parameter count differs from the actor's cannot be mapped by order.
    plain = torch.optim.AdamW(model.parameters()).state_dict()
    with pytest.raises(ValueError, match='cannot map by order'):
        v4_rl.load_by_name(again, plain, ['only', 'two'])

def parsed(*extra):
    captured = {}
    argv, original = sys.argv, v4_rl.learn
    try:
        sys.argv = ['v4_rl', '--run', '/tmp/x', '--initial', '/tmp/y.pt', *extra]
        v4_rl.learn = lambda args: captured.setdefault('args', args)
        v4_rl.main()
    finally:
        sys.argv, v4_rl.learn = argv, original
    return captured['args']


def test_learner_flags_parse():
    args = parsed('--actor-fp32', '--param-lr-scale', 'option_value=10', '--param-lr-scale', 'option_scalars=10')
    assert args.actor_fp32 is True and args.bf16 is True and not args.critic_fp32
    assert args.param_lr_scale == ['option_value=10', 'option_scalars=10']
    assert parsed().param_lr_scale == [] and parsed().actor_fp32 is False


def test_stage_passes_the_flags_through():
    import importlib.util
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/v31_stage.py').read_text()
    assert "['--actor-fp32'] if getattr(args, 'actor_fp32', False)" in source
    assert "('--param-lr-scale', value)" in source
    assert importlib.util.find_spec('swccg_policy.v4_rl') is not None
