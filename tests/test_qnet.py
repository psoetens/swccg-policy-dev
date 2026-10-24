"""Plan q-critic: the option-value network (swccg_policy.qnet): it starts at Q = 0, A_Q is centred under pi, only the
chosen option enters the loss, and a saved Q-net reloads to the same outputs."""
from pathlib import Path

import numpy as np
import pytest
import torch

from swccg_policy.qnet import chosen_option, policy_from_logits, q_advantages

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
NEEDS_DATA = pytest.mark.skipif(not DATA.is_dir() or not CHECKPOINT.exists() or not torch.cuda.is_available(),
                                reason='needs runs/dojo/collect-test, the focus-s2 checkpoint and a GPU')


def test_q_advantages_are_centred_under_pi():
    rng = np.random.default_rng(1)
    values = torch.as_tensor(rng.normal(size=(5, 4)), dtype=torch.float32)
    valid = torch.ones(5, 4, dtype=torch.bool)
    valid[0, 3] = False
    pi = torch.softmax(torch.as_tensor(rng.normal(size=(5, 4)), dtype=torch.float32).masked_fill(~valid, -1e9), -1)
    # A_Q of every option, weighted by pi, sums to 0 per decision
    every = torch.stack([q_advantages(values, valid, pi, torch.full((5,), o)) for o in range(4)], -1)
    np.testing.assert_allclose((pi * every.masked_fill(~valid, 0)).sum(-1).numpy(), 0, atol=1e-6)


def test_chosen_option_and_policy():
    assert chosen_option([3, 1]) == 3 and chosen_option([]) == 0
    pi = policy_from_logits([0.0, np.log(3.0)])
    np.testing.assert_allclose(pi, [0.25, 0.75])


def _setup():
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_trajectory import TrajectoryReader
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load(CHECKPOINT, 'cuda')
    games = TrajectoryReader(base[2], []).load(sorted(trajectory_files(DATA))[0])
    items = [item for game in games[:1] for item in game['items']][:8]
    return base, stack(items, 'cuda'), items


@NEEDS_DATA
def test_from_actor_starts_at_zero_and_reloads(tmp_path):
    from swccg_policy.qnet import from_actor, load_qnet, save_qnet
    base, batch, _ = _setup()
    qnet = from_actor(base[0])
    qnet.packed_attention = False
    with torch.no_grad():
        values = qnet(batch).float()
    assert torch.all(values[batch['omask']] == 0)
    with torch.no_grad():
        qnet.score[-1].bias.fill_(0.25)
    save_qnet(tmp_path / 'q.pt', qnet, base[2].sha, {'note': 'test'})
    again, meta = load_qnet(tmp_path / 'q.pt', base, 'cuda')
    again.packed_attention = False
    with torch.no_grad():
        torch.testing.assert_close(again(batch).float(), qnet(batch).float())
    assert meta == {'note': 'test'}


@NEEDS_DATA
def test_loss_uses_only_the_chosen_option():
    from swccg_policy.qnet import from_actor, q_loss
    base, batch, items = _setup()
    qnet = from_actor(base[0])
    qnet.packed_attention = False
    chosen = torch.zeros(len(items), dtype=torch.long, device='cuda')
    target = torch.ones(len(items), device='cuda')
    loss, values = q_loss(qnet, batch, chosen, target)
    assert float(loss) == pytest.approx(1.0)  # Q = 0 at the start, target 1
    grads = torch.autograd.grad(loss, values, retain_graph=True)[0]
    assert torch.all(grads[:, 1:] == 0) and torch.all(grads[:, 0] != 0)
