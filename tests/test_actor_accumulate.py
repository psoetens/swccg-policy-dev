"""Plan critic-noise Amendment 2: actor gradient accumulation (--actor-accumulate). K minibatches with the actor loss
divided by K accumulate to the gradient of their union, and ppo_pass steps the actor once per group."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from swccg_policy.ppo_step import PPOConfig, backward
from swccg_policy.v4_rl import ppo_pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
NEEDS_DATA = pytest.mark.skipif(not DATA.is_dir() or not CHECKPOINT.exists(),
                                reason='needs runs/dojo/collect-test and the focus-s2 checkpoint')
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def _setup():
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_trajectory import TrajectoryReader
    torch.backends.cuda.matmul.allow_tf32 = False  # as the learner (v4_rl.learn)
    torch.backends.cudnn.allow_tf32 = False
    actor, critic, tables, _ = load(CHECKPOINT, 'cuda')
    actor.packed_attention = False
    games = TrajectoryReader(tables, []).load(sorted(trajectory_files(DATA))[0])
    items = [item for game in games[:2] for item in game['items']][:16]
    critic_items = [item for game in games[:2] for item in game['critic_items']][:16]
    rng = np.random.default_rng(20261019)
    common = dict(sequences=[[0]] * 16, exclusions=[[]] * 16, old_logp=np.full(16, -0.7, np.float32),
                  advantages=rng.normal(size=16).astype(np.float32), returns=np.zeros(16, np.float32),
                  stack=stack, config=PPOConfig(bf16=False))
    return actor, critic, items, critic_items, common


@NEEDS_DATA
@NEEDS_GPU
def test_actor_accumulation_matches_the_union():
    actor, critic, items, critic_items, common = _setup()
    whole = copy.deepcopy(actor)
    backward(whole, critic, items, critic_items, np.arange(16), **common)
    parts = copy.deepcopy(actor)
    for quarter in np.split(np.arange(16), 4):
        backward(parts, critic, items, critic_items, quarter, actor_scale=0.25, **common)
    checked = 0
    for (name, a), (_, b) in zip(whole.named_parameters(), parts.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=1e-6, msg=name)
            checked += 1
    assert checked > 0


class _Counting(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


@NEEDS_DATA
@NEEDS_GPU
@pytest.mark.parametrize('accumulate, expected', [(1, 4), (2, 2), (4, 1), (3, 2)])
def test_ppo_pass_steps_the_actor_once_per_group(accumulate, expected):
    import collections
    actor, critic, items, critic_items, common = _setup()
    optimizers = [_Counting(actor.parameters()), _Counting(critic.parameters())]
    statistics, counts = collections.defaultdict(list), collections.Counter()
    ppo_pass(actor, critic, optimizers, items, critic_items, np.split(np.arange(16), 4), common['sequences'],
             common['exclusions'], common['old_logp'], common['advantages'], common['returns'], None,
             common['config'], True, 1, statistics, counts, actor_accumulate=accumulate)
    assert optimizers[0].steps == expected
    assert optimizers[1].steps == 4
