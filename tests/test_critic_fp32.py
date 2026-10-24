"""Plan critic-fp32 A-1: the --critic-fp32 switch changes only the critic path."""
from pathlib import Path

import numpy as np
import pytest
import torch

from swccg_policy.ppo_step import PPOConfig, critic_autocast
from swccg_policy import v4_rl

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
DATA = ROOT / 'runs/dojo/collect-test/trajectories'


def test_default_critic_follows_bf16():
    assert critic_autocast(PPOConfig()) is True
    assert critic_autocast(PPOConfig(bf16=False)) is False
    assert critic_autocast(PPOConfig(bf16=True, critic_bf16=False)) is False
    assert PPOConfig(bf16=True, critic_bf16=False).bf16 is True  # the actor keeps bf16


def test_learner_flag_parses():
    import sys
    argv = sys.argv
    try:
        sys.argv = ['v4_rl', '--run', '/tmp/x', '--initial', '/tmp/y.pt', '--critic-fp32']
        parser_args = None

        def capture(args):
            nonlocal parser_args
            parser_args = args
        original = v4_rl.learn
        v4_rl.learn = capture
        try:
            v4_rl.main()
        finally:
            v4_rl.learn = original
    finally:
        sys.argv = argv
    assert parser_args.critic_fp32 is True and parser_args.bf16 is True


@pytest.mark.skipif(not torch.cuda.is_available() or not CHECKPOINT.exists() or not DATA.is_dir(),
                    reason='needs a GPU, the focus-s2 checkpoint and runs/dojo/collect-test')
def test_fp32_values_match_plain_fp32():
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_trajectory import TrajectoryReader
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, critic, tables, _ = load(CHECKPOINT, 'cuda')
    critic.encoder.clear_tables()
    games = TrajectoryReader(tables, []).load(trajectory_files(DATA)[0])
    items = [item for game in games for item in game['critic_items']][:64]
    learner = v4_rl.predict_values(critic, items, PPOConfig(bf16=True, critic_bf16=False), 32)
    with torch.no_grad():
        plain = critic(stack(items, 'cuda')).float().cpu().numpy()
    # fp32 rounding only (cached card table, batching): measured 1.4e-6 on 2026-10-07.
    assert np.abs(learner - plain).max() < 1e-5
