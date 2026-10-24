import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_efficient_model import fixture
from swccg_policy.efficient import EfficientPolicy, EfficientCritic
from swccg_policy.ppo_step import PPOConfig, backward, microbatches, policy_loss, sequence_tensors
from swccg_policy.v4_data import parse, stack


def fixed_batch():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    items = []
    critics = []
    for index in range(7):
        row = copy.deepcopy(record)
        row[7][0] = index
        row[8].extend([copy.deepcopy(row[8][2]) for _ in range(index)])
        row[9][1][3] = "1_2"
        items.append(parse(row, tables))
        privileged = [token for token in row[8] if token[0] != 10]
        privileged += [copy.deepcopy(row[8][2]) for _ in range(index + 5)]
        row.extend([privileged, [], row[11]])
        critics.append(parse(row, tables, critic=True))
    return actor, critic, items, critics


@pytest.mark.parametrize("bf16", [False, True])
def test_separate_packing_matches_joint_loss_gradients(bf16):
    actor, critic, items, critics = fixed_batch()
    reference_actor = copy.deepcopy(actor)
    reference_critic = copy.deepcopy(critic)
    indices = np.arange(len(items))
    sequences = [[index % 2] for index in indices]
    exclusions = [None] * len(items)
    old = np.full(len(items), -0.7, np.float32)
    advantages = np.linspace(-1, 1, len(items), dtype=np.float32)
    returns = np.linspace(0.5, -0.5, len(items), dtype=np.float32)
    config = PPOConfig(actor_tokens=35, critic_tokens=55, bf16=bf16)
    batch = stack(items)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        logits = reference_actor(batch)
        values = reference_critic(stack(critics))
    picks, lengths, excluded = sequence_tensors(indices, sequences, exclusions, batch["omask"])
    loss, _ = policy_loss(logits, batch, picks, lengths, excluded,
                          torch.from_numpy(old), torch.from_numpy(advantages), config)
    (loss + config.value_coef * F.mse_loss(values.float(), torch.from_numpy(returns))).backward()
    _, counts = backward(actor, critic, items, critics, indices, sequences, exclusions,
                         old, advantages, returns, stack, config)
    assert counts["actor_microbatches"] > 1
    assert counts["critic_microbatches"] > 1
    for expected, actual in ((reference_actor, actor), (reference_critic, critic)):
        left = []
        right = []
        for original, packed in zip(expected.parameters(), actual.parameters(), strict=True):
            assert (original.grad is None) == (packed.grad is None)
            if original.grad is not None:
                left.append(original.grad.flatten())
                right.append(packed.grad.flatten())
        left = torch.cat(left)
        right = torch.cat(right)
        if bf16:
            # BF16 cancellation can dominate the relative error of near-zero gradients.
            # Bound each component with an absolute floor and 3% relative tolerance.
            torch.testing.assert_close(left, right, atol=5e-3, rtol=0.03)
        else:
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)


def test_microbatch_keeps_oversized_and_last_rows():
    items = [{"tok": np.zeros((length, 8))} for length in (2, 3, 100, 4, 1)]
    parts = list(microbatches(np.arange(5), items, 8))
    assert [part.tolist() for part in parts] == [[0, 1], [2], [3, 4]]
