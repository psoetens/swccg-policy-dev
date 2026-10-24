"""The learner's log-probability of a sampled pick sequence must equal the sampler's, also with excluded options (the preselected
cards of a step-by-step selection, PolicyAi envelope field "x")."""
import random

import numpy as np
import torch

from swccg_policy.rl import sample_sequence, seq_logprob


def check(n: int, k: int, excluded: list[int], none: list[int], seed: int) -> None:
    rng = random.Random(seed)
    logits = np.random.default_rng(seed).normal(size=n).astype(np.float32) * 3
    is_none = np.zeros(n, dtype=bool)
    is_none[none] = True
    seq, logp = sample_sequence(logits, is_none, k, 1.0, 0.0, rng, excluded)
    assert seq and not set(seq) & set(excluded)
    ex = torch.zeros((1, n), dtype=torch.bool)
    ex[0, excluded] = True
    lp = seq_logprob(torch.from_numpy(logits)[None], torch.ones((1, n), dtype=torch.bool), torch.from_numpy(is_none)[None],
                     torch.tensor([seq]), torch.tensor([len(seq)]), ex if excluded else None)
    assert abs(float(lp[0]) - logp) < 1e-4, (float(lp[0]), logp)


def test_excluded_matches():
    for seed in range(200):
        check(8, 1, [0, 1], [], seed)       # add one card to a 2-card preselection
        check(10, 2, [3], [], seed)          # add two cards
        check(6, 1, [], [5], seed)           # no exclusion (unchanged behaviour)


if __name__ == "__main__":
    test_excluded_matches()
    print("ok")
