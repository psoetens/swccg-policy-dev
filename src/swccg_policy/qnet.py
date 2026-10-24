"""Plan q-critic: an option-value network Q(s, o) on the actor's view of a decision.

The Q-net is the actor's architecture read as values: the actor scores every option row with `score`; here that score is
Q(s, o) (no logit cap). It starts from an actor's weights with the final score layer zeroed (Q = 0 everywhere, the
option features as the actor learned them) and is trained on the chosen option's value target (the λ-return the critic
is trained on). The actor's advantage of the chosen option is then A_Q = Q(s, o) - sum_o' pi(o') Q(s, o'): the expectation
over what happened after the decision is learned by Q, so the sampled outcome (a drawn card, a destiny) is not in A_Q.
The Q-net sees what the actor sees (no privileged rows), so its expectation is the one the actor can act on.
"""
import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .v4_checkpoint import load


def from_actor(actor):
    """A Q-net from an actor: a deep copy, no logit cap, the last score layer zeroed (Q = 0 at the start)."""
    qnet = copy.deepcopy(actor)
    qnet.cfg = dataclasses.replace(actor.cfg, logit_cap=0.0)
    with torch.no_grad():
        qnet.score[-1].weight.zero_()
        qnet.score[-1].bias.zero_()
    return qnet


def save_qnet(path, qnet, table_sha, metadata=None):
    """The Q-net's weights and config next to the checkpoint whose tables it uses (table_sha)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'cfg': qnet.cfg.to_dict(), 'state': {key: value.detach().cpu() for key, value in qnet.state_dict().items()},
                'table_sha': table_sha, 'qnet_meta': metadata or {}}, temporary)
    temporary.replace(path)


def load_qnet(path, base, device='cpu'):
    """A saved Q-net, built on the tables of `base` (a checkpoint path, or an (actor, critic, tables, checkpoint) tuple)."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    actor, _, tables, _ = load(base, device) if isinstance(base, (str, Path)) else base
    if saved['table_sha'] != tables.sha:
        raise ValueError('Q-net and base checkpoint use different tables')
    qnet = from_actor(actor)
    qnet.load_state_dict(saved['state'])
    return qnet.to(device), saved.get('qnet_meta', {})


def chosen_option(seq):
    """The option a decision's sampled answer started with (the first pick of a sequence)."""
    return int(seq[0]) if len(seq) else 0


def q_loss(qnet, batch, chosen, targets):
    """MSE between Q(s, chosen) and the targets, and the Q values of the batch (for statistics)."""
    values = qnet(batch).float()
    picked = values.gather(1, chosen.view(-1, 1)).squeeze(1)
    return F.mse_loss(picked, targets), values


def q_advantages(values, valid, pi, chosen):
    """A_Q = Q(s, chosen) - sum_o pi(o) Q(s, o) per decision (values, valid, pi: [batch, options])."""
    q = values.masked_fill(~valid, 0.0)
    expected = (pi * q).sum(-1)
    return q.gather(1, chosen.view(-1, 1)).squeeze(1) - expected


def option_values(model, items, batches, stack, device='cuda'):
    """Per item, the model's output over its options (numpy), no gradients: Q for a Q-net, logits for an actor."""
    out = [None] * len(items)
    model.eval()
    with torch.no_grad():
        for selected in batches(items):
            batch = stack([items[index] for index in selected], device)
            values = model(batch).float().cpu().numpy()
            for row, index in enumerate(selected):
                out[index] = values[row, :len(items[index]['opt'])]
    return out


def policy_from_logits(logits):
    """pi at T = 1 over one decision's options (numpy)."""
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()
