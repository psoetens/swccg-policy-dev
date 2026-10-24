"""Independent actor/critic token packing with the existing PPO loss and update boundaries."""

import contextlib
from dataclasses import dataclass
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .rl import seq_logprob


@dataclass(frozen=True)
class PPOConfig:
    actor_tokens: int = 32768
    critic_tokens: int = 32768
    clip: float = 0.2
    entropy: float = 0.003
    value_coef: float = 1.0
    actor_clip: float = 0.5
    critic_clip: float = 1.0
    bf16: bool = True
    # Plan critic-fp32: None = the critic follows bf16; False = the critic runs in fp32 (actor unchanged).
    critic_bf16: bool | None = None
    # Plan critic-v2: weight of the win-probability head's cross-entropy in the critic loss (0 = off).
    win_coef: float = 0.0
    # Plan learner-speed S-1: TF32 matmuls / convolutions for the critic only (its forward, backward and value passes).
    critic_tf32: bool = False
    # Plan learner-speed S-3: the critic's forward / backward on its own CUDA stream, overlapping the actor's.
    critic_stream: bool = False


def critic_autocast(config):
    return config.bf16 if config.critic_bf16 is None else config.critic_bf16


@contextlib.contextmanager
def critic_precision(config):
    """Plan learner-speed S-1: TF32 on while the critic runs, restored afterwards (the actor keeps the process setting)."""
    if not config.critic_tf32:
        yield
        return
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def microbatches(indices, items, budget):
    """Keep every row, including an individually oversized observation."""
    if budget < 1:
        raise ValueError("Positive token budget required")
    start = 0
    while start < len(indices):
        end = start
        longest = 1
        while end < len(indices):
            candidate = max(longest, len(items[indices[end]]["tok"]))
            if end > start and candidate * (end - start + 1) > budget:
                break
            longest = candidate
            end += 1
        yield indices[start:end]
        start = end


def sequence_tensors(indices, sequences, exclusions, option_mask):
    width = max(1, max(len(sequences[index]) for index in indices))
    picks = np.zeros((len(indices), width), np.int64)
    lengths = np.zeros(len(indices), np.int64)
    excluded = np.zeros(tuple(option_mask.shape), bool)
    for row, index in enumerate(indices):
        sequence = sequences[index]
        picks[row, :len(sequence)] = sequence
        lengths[row] = len(sequence)
        excluded[row, exclusions[index] or []] = True
    device = option_mask.device
    return tuple(torch.from_numpy(array).to(device) for array in (picks, lengths, excluded))


def policy_loss(logits, batch, sequences, lengths, excluded, old_logp, advantages, config):
    logits = logits.float()
    new_logp = seq_logprob(logits, batch["omask"], batch["opt"][:, :, 5] > 0,
                          sequences, lengths, excluded, sync_free=True)
    ratio = torch.exp(new_logp - old_logp)
    policy = -torch.minimum(ratio * advantages,
                            ratio.clamp(1 - config.clip, 1 + config.clip) * advantages).mean()
    log_probs = torch.log_softmax(logits, dim=1)
    probabilities = log_probs.exp().nan_to_num(0.0)
    entropy = -(probabilities * log_probs.nan_to_num(0.0)).sum(1).mean()
    stats = {"pg": policy, "entropy": entropy,
             "clipfrac": ((ratio - 1).abs() > config.clip).float().mean(),
             "approx_kl_old": (old_logp - new_logp).mean(),
             "approx_kl_nonnegative": (ratio - 1 - (new_logp - old_logp)).mean(),
             "max_abs_logit": logits.masked_fill(~batch["omask"], 0).abs().max()}
    # Plan actor-v6-inputs (BL-01): per decision, the largest |logit| over its legal options, its question template
    # and the option template behind it; concatenated (not averaged) per update.
    if "qtmpl" in batch:
        row_abs = logits.detach().masked_fill(~batch["omask"], 0).abs()
        row_max, row_option = row_abs.max(1)
        stats.update({"rows_logit": row_max, "rows_qtmpl": batch["qtmpl"].reshape(-1).long(),
                      "rows_otmpl": batch["opt"][:, :, 0].long().gather(1, row_option[:, None]).squeeze(1)})
    return policy - config.entropy * entropy, stats


def win_loss(logits, labels):
    """Mean binary cross-entropy over the rows with a known result (labels 1 / 0; NaN = undecided, masked)."""
    known = ~torch.isnan(labels)
    losses = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(labels), reduction="none")
    return (losses * known).sum() / known.sum().clamp_min(1)


def backward(actor, critic, items, critic_items, indices, sequences, exclusions,
             old_logp, advantages, returns, stack, config, policy_on=True, wins=None, critic_scale=1.0,
             actor_scale=1.0):
    """Accumulate one logical minibatch; callers clip/step once after BOTH models finish.

    Each micro-loss is weighted by its row count / logical row count. Splitting the
    two disjoint parameter graphs therefore preserves the original joint mean loss.
    Plan critic-v2: wins (per-row results, NaN = undecided) feed the win head when config.win_coef > 0, and
    critic_scale divides the critic loss when several minibatches accumulate into one critic step; actor_scale does the
    same for the actor (plan critic-noise Amendment 2).
    Plan learner-speed S-3: with config.critic_stream the critic's pass runs on its own CUDA stream (and thread),
    overlapping the actor's; the two graphs and optimizers are disjoint, so only the ordering changes.
    """
    arguments = (actor, critic, items, critic_items, indices, sequences, exclusions, old_logp, advantages, returns,
                 stack, config, policy_on, wins, critic_scale, actor_scale)
    device = next(actor.parameters()).device
    if not (config.critic_stream and device.type == "cuda"):
        results = [role_pass(role, *arguments) for role in ("actor", "critic")]
    else:
        stream = _critic_stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        outcome = {}

        def critic_work():
            try:
                with torch.cuda.stream(stream):
                    outcome["result"] = role_pass("critic", *arguments)
            except BaseException as error:  # noqa: BLE001 - re-raised in the calling thread
                outcome["error"] = error

        worker = threading.Thread(target=critic_work, name="critic-stream")
        worker.start()
        try:
            actor_result = role_pass("actor", *arguments)
        finally:
            worker.join()
            torch.cuda.current_stream(device).wait_stream(stream)
        if "error" in outcome:
            raise outcome["error"]
        results = [actor_result, outcome["result"]]
    statistics = {**results[0][0], **results[1][0]}
    return statistics, {**results[0][1], **results[1][1]}


_STREAMS = {}


def _critic_stream(device):
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device)
    return _STREAMS[device]


def role_pass(role, actor, critic, items, critic_items, indices, sequences, exclusions, old_logp, advantages, returns,
              stack, config, policy_on, wins, critic_scale, actor_scale=1.0):
    """The forward and backward of one model ("actor" or "critic") over one logical minibatch."""
    use_win = wins is not None and config.win_coef > 0
    device = next(actor.parameters()).device
    statistics = {}
    count = 0
    observations, budget = (items, config.actor_tokens) if role == "actor" else (critic_items, config.critic_tokens)
    for selected in microbatches(indices, observations, budget):
        count += 1
        batch = stack([observations[index] for index in selected], device)
        weight = len(selected) / len(indices)
        with torch.set_grad_enabled(role == "critic" or policy_on), \
                (critic_precision(config) if role == "critic" else contextlib.nullcontext()):
            autocast = config.bf16 if role == "actor" else critic_autocast(config)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                if role == "actor":
                    prediction = actor(batch)
                elif use_win:
                    prediction, win_logits = critic.heads(batch)
                else:
                    prediction = critic(batch)
            if role == "actor":
                picks, lengths, excluded = sequence_tensors(selected, sequences, exclusions, batch["omask"])
                old = torch.as_tensor(old_logp[selected], device=device)
                advantage = torch.as_tensor(advantages[selected], device=device)
                loss, stats = policy_loss(prediction, batch, picks, lengths, excluded, old, advantage, config)
                raw = getattr(actor, "raw_max_abs_logit", None)
                if raw is not None:
                    stats["raw_max_abs_logit"] = raw
                if actor_scale != 1.0:
                    loss = loss * actor_scale
            else:
                targets = torch.as_tensor(returns[selected], device=device)
                value_loss = F.mse_loss(prediction.float(), targets)
                loss = config.value_coef * value_loss
                stats = {"vloss": value_loss}
                if use_win:
                    result = win_loss(win_logits, torch.as_tensor(wins[selected], device=device))
                    loss = loss + config.win_coef * result
                    stats["win_loss"] = result
                if critic_scale != 1.0:
                    loss = loss * critic_scale
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; update not published")
            if role == "critic" or policy_on:
                (loss * weight).backward()
        for name, value in stats.items():
            value = value.detach()
            if name.startswith("rows_"):
                statistics[name] = torch.cat((statistics[name], value)) if name in statistics else value
            elif name in ("max_abs_logit", "raw_max_abs_logit"):
                statistics[name] = torch.maximum(statistics.get(name, value), value)
            else:
                statistics[name] = statistics.get(name, 0.0) + value * weight
    return statistics, {role + "_microbatches": count}


def step(actor, critic, optimizers, *args, config, policy_on=True, critic_zero=True, critic_step=True,
         actor_zero=True, actor_step=True, **kwargs):
    """Plan critic-v2: with critic_zero / critic_step False the critic's gradients accumulate across calls; plan
    critic-noise Amendment 2: actor_zero / actor_step the same for the actor."""
    if actor_zero:
        optimizers[0].zero_grad(set_to_none=True)
    if critic_zero:
        optimizers[1].zero_grad(set_to_none=True)
    statistics, counts = backward(actor, critic, *args, config=config, policy_on=policy_on, **kwargs)
    if critic_step:
        statistics["critic_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            critic.parameters(), config.critic_clip, error_if_nonfinite=True)
        optimizers[1].step()
    if policy_on and actor_step:
        statistics["actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), config.actor_clip, error_if_nonfinite=True)
        optimizers[0].step()
    return statistics, counts
