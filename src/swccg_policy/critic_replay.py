"""Plan critic-replay: a critic-only replay cache over the last W updates.

Each update stores, per accepted game, the critic item of its first decision and of every k-th decision, with the Monte
Carlo return (lambda = 1, Dojo rewards included) and the win label (1 won, 0 lost, NaN undecided). After the PPO pass
the critic takes extra steps on minibatches sampled uniformly from the cache, so it sees the matchups of ~W x 30 games
per update instead of the latest ~30. The actor never sees replay data; MC targets do not go stale.

Modes: "uniform" samples the whole window every update (each entry is reused ~W x ratio x 256 / window-size times,
which overfit in plan critic-replay); "once" gives every stored entry a random slot in the next W updates and trains
on it exactly once, in that slot (rows per update ~ rows stored per update; no reuse).
"""

import collections

import numpy as np
import torch
import torch.nn.functional as F

from .ppo_step import critic_autocast, critic_precision, microbatches, win_loss


def mc_returns(games):
    """Monte Carlo reward-to-go per step (gamma 1): the terminal reward plus every Dojo / step / milestone reward."""
    out = []
    for game in games:
        rewards = np.zeros(len(game['items']))
        rewards[-1] = game['reward']
        for entry in (*game.get('dojo_rewards', ()), *game.get('step_rewards', ()), *game.get('milestone_rewards', ())):
            rewards[int(entry[0])] += entry[1]
        out.append(np.cumsum(rewards[::-1])[::-1])
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def selected_steps(count, stride):
    """The first decision and every stride-th one after it."""
    return np.arange(0, count, stride)


class CriticReplay:
    def __init__(self, updates, stride, seed=0, mode='uniform'):
        if updates < 1 or stride < 1 or mode not in ('uniform', 'once'):
            raise ValueError("replay needs updates >= 1, stride >= 1 and mode 'uniform' or 'once'")
        self.stride = stride
        self.updates = updates
        self.mode = mode
        self.batches = collections.deque(maxlen=updates)  # one entry per update: (items, returns, labels, first mask)
        self.slots = collections.defaultdict(list)  # mode "once": update index -> [(item, return, label, first)]
        self.update = 0
        self.rng = np.random.default_rng(seed)

    def add(self, games):
        items, returns, labels, first = [], [], [], []
        mc = mc_returns(games)
        position = 0
        for game in games:
            count = len(game['items'])
            label = {1.0: 1.0, -1.0: 0.0}.get(float(game['reward']), np.nan)
            for index in selected_steps(count, self.stride):
                item = game['critic_items'][index]
                # Decoded arrays may be views into the game's raw buffer; copy so the cache doesn't pin whole games.
                items.append({key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in item.items()})
                returns.append(mc[position + index])
                labels.append(label)
                first.append(index == 0)
            position += count
        self.update += 1
        if self.mode == 'once':
            # Slots this update ... this update + W - 1: on average an entry waits W / 2 updates.
            for entry, slot in zip(zip(items, returns, labels, first),
                                   self.update + self.rng.integers(0, self.updates, size=len(items))):
                self.slots[int(slot)].append(entry)
            return
        self.batches.append((items, np.asarray(returns, np.float32), np.asarray(labels, np.float32), np.asarray(first)))

    def __len__(self):
        if self.mode == 'once':
            return sum(len(entries) for entries in self.slots.values())
        return sum(len(batch[0]) for batch in self.batches)

    def due(self):
        """Mode "once": the entries whose slot is this update, removed from the cache."""
        entries = self.slots.pop(self.update, [])
        if not entries:
            return [], np.zeros(0, np.float32), np.zeros(0, np.float32)
        items, returns, labels, _ = zip(*entries)
        return list(items), np.asarray(returns, np.float32), np.asarray(labels, np.float32)

    def flat(self):
        if self.mode == 'once':
            entries = [entry for slot in sorted(self.slots) for entry in self.slots[slot]]
            if not entries:
                return [], np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, bool)
            items, returns, labels, first = zip(*entries)
            return list(items), np.asarray(returns, np.float32), np.asarray(labels, np.float32), np.asarray(first)
        items = [item for batch in self.batches for item in batch[0]]
        returns = np.concatenate([batch[1] for batch in self.batches]) if self.batches else np.zeros(0, np.float32)
        labels = np.concatenate([batch[2] for batch in self.batches]) if self.batches else np.zeros(0, np.float32)
        first = np.concatenate([batch[3] for batch in self.batches]) if self.batches else np.zeros(0, bool)
        return items, returns, labels, first

    def steps(self, new_rows, ratio, size):
        """Replay steps per update: ratio x the update's new rows, in minibatches of `size`."""
        return int(round(ratio * new_rows / size))

    def _backward(self, critic, config, stack, items, returns, labels, indices, use_win, device, totals, tokens):
        """One replay step's forward and backward over its microbatches (plan learner-speed: under critic_precision)."""
        for selected in microbatches(indices, items, tokens or config.critic_tokens):
            batch = stack([items[i] for i in selected], device)
            weight = len(selected) / len(indices)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=critic_autocast(config)):
                if use_win:
                    value, logits = critic.heads(batch)
                else:
                    value, logits = critic(batch), None
            value_loss = F.mse_loss(value.float(), torch.as_tensor(returns[selected], device=device))
            loss = config.value_coef * value_loss
            totals['replay_vloss'] += float(value_loss.detach()) * weight
            if use_win:
                result = win_loss(logits, torch.as_tensor(labels[selected], device=device))
                loss = loss + config.win_coef * result
                totals['replay_win_loss'] += float(result.detach()) * weight
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite replay loss; update not published')
            (loss * weight).backward()

    def train(self, critic, optimizer, config, stack, steps, size, tokens=None):
        """Mode "uniform": `steps` critic steps on uniform samples of `size` rows. Mode "once": the due entries, each
        exactly once, shuffled into steps of `size` rows (`steps` is ignored). Returns mean losses. tokens: the
        microbatch token budget (default config.critic_tokens)."""
        if self.mode == 'once':
            items, returns, labels = self.due()
            order = self.rng.permutation(len(items))
            plan = [order[start:start + size] for start in range(0, len(items), size)]
        else:
            items, returns, labels, _ = self.flat()
            plan = [self.rng.choice(len(items), size=min(size, len(items)), replace=False)
                    for _ in range(max(steps, 0))] if items else []
        if not plan:
            return {}
        steps = len(plan)
        device = next(critic.parameters()).device
        use_win = config.win_coef > 0 and getattr(critic.cfg, 'win_head', False)
        totals = collections.defaultdict(float)
        critic.train()
        for indices in plan:
            indices = indices[np.argsort([len(items[i]['tok']) for i in indices], kind='stable')]
            optimizer.zero_grad(set_to_none=True)
            with critic_precision(config):
                self._backward(critic, config, stack, items, returns, labels, indices, use_win, device, totals, tokens)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.critic_clip, error_if_nonfinite=True)
            optimizer.step()
        result = {name: value / steps for name, value in totals.items()}
        result['replay_rows'] = sum(len(indices) for indices in plan)
        return result
