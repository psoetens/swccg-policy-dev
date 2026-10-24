"""Plan turn-credit: the actor's advantage per decision, either per-decision GAE (today's) or anchored at the learner
seat's own-turn starts.

Turn credit: the decisions of one seat in a game are split into rounds by their anchors, the first decision of each of
the seat's own turns (`head[7]` my-turn flag = 1 and `head[4]` the seat's turn number above the previous anchor's).
Within a round nothing is discounted: decision t's first-round term is the rewards from t to the next anchor + V(next
anchor) - V(t), so the values between t and the anchor cancel as targets (each decision keeps its own V(t) as the
baseline). Across rounds the round deltas D_k = rewards in round k + V(anchor k+1) - V(anchor k) (V = 0 after the last
decision) are added with weight round_lambda^j (round_lambda = 1 for `rounds` n: the first n rounds, undiscounted).
gamma is 1 (the recipe's); the game result and the Dojo rewards are the rewards, as in v4_rl.advantages.
"""
import numpy as np

HEAD_TURN, HEAD_MY_TURN = 4, 7


def step_rewards(game, count):
    """The per-decision rewards of one game: its result on the last decision, the Dojo entries on theirs."""
    rewards = np.zeros(count)
    rewards[-1] = game['reward']
    for entry in (*game.get('dojo_rewards', ()), *game.get('step_rewards', ()), *game.get('milestone_rewards', ())):
        rewards[int(entry[0])] += entry[1]
    return rewards


def turn_anchors(heads):
    """Indices of the decisions that start one of the seat's own turns (heads: the decisions' `head` rows in order)."""
    anchors, last = [], None
    for index, head in enumerate(heads):
        if head[HEAD_MY_TURN] == 1 and (last is None or head[HEAD_TURN] > last):
            anchors.append(index)
            last = head[HEAD_TURN]
    return anchors


def gae(values, rewards, lam, gamma=1.0):
    """Per-decision GAE(lam), the same recursion as v4_rl.advantages (unnormalised)."""
    count = len(values)
    advantage = np.zeros(count)
    last = 0.0
    for index in range(count - 1, -1, -1):
        following = values[index + 1] if index + 1 < count else 0.0
        delta = rewards[index] + gamma * following - values[index]
        last = delta + gamma * lam * last
        advantage[index] = last
    return advantage


def turn_credit(values, rewards, anchors, round_lambda=None, rounds=None):
    """Turn-anchored advantages (unnormalised) of one seat's decisions in one game. Exactly one of round_lambda
    (geometric weights over rounds; 0 = up to the next anchor only) and rounds (the first n rounds, undiscounted)."""
    if (round_lambda is None) == (rounds is None):
        raise ValueError('give exactly one of round_lambda and rounds')
    if rounds is not None and rounds < 1:
        raise ValueError('rounds must be >= 1')
    count = len(values)
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    tail_rewards = np.concatenate([np.cumsum(rewards[::-1])[::-1], [0.0]])  # rewards from index to the end
    bounds = list(anchors) + [count]
    # D_k: round k from anchor k to anchor k+1 (the last round to the end, where V = 0).
    deltas = np.array([tail_rewards[a] - tail_rewards[b] + (values[b] if b < count else 0.0) - values[a]
                       for a, b in zip(bounds[:-1], bounds[1:])])
    if round_lambda is not None:
        tails = np.zeros(len(deltas) + 1)  # tails[k] = sum_j>=0 round_lambda^j D_{k+j}
        for k in range(len(deltas) - 1, -1, -1):
            tails[k] = deltas[k] + round_lambda * tails[k + 1]
    else:
        sums = np.concatenate([np.cumsum(deltas[::-1])[::-1], [0.0]])  # sums[k] = D_k + ... + D_last
    advantage = np.zeros(count)
    following = 0  # index into anchors of the first anchor after decision t
    for t in range(count):
        while following < len(anchors) and anchors[following] <= t:
            following += 1
        if following == len(anchors):  # no further own turn: the rest of the game, undiscounted
            advantage[t] = tail_rewards[t] - values[t]
            continue
        b = anchors[following]
        first = tail_rewards[t] - tail_rewards[b] + values[b] - values[t]
        if round_lambda is not None:
            advantage[t] = first + round_lambda * tails[following]
        else:
            last = min(following + rounds - 1, len(deltas))
            advantage[t] = first + sums[following] - sums[last]
    return advantage
