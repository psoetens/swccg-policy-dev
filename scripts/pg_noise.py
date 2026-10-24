"""Plan critic-noise E1: the gradient noise scale from the Gram matrix of independent minibatch gradients (numpy only).

For minibatch means g = G + noise with per-decision covariance S, E[g_a . g_b] = |G|^2 (a != b) and
E|g|^2 = |G|^2 + tr(S) / B, so the mean pairwise cosine is c ~ 1 / (1 + B_noise / B) with B_noise = tr(S) / |G|^2,
i.e. B_noise = B (1 - c) / c. The interval resamples whole minibatches (bootstrap over minibatch pairs).
"""
import numpy as np

E1_THRESHOLD = 12000  # decisions: the lower 95% bound of B_noise(GAE) must reach this
E1_CONSISTENCY = 2.0  # B_noise at 1,024 and 4,096 within this factor of each other


def b_noise(c, size):
    """B (1 - c) / c; c <= 0 (no signal resolved) gives infinity."""
    return float(size * (1 - c) / c) if c > 0 else float('inf')


def pair_means(gram, weights):
    """Mean off-diagonal entry of GRAM over the pairs of a weighted (resampled) set of minibatches: rows of WEIGHTS
    count how often each minibatch was drawn; pairs of a minibatch with its own copy are left out."""
    off = gram - np.diag(np.diag(gram))
    numerator = np.einsum('ra,ab,rb->r', weights, off, weights)
    return numerator / (weights.sum(1) ** 2 - (weights ** 2).sum(1))


def noise_scale(gram, size, resamples=2000, seed=990001, level=0.95):
    """Mean pairwise cosine c of the minibatch gradients behind GRAM (k x k dot products), its bootstrap interval over
    minibatches, and B_noise with the interval mapped through B (1 - c) / c (decreasing in c). Also the
    dot-product version c_dot = mean(g_a . g_b) / mean|g|^2 (the textbook ratio of means), for comparison."""
    gram = np.asarray(gram, np.float64)
    k = len(gram)
    norms = np.sqrt(np.clip(np.diag(gram), 1e-300, None))
    cos = gram / np.outer(norms, norms)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(resamples, k))
    weights = np.stack([np.bincount(row, minlength=k) for row in draws]).astype(np.float64)
    weights = weights[(weights > 0).sum(1) >= 2]  # at least one pair of distinct minibatches
    alpha = (1 - level) / 2
    c = float(pair_means(cos, np.ones((1, k)))[0])
    boot = pair_means(cos, weights)
    c_lo, c_hi = (float(x) for x in np.quantile(boot, [alpha, 1 - alpha]))
    mean_square = np.diag(gram).mean()
    c_dot = float(pair_means(gram, np.ones((1, k)))[0] / mean_square)
    boot_dot = pair_means(gram, weights) / ((weights * np.diag(gram)).sum(1) / weights.sum(1))
    dot_lo, dot_hi = (float(x) for x in np.quantile(boot_dot, [alpha, 1 - alpha]))
    return {'minibatch': size, 'minibatches': k, 'c': c, 'c_lo': c_lo, 'c_hi': c_hi,
            'b_noise': b_noise(c, size), 'b_noise_lo': b_noise(c_hi, size), 'b_noise_hi': b_noise(c_lo, size),
            'c_dot': c_dot, 'c_dot_lo': dot_lo, 'c_dot_hi': dot_hi,
            'b_noise_dot': b_noise(c_dot, size), 'b_noise_dot_lo': b_noise(dot_hi, size),
            'b_noise_dot_hi': b_noise(dot_lo, size), 'mean_norm': float(norms.mean())}


def e1_verdict(gae_cells, none_cells=None, small=1024, large=4096):
    """The pre-registered E1 readout from the GAE source's noise_scale cells per minibatch size (and the no-baseline
    source's, for the critic gain). E1 holds iff the lower 95% bound of B_noise(GAE) at the larger size reaches
    E1_THRESHOLD and the point estimates at SMALL and LARGE agree within E1_CONSISTENCY. 'holds_strict' also asks
    the smaller size's lower bound to reach the threshold. Missing sizes: 'unmeasured'."""
    if small not in gae_cells or large not in gae_cells:
        return {'E1': 'unmeasured', 'reason': f'needs minibatch sizes {small} and {large}'}
    a, b = gae_cells[small], gae_cells[large]
    finite = np.isfinite(a['b_noise']) and np.isfinite(b['b_noise'])
    ratio = a['b_noise'] / b['b_noise'] if finite else float('nan')
    consistent = bool(finite and 1 / E1_CONSISTENCY <= ratio <= E1_CONSISTENCY)
    large_enough = bool(b['b_noise_lo'] >= E1_THRESHOLD)
    out = {f'b_noise_gae_{small}': a['b_noise'], f'b_noise_gae_{small}_lo': a['b_noise_lo'],
           f'b_noise_gae_{large}': b['b_noise'], f'b_noise_gae_{large}_lo': b['b_noise_lo'],
           f'ratio_{small}_over_{large}': ratio, 'consistent_within_2x': consistent,
           'lower_bound_at_least_threshold': large_enough, 'threshold': E1_THRESHOLD,
           'E1': 'holds' if large_enough and consistent else 'fails',
           'holds_strict': bool(large_enough and consistent and a['b_noise_lo'] >= E1_THRESHOLD)}
    if not finite:
        out['E1'] = 'holds' if large_enough else 'fails'  # no signal resolved at all: noise-dominated, no ratio
        out['holds_strict'] = bool(large_enough and a['b_noise_lo'] >= E1_THRESHOLD)
    if none_cells:
        out['critic_gain'] = {size: gae_cells[size]['c'] / none_cells[size]['c'] if none_cells[size]['c'] > 0
                              else float('inf') for size in gae_cells if size in none_cells}
    return out
