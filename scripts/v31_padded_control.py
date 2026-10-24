"""Background grid helpers of the padded probes (pile defaults, grid rows and their decoding). Plan remove-old-generations
removed the frozen-V3 control converter that used them."""

from functools import lru_cache

import numpy as np



MASKS = (3, 5, 6, 7)


def pile_defaults(item):
    result = {}
    for token in item["tok"]:
        if token[0] == 3:
            for scope in range(2):
                if token[7] & (2 << scope):
                    result[int(token[3]), int(token[2]) - 32, scope] = (int(token[6]), int(token[7] & 1))
    return tuple(sorted(result.items()))


def decode_rows(tokens, floats, blueprint, owner, defaults):
    """Independent schema-3 constraint decoder, including complete-pile exclusions."""
    piles = dict(defaults)
    grid = np.zeros((17, 3, 3), np.uint8)
    for quantity in range(13):
        if quantity == 6:
            continue
        for scope in range(2):
            size, complete = piles.get((owner, quantity, scope), (0, 0))
            grid[quantity, scope] = [0, 0 if complete else size, complete]
    selected = [(token, values) for token, values in zip(tokens, floats, strict=True)
                if token[0] in (2, 4) and token[1] == blueprint and token[3] == owner]
    for token, values in selected:
        if token[0] != 2:
            continue
        for scope in range(2):
            if token[7] & (2 << scope):
                grid[token[2] - 32, scope] = [token[6], int(values[1]), 1]
    for quantity, mask in enumerate(MASKS, 13):
        for scope in range(2):
            grid[quantity, scope, :2] = sum((grid[pile, scope, :2].astype(int)
                                            for pile in range(3) if mask & (1 << pile)), np.zeros(2, int))
    for token, values in selected:
        if token[0] != 4:
            continue
        for scope in range(2):
            if token[7] & (2 << scope):
                grid[13 + MASKS.index(int(token[4]) - 1), scope] = [token[6], int(values[1]), 1]
    return grid


@lru_cache(maxsize=2048)
def grid_rows(blueprint, owner, content, defaults, synthetic):
    grid = np.frombuffer(content, np.uint8).reshape(17, 3, 3)
    default_grid = decode_rows([], [], blueprint, owner, defaults)
    rows = []
    values = []
    for quantity in range(17):
        groups = {}
        for scope in range(2):
            lo, hi, asserted = map(int, grid[quantity, scope])
            if not asserted:
                continue
            if quantity < 13 and np.array_equal(grid[quantity, scope], default_grid[quantity, scope]):
                continue
            groups.setdefault((lo, hi), 0)
            groups[lo, hi] |= 2 << scope
        for (lo, hi), flags in groups.items():
            rows.append([2 if quantity < 13 else 4, blueprint, 32 + quantity if quantity < 13 else 63,
                         owner, 0 if quantity < 13 else MASKS[quantity - 13] + 1, 0, lo,
                         flags | (int(lo == hi) if quantity < 13 else 0)])
            values.append([0, hi, 0, 0])
    if not synthetic:
        reconstructed = decode_rows(rows, values, blueprint, owner, defaults)
        if not np.array_equal(reconstructed, grid):
            wrong = np.argwhere(reconstructed != grid).tolist()
            return None, f"Unrepresentable background grid {blueprint}/{owner}: {wrong[:8]}"
    return np.asarray(rows, np.int32).reshape(-1, 8), np.asarray(values, np.float32).reshape(-1, 4)
