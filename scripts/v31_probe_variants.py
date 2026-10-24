"""Isolated Amendment 6 controls; production observation contracts stay unchanged.

Plan remove-old-generations: the frozen-V3 control (`frozen_module`, `frozen_model`, `frozen_stack`, `schema3_probe`)
was removed with V3; the per-scope prototype stays.
"""

import numpy as np


def per_scope_probe(item):
    """Split grids by scope, merging identical scope slices without consuming RNG."""
    tokens = []
    floats = []
    grids = []
    mapping = np.zeros(len(item["tok"]) + 1, np.int32)
    for index, (original, values) in enumerate(zip(item["tok"], item["tokf"], strict=True)):
        mapping[index + 1] = len(tokens) + 1
        if original[0] != 2:
            tokens.append(original.copy())
            floats.append(values.copy())
            continue
        grid = item["kgrid"][original[4] - 1].reshape(17, 3, 3)
        groups = {}
        for scope in range(3):
            if grid[:, scope, 2].any():
                groups.setdefault(grid[:, scope].tobytes(), []).append(scope)
        if not groups:
            raise ValueError("Knowledge token has no asserted scope")
        for scopes in groups.values():
            split = np.zeros_like(grid)
            split[:, scopes] = grid[:, scopes]
            grids.append(split.reshape(153))
            token = original.copy()
            token[4] = len(grids)
            token[7] = sum(1 << scope for scope in scopes)
            tokens.append(token)
            floats.append(values.copy())
    tokens = np.asarray(tokens, np.int32)
    floats = np.asarray(floats, np.float32)
    tokens[:, 5] = mapping[tokens[:, 5]]
    edges = tokens[:, 0] == 8
    floats[edges, 0] = mapping[floats[edges, 0].astype(int)]
    options = item["opt"].copy()
    options[:, 2] = mapping[options[:, 2]]
    result = {**item, "tok": tokens, "tokf": floats, "opt": options,
              "kgrid": np.asarray(grids, np.uint8).reshape(-1, 153)}
    validate_per_scope(result, actor=not item.get("privileged", False))
    return result


def validate_per_scope(item, actor=True):
    """Prototype-only scope contract; ordinary schema-4 validation runs before splitting."""
    seen = {}
    bindings = []
    for token in item["tok"]:
        if token[0] != 2:
            continue
        mask = int(token[7])
        if not 1 <= mask <= 7 or (actor and mask & 4):
            raise ValueError("Invalid or actor-private knowledge scope mask")
        key = tuple(token[[1, 3]])
        if seen.get(key, 0) & mask:
            raise ValueError("Overlapping knowledge scope groups")
        seen[key] = seen.get(key, 0) | mask
        bindings.append(int(token[4]))
        grid = item["kgrid"][token[4] - 1].reshape(17, 3, 3)
        selected = []
        for scope in range(3):
            if mask & (1 << scope):
                if not grid[:, scope, 2].any():
                    raise ValueError("Selected scope lacks asserted cells")
                selected.append(grid[:, scope])
            elif grid[:, scope].any():
                raise ValueError("Unselected scope must be entirely zero")
        if any(not np.array_equal(selected[0], other) for other in selected[1:]):
            raise ValueError("Merged scopes must have identical grids")
    if sorted(bindings) != list(range(1, len(item["kgrid"]) + 1)):
        raise ValueError("Each grid must have exactly one token binding")
