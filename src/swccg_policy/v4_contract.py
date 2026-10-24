"""Vectorized ABI-4 validation; vocabulary and table hashes are validated once by the loader."""

import numpy as np

from .soft_vocabulary import SoftVocabulary


def validate(item, vocabulary, n_cards, actor=False):
    if not isinstance(vocabulary, SoftVocabulary):
        raise TypeError("An already-validated SoftVocabulary is required")
    if item.get("schema_version") != 4:
        raise ValueError("Schema 4 required")
    if actor and item.get("privileged", False):
        raise ValueError("Privileged payload rejected by actor path")
    tokens, floats, options, head, grid = (item[key] for key in ("tok", "tokf", "opt", "head", "kgrid"))
    size = len(tokens)
    if (tokens.shape != (size, 8) or floats.shape not in ((size, 4), (size, 8)) or options.shape != (len(options), 6)
            or head.shape != (22,) or grid.shape != (len(grid), 153)):
        raise ValueError("ABI-4 shape mismatch")
    if (tokens.dtype != np.int32 or options.dtype != np.int32 or head.dtype != np.int32
            or floats.dtype != np.float32 or grid.dtype != np.uint8):
        raise ValueError("ABI-4 dtype mismatch")
    if not np.isfinite(floats).all() or not (0 <= head[0] < 7 and head[2] == 0 and head[3] in (0, 1)):
        raise ValueError("Invalid head or nonfinite floats")
    if not 0 <= head[1] < len(vocabulary.payload["q"]):
        raise ValueError("Question ID outside vocabulary")
    if np.any((options[:, 0] < 0) | (options[:, 0] >= len(vocabulary.payload["o"]))) or np.any((options[:, 1] < 0) | (options[:, 1] > 1)):
        # Plan multiselect-count stage B: column 1 (formerly reserved 0) = the option's card is already selected.
        raise ValueError("Invalid option template/reserved field")
    if np.any((tokens[:, 1] < 0) | (tokens[:, 1] >= n_cards)) or np.any(
            (options[:, 3] < 0) | (options[:, 3] >= n_cards)):
        raise ValueError("Blueprint outside card table")
    kind, _, zone, owner, slot, parent, count, flags = tokens.T
    if np.any((kind < 0) | (kind > 10) | (kind == 4) | (kind == 6)
              | (zone < 0) | (zone >= 64) | (owner < 0) | (owner > 1)):
        raise ValueError("Invalid kind, zone or owner")
    edges = kind == 8
    if np.any((parent[edges] < 1) | (parent[edges] > size) | (floats[edges, 0] < 1)
              | (floats[edges, 0] > size) | (floats[edges, 0] % 1 != 0)
              | (slot[edges] < 1) | (slot[edges] > 14)):
        raise ValueError("Invalid physical relationship")
    context = kind == 7
    if actor and np.isin(slot[context], vocabulary.privileged_context_ids).any():
        raise ValueError("Privileged context rejected by actor path")
    if np.any((slot[context] < 0) | (slot[context] >= len(vocabulary.payload["context"]))
              | (parent[context] < 0) | (parent[context] > size) | (flags[context] < 0) | (flags[context] > 127)):
        raise ValueError("Invalid context")
    indices = np.column_stack((floats[context, 0], floats[context, 2:4], count[context]))
    if np.any((indices < -1) | (indices % 1 != 0)) or np.any((indices[:, :-1] == -1) & (indices[:, 1:] != -1)):
        raise ValueError("Invalid ordered context index tuple")
    references = (kind == 9) | (kind == 10)
    option_references = kind == 10
    if np.any(slot[references] < 1) or np.any(floats[kind == 9, 0] != 0) or np.any(
            (floats[option_references, 0] < 1) | (floats[option_references, 0] > len(options))
            | (floats[option_references, 0] % 1 != 0)):
        raise ValueError("Invalid reference ordinal or option routing")
    if np.any((options[:, 2] < 0) | (options[:, 2] > size)):
        raise ValueError("Invalid option source")
    retained_facts = (kind == 3) | (kind == 5)
    if np.any((flags[retained_facts] < 2) | (flags[retained_facts] > 15)):
        raise ValueError("Invalid retained fact scope")
    if actor and np.any((flags[retained_facts] // 2) & 4):
        raise ValueError("Privileged actor fact")
    known = kind == 2
    if not np.array_equal(np.sort(slot[known]), np.arange(1, len(grid) + 1)):
        raise ValueError("Grid rows must be bound exactly once")
    if len(set(map(tuple, tokens[known][:, [1, 3]]))) != len(grid):
        raise ValueError("Duplicate blueprint/owner grid token")
    if np.any(tokens[known][:, [2, 5, 6, 7]]) or np.any(floats[known]):
        raise ValueError("Reserved grid-token fields must be zero")
    cells = grid.reshape(-1, 17, 3, 3)
    if np.any(cells[..., 0] > cells[..., 1]) or np.any(cells[..., 2] > 1):
        raise ValueError("Invalid grid bounds/assertion")
    if np.any(cells[:, 6]) or actor and np.any(cells[:, :, 2]):
        raise ValueError("Skipped TABLE cells or opponent-private actor cells are nonzero")
