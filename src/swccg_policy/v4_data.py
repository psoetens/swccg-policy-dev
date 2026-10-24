"""Schema-4 JSON conversion and lossless int32/grid batching."""

import numpy as np
import torch

from .data import DTYPE_INDEX
from .v4_contract import validate


def parse(record, tables, critic=False, width=None):
    if record[0] != 4 or len(record) not in (12, 15):
        raise ValueError("Final schema-4 record required; legacy/pre-grid records are refused")
    if critic and len(record) != 15:
        raise ValueError("Privileged observation missing")
    vocabulary = tables.vocabulary
    tokens = record[12] if critic else record[8]
    raw_grid = record[14] if critic else record[11]
    integers = []
    floats = []
    for token in tokens:
        kind, blueprint, zone, owner, slot, parent, count, flags, *values = token
        if not critic and kind == 7 and slot.startswith(("rules.", "priv.")):
            raise ValueError("Privileged context rejected by actor path")
        slot = vocabulary.lookup("context", slot) if kind == 7 else slot + 1
        integers.append((kind, tables.cards.lookup(blueprint), zone, owner, slot, parent + 1, count, flags))
        floats.append(values)
    options = []
    if not critic:
        # Plan multiselect-count stage B: an optional seventh field 1 = the option's card is already selected (column 1).
        for template, _, source, blueprint, _, value, *selected in record[9]:
            # Gemp uses inPlay for board sources and rules for actions without an attached card.
            if blueprint in ("inPlay", "rules"):
                blueprint_id = 1
            else:
                blueprint_id = tables.cards.lookup(blueprint) if blueprint else 0
            options.append((vocabulary.lookup("o", template), int(selected[0]) if selected else 0, source + 1,
                            blueprint_id,
                            0 if value is None else value + 1, int(template in ("<pass>", "<none>"))))
    head = np.asarray([DTYPE_INDEX[record[4]], vocabulary.lookup("q", record[5]), 0,
                       0 if record[3] == "DS" else 1, *record[7]], dtype=np.int32)
    # Check before casting: uint8 conversion must never silently wrap overflowing bounds.
    grid = np.asarray(raw_grid).reshape(-1, 153)
    if np.any((grid < 0) | (grid > 255) | (grid % 1 != 0)):
        raise ValueError("Grid does not fit uint8")
    raw_tokens = np.asarray(integers).reshape(-1, 8)
    if np.any((raw_tokens < np.iinfo(np.int32).min) | (raw_tokens > np.iinfo(np.int32).max)):
        raise ValueError("Token integer exceeds ABI-4 int32")
    item = {"schema_version": 4, "table_sha": tables.sha, "privileged": critic,
            "tok": raw_tokens.astype(np.int32), "tokf": float_rows(floats, width),
            "opt": np.asarray(options, np.int32).reshape(-1, 6), "head": head, "kgrid": grid.astype(np.uint8)}
    validate(item, vocabulary, len(tables.cards.ids), actor=not critic)
    return item


def float_rows(floats, width=None):
    """ABI-4 tokens carry 4 floats; schema-4 location tokens may add 4 location-state values (plan objective-dojo
    Amendment 2). width None keeps 4 unless a token has 8; tensor ABI 5 asks for 8 explicitly."""
    if any(len(values) not in (4, 8) for values in floats):
        raise ValueError("Token float rows must have 4 or 8 values")
    width = width or (8 if any(len(values) == 8 for values in floats) else 4)
    if width == 4 and any(len(values) == 8 for values in floats):
        raise ValueError("Eight-float tokens in a four-float (ABI-4) item")
    rows = np.zeros((len(floats), width), np.float32)
    for index, values in enumerate(floats):
        rows[index, :len(values)] = values
    return rows


def stack(items, device="cpu", pad=None, pin=False):
    if not items or any(item.get("schema_version") != 4 for item in items):
        raise ValueError("Nonempty schema-4 batch required")
    shape = (len(items), max(1, max(len(item["tok"]) for item in items)),
             max(1, max(len(item["opt"]) for item in items)), max(1, max(len(item["kgrid"]) for item in items)))
    batch, length, options, rows = shape if pad is None else pad
    if any(actual > padded for actual, padded in zip(shape, (batch, length, options, rows))):
        raise ValueError("Padding cannot truncate observations")
    arrays = {"tok": np.zeros((batch, length, 8), np.int32),
              "tokf": np.zeros((batch, length, max(item["tokf"].shape[1] for item in items)), np.float32),
              "opt": np.zeros((batch, options, 6), np.int32), "head": np.zeros((batch, 22), np.int32),
              "tmask": np.zeros((batch, length), bool), "omask": np.zeros((batch, options), bool),
              "kgrid": np.zeros((batch, rows, 153), np.uint8)}
    for index, item in enumerate(items):
        for key in ("tok", "opt", "kgrid"):
            arrays[key][index, :len(item[key])] = item[key]
        arrays["tokf"][index, :len(item["tokf"]), :item["tokf"].shape[1]] = item["tokf"]
        arrays["head"][index] = item["head"]
        arrays["tmask"][index, :len(item["tok"]) ] = True
        arrays["omask"][index, :len(item["opt"])] = True
    result = {}
    for key, array in arrays.items():
        tensor = torch.from_numpy(array)
        result[key] = (tensor.pin_memory() if pin else tensor).to(device, non_blocking=pin)
    head = result.pop("head")
    for key, column in (("dtype", 0), ("qtmpl", 1), ("qbp", 2), ("side", 3)):
        result[key] = head[:, column]
    result["glob"] = head[:, 4:]
    return result
