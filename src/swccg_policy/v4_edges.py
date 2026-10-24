"""Sparse ABI-4 edges, including context links carried directly by token rows."""

import torch


def sparse_edges(batch):
    tokens = batch["tok"].long()
    floats = batch["tokf"]
    kind, _, _, _, slot, parent, _, flags = tokens.unbind(-1)
    context = kind == 7
    linked = ((kind == 8) | (context & (parent > 0))) & batch["tmask"]
    if "edge_rows" in batch:
        packed = batch["edge_rows"].long()
        valid = packed >= 0
        rows = packed.clamp_min(0)
    else:
        rows = linked.flatten().nonzero().flatten()
        valid = torch.ones_like(rows, dtype=torch.bool)
    length = tokens.shape[1]
    offset = torch.div(rows, length, rounding_mode="floor") * length
    is_context = context.flatten()[rows]
    is_alias = is_context & ((flags.flatten()[rows] & 8) != 0)
    is_reference = is_context & ((flags.flatten()[rows] & 64) != 0)
    physical_source = parent.flatten()[rows] - 1 + offset
    source = torch.where(is_context, torch.where(is_alias, rows - 1, rows), physical_source)
    target = torch.where(is_context, parent.flatten()[rows] - 1, floats[..., 0].long().flatten()[rows] - 1)
    target = target + offset
    context_relation = torch.where(is_alias, 15, torch.where(is_reference, 16, 13))
    relation = torch.where(is_context, context_relation, slot.flatten()[rows])
    binding = torch.where(is_alias, rows - 1, rows)
    return tuple(torch.where(valid, value, 0) for value in (source, target, relation, binding)) + (valid,)


def degrees(reference, source, target, valid):
    result = reference.new_zeros((reference.shape[0], 2))
    zeros = torch.zeros_like(valid)
    outgoing = torch.stack((valid, zeros), -1).to(reference.dtype)
    incoming = torch.stack((zeros, valid), -1).to(reference.dtype)
    return result.index_add(0, source, outgoing).index_add(0, target, incoming)
