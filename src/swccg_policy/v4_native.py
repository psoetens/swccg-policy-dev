"""ABI-4 flat inference with owned precomputed card and template buffers."""

import copy
import os
from pathlib import Path

import torch


class FlatPolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.m.clear_tables()
        self.register_buffer("tensor_abi", torch.tensor(model.cfg.tensor_abi))
        self.register_buffer("table_sha", torch.tensor(list(bytes.fromhex(model.table_sha)), dtype=torch.uint8))
        with torch.no_grad():
            for name, module in (("card_table", model.cards), ("question_table", model.qtmpl),
                                 ("option_table", model.otmpl), ("context_table", model.context)):
                self.register_buffer(name, module.table().detach().clone())

    def batch(self, tok, tokf, tmask, opt, omask, head, kgrid):
        self.m._card_table = self.card_table
        self.m._question_table = self.question_table
        self.m._option_table = self.option_table
        self.m._context_table = self.context_table
        return {"tok": tok, "tokf": tokf, "tmask": tmask, "opt": opt, "omask": omask, "kgrid": kgrid,
                "dtype": head[:, 0], "qtmpl": head[:, 1], "qbp": head[:, 2], "side": head[:, 3],
                "glob": head[:, 4:]}

    def forward(self, tok, tokf, tmask, opt, omask, head, kgrid):
        return self.m(self.batch(tok, tokf, tmask, opt, omask, head, kgrid)).float()

    def forward_packed(self, tok, tokf, tmask, opt, omask, head, kgrid, edge_rows, question_rows, option_rows):
        batch = self.batch(tok, tokf, tmask, opt, omask, head, kgrid)
        batch.update(edge_rows=edge_rows, question_rows=question_rows, option_rows=option_rows)
        return self.m(batch).float()


def example(device, width=4):
    tokens = torch.zeros((2, 16, 8), dtype=torch.int32, device=device)
    floats = torch.zeros((2, 16, width), device=device)
    token_mask = torch.ones((2, 16), dtype=torch.bool, device=device)
    options = torch.zeros((2, 3, 6), dtype=torch.int32, device=device)
    option_mask = torch.ones((2, 3), dtype=torch.bool, device=device)
    head = torch.zeros((2, 22), dtype=torch.int32, device=device)
    grid = torch.zeros((2, 1, 153), dtype=torch.uint8, device=device)
    return tokens, floats, token_mask, options, option_mask, head, grid


def export_ts(model, path):
    device = next(model.parameters()).device
    flat = FlatPolicy(copy.deepcopy(model).eval())
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            inputs = example(device, 8 if model.cfg.tensor_abi == 5 else 4)
            rows = tuple(torch.full((2,), -1, dtype=torch.long, device=device) for _ in range(3))
            traced = torch.jit.trace_module(flat, {"forward": inputs, "forward_packed": (*inputs, *rows)},
                                             check_trace=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    traced.save(str(temporary))
    os.replace(temporary, path)
    return traced


def update_traced(traced, model):
    """Copy weights and recomputed inference buffers without tracing or changing storage addresses."""
    if int(traced.tensor_abi) != model.cfg.tensor_abi or model.cfg.tensor_abi not in (4, 5):
        raise ValueError("ABI-4/5 traced module of the model's tensor ABI required")
    incoming = FlatPolicy(copy.deepcopy(model).eval()).state_dict()
    current = traced.state_dict()
    if not torch.equal(current["table_sha"].cpu(), incoming["table_sha"].cpu()):
        raise ValueError("In-place update cannot change the acting table SHA")
    if set(incoming) != set(current) or any(current[key].shape != incoming[key].shape for key in current):
        raise ValueError("In-place update requires identical architecture and table dimensions")
    with torch.no_grad():
        for key, destination in current.items():
            destination.copy_(incoming[key])
