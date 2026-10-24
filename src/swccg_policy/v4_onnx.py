"""ABI-4 browser export with owned precomputed tables and dynamic grids."""

import copy

import torch

from .v4_native import FlatPolicy


INPUTS = ["tok", "tokf", "tmask", "opt", "omask", "dtype", "qtmpl", "qbp", "side", "glob", "kgrid"]


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flat = FlatPolicy(copy.deepcopy(model).cpu().eval())

    def forward(self, tok, tokf, tmask, opt, omask, dtype, qtmpl, qbp, side, glob, kgrid):
        head = torch.cat((dtype[:, None], qtmpl[:, None], qbp[:, None], side[:, None], glob), dim=1)
        return self.flat(tok, tokf, tmask, opt, omask, head, kgrid)


def export(model, batch, path):
    wrapper = Wrapper(model).eval()
    batch_size = torch.export.Dim("B", min=1, max=1024)
    tokens = torch.export.Dim("T", min=1, max=8192)
    options = torch.export.Dim("O", min=1, max=100000)
    grids = torch.export.Dim("K", min=1, max=8192)
    shapes = {key: {0: batch_size} for key in INPUTS}
    for key in ("tok", "tokf", "tmask"):
        shapes[key][1] = tokens
    for key in ("opt", "omask"):
        shapes[key][1] = options
    shapes["kgrid"][1] = grids
    arguments = tuple(batch[key].cpu() for key in INPUTS)
    with torch.no_grad():
        program = torch.onnx.export(wrapper, arguments, input_names=INPUTS, output_names=["logits"],
                                    dynamic_shapes=shapes, dynamo=True, opset_version=18)
    program.save(path)
