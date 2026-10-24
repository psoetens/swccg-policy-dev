"""Offline template priors, reproducible identities and exact prefix growth."""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .learner_speed import compiled_table, single_gemm
from .efficient_base import PrintedCards
from .v4_vocabulary import OOV_BUCKETS


BLOCK = 64


def fixed_blocks(rows, transform, single=False):
    """Keep GEMM dimensions unchanged when rows are appended, including the final block. Plan learner-speed S-5
    (single=True, learner only): the same row-wise transform over all blocks in one call (one GEMM per layer)."""
    if single:
        count = rows.shape[0]
        return transform(F.pad(rows, (0, 0, 0, -count % BLOCK)))[:count]
    output = []
    for start in range(0, rows.shape[0], BLOCK):
        block = rows[start:start + BLOCK]
        size = block.shape[0]
        padded = F.pad(block, (0, 0, 0, BLOCK - size))
        output.append(transform(padded)[:size])
    return torch.cat(output, dim=0)


class TextPriorEmbedding(nn.Module):
    def __init__(self, priors, width, *, keys=None, family="q", seed=0, prior_scale=1.0, identity_std=0.0,
                 oov=OOV_BUCKETS):
        super().__init__()
        # Plan actor-v6-inputs: oov = this family's reserved bucket rows (0 for a context-v5 table without buckets).
        self.oov = oov
        prior = torch.as_tensor(priors, dtype=torch.float16)
        if prior.ndim != 2 or prior.shape[1] != 768 or prior.shape[0] < oov:
            raise ValueError("Expected [OOV buckets + known, 768] template priors")
        if torch.any(prior[:oov] != 0):
            raise ValueError("OOV buckets cannot have text priors")
        self.register_buffer("priors", prior)
        self.keys = self.ordered_keys(keys, len(prior))
        self.family = family
        self.seed = seed
        self.prior_scale = prior_scale
        self.identity_std = identity_std
        self.projection = nn.Linear(768, width)
        self.residual = nn.Embedding(len(prior), width)
        if identity_std:
            with torch.no_grad():
                self.residual.weight.copy_(self.identity_rows(0, len(prior)))
        else:
            # Checkpoints predating A9 had zero known residuals and learned OOV rows.
            nn.init.zeros_(self.residual.weight)
            nn.init.normal_(self.residual.weight[:oov], std=0.02)

    @staticmethod
    def ordered_keys(keys, size):
        if keys is None:
            return tuple(str(index) for index in range(size))
        if len(keys) != size or sorted(keys.values()) != list(range(size)):
            raise ValueError("Template keys must cover each prior row exactly once")
        return tuple(key for key, _ in sorted(keys.items(), key=lambda item: item[1]))

    def identity_rows(self, start, stop):
        rows = torch.empty((stop - start, self.residual.embedding_dim), dtype=torch.float32)
        for index in range(start, stop):
            identity = ([self.family, "bucket", index] if index < self.oov
                        else [self.seed, self.family, "key", self.keys[index]])
            encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()
            row_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & ((1 << 63) - 1)
            generator = torch.Generator(device="cpu").manual_seed(row_seed)
            rows[index - start].normal_(std=self.identity_std, generator=generator)
        return rows.to(self.residual.weight)

    def table(self):
        fast = compiled_table(self)  # plan learner-speed S-2 (registered only by the learner)
        if fast is not None:
            return fast()
        known = self.priors[self.oov:].to(self.projection.weight.dtype)
        if known.shape[0] == 0:
            return self.residual.weight
        projected = fixed_blocks(known, self.projection, single_gemm(self)) * self.prior_scale
        return torch.cat((self.residual.weight[:self.oov],
                          projected + self.residual.weight[self.oov:]), dim=0)

    def forward(self, ids):
        return F.embedding(ids, self.table())

    def grow(self, priors, keys=None):
        incoming = torch.as_tensor(priors, dtype=self.priors.dtype, device=self.priors.device)
        size = len(self.priors)
        if len(incoming) < size or not torch.equal(incoming[:size], self.priors):
            raise ValueError("Text priors are not an exact prefix extension")
        incoming_keys = self.ordered_keys(keys, len(incoming))
        if incoming_keys[:size] != self.keys:
            raise ValueError("Template keys are not an exact prefix extension")
        self.keys = incoming_keys
        residual = self.residual.weight.new_zeros((len(incoming), self.residual.embedding_dim))
        residual[:size] = self.residual.weight.detach()
        if self.identity_std:
            residual[size:] = self.identity_rows(size, len(incoming))
        self.priors = incoming
        self.residual = nn.Embedding.from_pretrained(residual, freeze=False)


class GrowingPrintedCards(PrintedCards):
    def prior(self):
        rows = torch.cat((self.features, self.text), dim=-1)
        feature_width = self.features.shape[1]

        def project(block):
            features = block[:, :feature_width]
            text = block[:, feature_width:]
            return self.combine(self.numbers(features[:, self.numeric_indices])
                                + self.structured(features[:, self.structured_indices])
                                + self.text_projection(text))

        return fixed_blocks(rows, project, single_gemm(self))

    def grow(self, features, text):
        features = torch.as_tensor(features, dtype=self.features.dtype, device=self.features.device)
        text = torch.as_tensor(text, dtype=self.text.dtype, device=self.text.device)
        size = self.features.shape[0]
        if not torch.equal(features[:size], self.features) or not torch.equal(text[:size], self.text):
            raise ValueError("Card features/text are not an exact prefix extension")
        residual = self.delta.weight.new_zeros((len(features), self.delta.embedding_dim))
        residual[:size] = self.delta.weight.detach()
        self.features = features
        self.text = text
        self.delta = nn.Embedding.from_pretrained(residual, freeze=False)
