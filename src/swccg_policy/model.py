"""Card-embedding encoder + option-scoring policy.

E[c] = MLP(structured_features(c) ⊕ text_embedding(c)) + δ[c]   (features/text frozen buffers, δ zero-initialised)
Sequence = [question token, global token, card/location/unseen tokens...] -> transformer encoder.
Each legal option is encoded (template, blueprint(s), value, none-flag, contextual output of its source token) and scored
against the question token's output. Softmax over the legal options only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ZONES = 40
N_KINDS = 3
N_SLOTS = 40
N_FLAGS = 12
N_GLOBAL = 18
N_DTYPES = 7
N_VALUES = 33


@dataclass
class ModelConfig:
    n_cards: int
    n_card_features: int
    text_dim: int
    n_qtmpl: int
    n_otmpl: int
    d: int = 384
    layers: int = 6
    heads: int = 8
    ff: int = 1536
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class CardEmbedding(nn.Module):
    def __init__(self, features: np.ndarray, text: np.ndarray, d: int):
        super().__init__()
        self.register_buffer("features", torch.tensor(features, dtype=torch.float32), persistent=True)
        self.register_buffer("text", torch.tensor(text, dtype=torch.float32), persistent=True)
        self.mlp = nn.Sequential(nn.Linear(features.shape[1] + text.shape[1], 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.delta = nn.Embedding(features.shape[0], d)
        nn.init.zeros_(self.delta.weight)

    def table(self) -> torch.Tensor:
        e = self.mlp(torch.cat([self.features, self.text], dim=1)) + self.delta.weight
        e = e.clone()
        e[0] = 0.0  # padding blueprint
        return e

    def prior(self) -> torch.Tensor:
        """Embedding without the learned per-card residual (what the shared MLP predicts from features + text)."""
        return self.mlp(torch.cat([self.features, self.text], dim=1))


class Policy(nn.Module):
    def __init__(self, cfg: ModelConfig, features: np.ndarray, text: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.cards = CardEmbedding(features, text, d)
        self.zone = nn.Embedding(N_ZONES, d)
        self.kind = nn.Embedding(N_KINDS, d)
        self.mine = nn.Embedding(2, d)
        self.slot = nn.Embedding(N_SLOTS, d)
        self.flags = nn.Linear(N_FLAGS, d, bias=False)
        self.count = nn.Linear(1, d, bias=False)
        self.stats = nn.Linear(8, d)
        self.parent = nn.Linear(d, d, bias=False)
        self.phase = nn.Embedding(32, d)
        self.glob = nn.Linear(N_GLOBAL, d)
        self.side = nn.Embedding(2, d)
        self.dtype = nn.Embedding(N_DTYPES, d)
        self.qtmpl = nn.Embedding(cfg.n_qtmpl, d)
        self.otmpl = nn.Embedding(cfg.n_otmpl, d)
        self.value = nn.Embedding(N_VALUES, d)
        self.none = nn.Embedding(2, d)
        self.type_pos = nn.Embedding(3, d)  # 0 question, 1 global, 2 card token
        self.text_bp = nn.Linear(d, d, bias=False)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.opt_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def cache_card_table(self) -> None:
        """Inference only: compute the card embedding table once (it depends only on the weights) instead of in every forward.
        Must not be used while training; a model with new weights must call it again."""
        with torch.no_grad():
            self._card_table = self.cards.table()

    def forward(self, b: dict, return_hidden: bool = False):
        """Returns option logits [B, O] (invalid options = -inf); with return_hidden also the contextual outputs of the
        question and global tokens [B, 2d] (used by the RL value head)."""
        E = self._card_table if getattr(self, "_card_table", None) is not None else self.cards.table()
        tok = b["tok"].long()                     # [B, T, 8]
        tokf = b["tokf"]                          # [B, T, 4] (nan = n/a)
        tmask = b["tmask"]                        # [B, T] True = real token
        B, T, _ = tok.shape
        kind, bp, zone, mine, slot, attach, count, flags = tok.unbind(-1)
        # floor(flags / 2^i) mod 2 == (flags >> i) & 1 for these non-negative flags, and exports to ONNX (bitwise ops do not).
        pow2 = torch.pow(2, torch.arange(N_FLAGS, device=tok.device)).to(flags.dtype)
        bits = torch.remainder(torch.div(flags.unsqueeze(-1), pow2, rounding_mode="floor"), 2).float()
        stat_mask = torch.isnan(tokf)
        stat = torch.cat([torch.nan_to_num(tokf, nan=0.0) / 5.0, (~stat_mask).float()], dim=-1)
        parent_idx = (attach - 1).clamp(min=0)
        parent_bp = torch.gather(bp, 1, parent_idx)
        parent_e = E[parent_bp] * (attach > 0).unsqueeze(-1).float()
        x = (E[bp] + self.zone(zone.clamp(0, N_ZONES - 1)) + self.kind(kind.clamp(0, N_KINDS - 1)) + self.mine(mine.clamp(0, 1))
             + self.slot(slot.clamp(0, N_SLOTS - 1)) + self.flags(bits) + self.count(torch.log1p(count.float()).unsqueeze(-1))
             + self.stats(stat) + self.parent(parent_e) + self.type_pos.weight[2])
        g = b["glob"].float()
        gn = torch.cat([g[:, :2] / 10.0, g[:, 3:4], g[:, 4:16] / 30.0, g[:, 16:18]], dim=1)
        gn = F.pad(gn, (0, N_GLOBAL - gn.shape[1]))
        gtok = self.glob(gn) + self.phase(g[:, 2].long().clamp(0, 31) + 0) + self.side(b["side"].long()) + self.type_pos.weight[1]
        qtok = self.dtype(b["dtype"].long()) + self.qtmpl(b["qtmpl"].long()) + E[b["qbp"].long()] + self.type_pos.weight[0]
        seq = torch.cat([qtok.unsqueeze(1), gtok.unsqueeze(1), x], dim=1)
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=tok.device), ~tmask], dim=1)
        h = self.norm(self.encoder(seq, src_key_padding_mask=pad))
        hq, htok = h[:, 0], h[:, 2:]
        opt = b["opt"].long()                     # [B, O, 6]
        omask = b["omask"]
        tmpl, tbp, src, obp, val, none = opt.unbind(-1)
        src_idx = (src - 1).clamp(min=0)
        src_h = torch.gather(htok, 1, src_idx.unsqueeze(-1).expand(-1, -1, htok.shape[-1])) * (src > 0).unsqueeze(-1).float()
        o = (self.otmpl(tmpl) + E[obp] + self.text_bp(E[tbp]) + self.value(val.clamp(0, N_VALUES - 1)) + self.none(none.clamp(0, 1))
             + self.dtype(b["dtype"].long()).unsqueeze(1))
        o = self.opt_mlp(torch.cat([o, src_h], dim=-1))
        hq_e = hq.unsqueeze(1).expand_as(o)
        logits = self.score(torch.cat([hq_e, o, hq_e * o], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~omask, float("-inf"))
        if return_hidden:
            return logits, torch.cat([hq, h[:, 1]], dim=-1)
        return logits


def imitation_loss(logits: torch.Tensor, chosen: torch.Tensor, omask: torch.Tensor) -> torch.Tensor:
    """-log P(chosen set) = -(logsumexp over chosen - logsumexp over legal)."""
    logits = logits.float()
    all_ = torch.logsumexp(logits, dim=1)
    ch = logits.masked_fill(~(chosen & omask), float("-inf"))
    return (all_ - torch.logsumexp(ch, dim=1)).mean()
