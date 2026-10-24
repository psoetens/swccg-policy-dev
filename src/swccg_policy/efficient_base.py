"""Shared base of the V3.1 knowledge model (schema 4, `efficient.py`).

Plan remove-old-generations: the parts V3.1 used from the removed V2 (`knowledge.py`) and V3 (`relational.py`)
modules, moved here unchanged: the configuration fields, the printed-card table, the reference pool, the option
attention block, and the policy's `forward` / `packed_encode` / `bits` / `rows`. `EfficientPolicy` supplies
`__init__`, `encode`, `queries` and `cache_card_table`; `EfficientConfig` supplies the model contract
(`__post_init__`).
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BaseConfig:
    n_cards: int
    n_card_features: int
    text_dim: int
    n_qtmpl: int
    n_otmpl: int
    n_context: int
    numeric_columns: list[int]
    schema_version: int = 4
    tensor_abi: int = 4
    model: str = 'knowledge_v3_1'
    d: int = 256
    layers: int = 6
    heads: int = 8
    ff: int = 1024
    dropout: float = 0.0
    relation_layers: int = 3
    option_layers: int = 2
    option_chunk: int = 128
    def to_dict(self): return asdict(self)

class PrintedCards(nn.Module):
    def __init__(self, cfg, features, text):
        super().__init__()
        numeric = cfg.numeric_columns
        other = [i for i in range(features.shape[1]) if i not in numeric]
        if not numeric or not other or len(set(numeric)) != len(numeric) or any(i < 0 or i >= features.shape[1] for i in numeric):
            raise ValueError('explicit numeric/presence and structured columns required')
        self.register_buffer('features', torch.as_tensor(features, dtype=torch.float32))
        self.register_buffer('text', torch.as_tensor(text, dtype=torch.float32))
        self.register_buffer('numeric_indices', torch.tensor(numeric))
        self.register_buffer('structured_indices', torch.tensor(other))
        self.numbers = nn.Sequential(nn.Linear(len(numeric), cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        self.structured = nn.Linear(len(other), cfg.d)
        self.text_projection = nn.Linear(text.shape[1], cfg.d)
        self.combine = nn.Sequential(nn.GELU(), nn.Linear(cfg.d, cfg.d))
        self.delta = nn.Embedding(features.shape[0], cfg.d)
        nn.init.zeros_(self.delta.weight)
    def prior(self):
        return self.combine(self.numbers(self.features[:, self.numeric_indices]) +
                            self.structured(self.features[:, self.structured_indices]) + self.text_projection(self.text))
    def table(self):
        from .learner_speed import compiled_table  # plan learner-speed S-2 (registered only by the learner)
        fast = compiled_table(self)
        if fast is not None:
            return fast()
        e = self.prior() + self.delta.weight
        return torch.cat([torch.zeros_like(e[:1]), e[1:]], 0)

class ReferencePool(nn.Module):
    """Gated sum conditioned on template and reference ordinal; retains counts."""
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1), nn.Sigmoid())
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.count = nn.Linear(2, d)
        self.count_scale = 60.0  # plan critic-v2: the critic's v2 input scaling sets 1.0
    def forward(self, refs, templates, owners, valid):
        weight=valid.unsqueeze(-1)
        total = templates.new_zeros(templates.shape).index_add(0, owners, (self.value(refs) * self.gate(torch.cat([refs, templates[owners]], -1)) * weight).to(templates.dtype))
        n = templates.new_zeros((templates.shape[0], 1)).index_add(0, owners, weight.to(templates.dtype))
        return total + self.count(torch.cat([n / self.count_scale, (n == 0).to(n.dtype)], -1))

class OptionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d
        self.heads = cfg.heads
        self.blueprint_bias = nn.Parameter(torch.full((cfg.heads,), 8.0))
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d); self.memory_norm = nn.LayerNorm(d); self.ffnorm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))
    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
    def prepare(self, memory):
        z = self.memory_norm(memory)
        return self.split(self.k(z)), self.split(self.v(z))
    def forward(self, queries, k, v, mask, matches):
        bias = matches[:, None].to(queries.dtype) * self.blueprint_bias[None, :, None, None]
        bias = bias.masked_fill(~mask[:, None, None, :], float('-inf'))
        z = F.scaled_dot_product_attention(self.split(self.q(self.norm(queries))), k, v, attn_mask=bias)
        # SDPA backends expose different strides; force a materialized layout
        # before merging heads so ONNX decomposition agrees with eager execution.
        queries = queries + self.out(z.transpose(1, 2).clone(memory_format=torch.contiguous_format).flatten(2))
        return queries + self.ff(self.ffnorm(queries))

class BasePolicy(nn.Module):
    """`forward` over the subclass's `encode` / `queries`; no parameters of its own."""
    def packed_encode(self, seq, padding):
        """Training-only FlashAttention over valid tokens, using the same weights.

        Export keeps the portable dense path. No positional embeddings or cross-
        example edges exist, so removing masked padding preserves valid outputs.
        """
        from torch.nn.attention.varlen import varlen_attn
        valid=(~padding).flatten().nonzero().flatten()
        lengths=(~padding).sum(1,dtype=torch.int32)
        cu=F.pad(lengths.cumsum(0,dtype=torch.int32),(1,0))
        x=seq.flatten(0,1).index_select(0,valid)
        heads=self.cfg.heads;dim=self.cfg.d//heads
        for layer in self.encoder.layers:
            qkv=F.linear(layer.norm1(x),layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias)
            q,k,v=qkv.reshape(-1,3,heads,dim).unbind(1)
            z=varlen_attn(q,k,v,cu,cu,seq.shape[1],seq.shape[1]).flatten(1)
            a=layer.dropout1(layer.self_attn.out_proj(z))
            if hasattr(layer,'gain1'): a=layer.gain1*layer.out_norm1(a)  # plan dilution-names-ab, fix 1
            x=x+a
            z=layer.norm2(x)
            f=layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(z)))))
            if hasattr(layer,'gain2'): f=layer.gain2*layer.out_norm2(f)
            x=x+f
        x=self.norm(x)
        return x.new_zeros(seq.numel()//seq.shape[-1],seq.shape[-1]).index_copy(0,valid,x).reshape(seq.shape)
    @staticmethod
    def bits(values, n):
        return ((values.unsqueeze(-1) // (2**torch.arange(n, device=values.device))) % 2).float()
    @staticmethod
    def rows(b, key, mask):
        # Native capture supplies bounded sparse-row buckets with -1 padding.
        # Eager/ONNX continue to discover exactly the actual sparse rows.
        if key in b:
            rows=b[key].long()
            return rows.clamp_min(0), rows>=0
        rows=mask.flatten().nonzero().flatten()
        return rows,torch.ones_like(rows,dtype=torch.bool)
    def forward(self, b, return_hidden=False):
        h,E = self.encode(b)
        queries = self.queries(b,h,E)
        mask = torch.cat([torch.ones_like(b['tmask'][:,:1]).expand(-1,2),b['tmask']],1)
        kv = [layer.prepare(h) for layer in self.decoder]
        memory_bp = torch.cat([torch.zeros_like(b['opt'][:, :1, 3]).expand(-1, 2), b['tok'][..., 1]], 1)
        option_bp = b['opt'][..., 3]
        # A traced model evaluates all queries together; eager training chunks
        # exactly since neither normalization nor attention mixes option rows.
        if torch.jit.is_tracing() or torch.compiler.is_compiling():
            matches = (option_bp[:, :, None] == memory_bp[:, None, :]) & (option_bp[:, :, None] > 1)
            z=queries
            for layer,(k,v) in zip(self.decoder,kv): z=layer(z,k,v,mask,matches)
            logits=self.score(z).squeeze(-1)
        else:
            parts=[]
            for z, bp in zip(queries.split(self.cfg.option_chunk,dim=1), option_bp.split(self.cfg.option_chunk,dim=1)):
                match = (bp[:, :, None] == memory_bp[:, None, :]) & (bp[:, :, None] > 1)
                for layer,(k,v) in zip(self.decoder,kv): z=layer(z,k,v,mask,match)
                parts.append(self.score(z).squeeze(-1))
            logits=torch.cat(parts,1)
        logits=logits.float().masked_fill(~b['omask'],float('-inf'))
        return (logits,torch.cat([h[:,0],h[:,1]],-1)) if return_hidden else logits
