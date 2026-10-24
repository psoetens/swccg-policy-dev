"""Schema-4 relational model: bounds grids, text priors and inline context edges."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .efficient_base import BaseConfig, BasePolicy, OptionAttention, ReferencePool
from .text_priors import GrowingPrintedCards, TextPriorEmbedding
from .v4_vocabulary import buckets
from .v4_edges import degrees, sparse_edges


@dataclass
class EfficientConfig(BaseConfig):
    schema_version: int = 4
    tensor_abi: int = 4
    model: str = "knowledge_v3_1"
    fact_value: bool = False
    post_degree: bool = True
    grid_indicators: bool = True
    embedding_init_std: float = 0.02
    template_prior_scale: float = 33.0
    template_identity_std: float = 1.0
    template_identity_seed: int = 904000
    component_norm: bool = True
    component_label_gain: float = 0.1
    component_norm_scope: str = "options_context"
    # Amendment 13: actor soft cap cap*tanh(z/cap) on valid option logits; 0 disables it.
    logit_cap: float = 0.0
    # D16 B-run: "opp_hidden" removes opponent hidden-pile tracking from the actor input; "none" keeps all.
    knowledge_mask: str = "none"
    # Plan objective-dojo BL-20(a): printed and current Light/Dark Force icons of each location token (packed by the
    # engine into the location token's float a). The projection's last layer starts at zero, so an extended
    # checkpoint initially computes exactly what its parent did. Older checkpoints load with False.
    location_icons: bool = False
    # Amendment 2 (BL-20b): engine totals per location token (power, ability, drain; us and them) in tokf columns 4-7
    # under tensor ABI 5. Zero-initialised last layer, registered last, like location_icons.
    location_state: bool = False
    # Plan critic-v2 (critic only): "v2" replaces the uniform /60 input scaling by per-feature scales (V2_* below);
    # "legacy" is the original encoding. win_head adds a win-probability head next to the value head.
    input_scaling: str = "legacy"
    win_head: bool = False
    # Plan actor-v6-inputs: context scales under "v2" for context version 5 (p99 of index + 1 is 18, of |value| 8, on
    # v5 games); 0 keeps the critic-v2 constants V2_CONTEXT_INDEX / V2_CONTEXT_VALUE (measured on context v4).
    context_index_scale: float = 0.0
    context_value_scale: float = 0.0
    # Plan option-inputs-and-balance-probe (actor only): "v2" adds to every option's base (a) its integer value on the
    # pile-count scale, a value embedding and range-end flags, (b) the card embedding and, for a location, the location
    # state of the token the option points to, (c) log1p of the reference match. Every new projection starts at zero,
    # registered last, so an extended checkpoint initially computes exactly what its parent did. Older checkpoints load
    # with "legacy".
    option_inputs: str = "legacy"
    # Plan dilution-names-ab, fix 1: each transformer sublayer's output is layer-normalised and scaled by a learnable
    # per-channel gain (start residual_gain) before it is added, so no block can drown the token it reads (block 1's
    # feed-forward wrote 35-50x its input in the actor, 600-1,000x in the critic). Older checkpoints load with False.
    residual_norm: bool = False
    residual_gain: float = 1.25
    # Plan dilution-names-ab, fix 2: a message step after the graph rounds from each card token to the own card tokens
    # whose text names it or that it names (the within-deck combo map, a fixed buffer). Zero-initialised output.
    names_round: bool = False
    names_map: str = "reports/data/card-identity-combos/combo-map.json"
    # Plan links-round2 (BL-45): the names step's output is layer-normalised and scaled by a learnable per-channel gain
    # that starts at 0 (its MLP keeps a normal initialisation), so it cannot drown the token it reads and an extended
    # checkpoint initially computes exactly what its parent did. Older checkpoints load with False.
    names_norm: bool = False
    # Plan links-round2: counter links. Each card token reads the other side's card tokens in play that its text names
    # (the counter map, a fixed directed buffer: own card -> opponent card). Its own MLP, normalisation and zero-start
    # gain, as names_norm.
    counter_round: bool = False
    counter_map: str = "reports/data/links-round2/counter-map.json"
    # Plan multiselect-count stage B (actor only): the option row's column 1 (formerly reserved 0) marks an option whose
    # card is already selected in a step-by-step card selection (PolicyAi's count choice, Gemp's preselection); a
    # projection of that flag is added to the option's base. Zero-initialised and registered last, so an extended
    # checkpoint computes exactly what its parent did. Older checkpoints load with False (the flag is ignored).
    option_selected: bool = False

    def __post_init__(self):
        if (self.schema_version, self.model) != (4, "knowledge_v3_1") or self.tensor_abi not in (4, 5):
            raise ValueError("Incompatible V3.1 model contract")
        if self.location_state and self.tensor_abi != 5:
            raise ValueError("location_state needs tensor ABI 5 (eight floats per token)")
        if self.option_chunk < 1 or self.option_layers != 2 or self.heads != 8:
            raise ValueError("V3.1 requires two option blocks, eight heads and a positive chunk size")
        if self.knowledge_mask not in ("none", "opp_hidden"):
            raise ValueError("knowledge_mask must be 'none' or 'opp_hidden'")
        if not 0 <= self.logit_cap < float("inf"):
            raise ValueError("Logit cap must be finite and nonnegative")
        if not 0 < self.embedding_init_std <= 1:
            raise ValueError("Categorical embedding initialization std must be in (0, 1]")
        if self.input_scaling not in ("legacy", "v2"):
            raise ValueError("input_scaling must be 'legacy' or 'v2'")
        if self.names_norm and not self.names_round:
            raise ValueError("names_norm needs names_round")
        if self.option_inputs not in ("legacy", "v2"):
            raise ValueError("option_inputs must be 'legacy' or 'v2'")
        if (self.context_index_scale or self.context_value_scale) and self.input_scaling != "v2":
            raise ValueError("context scales apply to input_scaling 'v2' only")
        if self.context_index_scale < 0 or self.context_value_scale < 0:
            raise ValueError("context scales must be nonnegative")
        if not 0 < self.template_prior_scale < float("inf") or not 0 <= self.template_identity_std < float("inf"):
            raise ValueError("Template scale must be finite and positive; identity std finite and nonnegative")


class DirectedRound(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.direction = nn.Embedding(2, width)
        self.message = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Sequential(nn.Linear(4 * width, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden, source, target, relation, valid):
        sender = torch.cat((source, target))
        receiver = torch.cat((target, source))
        direction = torch.cat((torch.zeros_like(source), torch.ones_like(source)))
        inputs = torch.cat((hidden[sender], hidden[receiver], torch.cat((relation, relation)),
                            self.direction(direction)), -1)
        weight = torch.cat((valid, valid)).unsqueeze(-1)
        messages = (self.message(inputs) * self.gate(inputs) * weight).to(hidden.dtype)
        total = torch.zeros_like(hidden).index_add(0, receiver, messages)
        return self.norm(hidden + total)


COMPONENTS = ("card", "kind", "zone", "owner", "numeric", "entity_flags", "retained", "context_index",
              "context_value", "sentinel", "context_role", "context_template", "reference_flags", "position",
              "grid", "degree", "question_template", "dtype", "glob", "phase", "side")
OPTION_COMPONENTS = ("option_template", "option_card")
LABEL_COMPONENTS = ("kind", "zone", "owner", "sentinel", "context_role", "dtype", "side")
CONTEXT_COMPONENTS = ("context_index", "context_value", "sentinel", "context_role", "context_template")
# D16: opponent hidden piles (Reserve, Force, Used, Hand, Transit, Insert) and the cross-pile cells built from them.
HIDDEN_QUANTITIES = (0, 1, 2, 4, 11, 12, 13, 14, 15, 16)
HIDDEN_ORDER_ZONES = tuple(32 + pile for pile in (0, 1, 2, 4, 11, 12))


def hidden_mask_constants(device=None):
    """Removed grid cells and order zones as tensors. Models keep them as buffers: the native runtime
    captures CUDA graphs, which forbid host-to-device copies of Python-list constants."""
    cells = torch.zeros((17, 3, 3), dtype=torch.bool, device=device)
    cells[list(HIDDEN_QUANTITIES)] = True
    return cells.flatten(), torch.tensor(HIDDEN_ORDER_ZONES, dtype=torch.int32, device=device)


def mask_opponent_hidden(batch, cells=None, zones=None):
    """Return a batch without opponent hidden-pile tracking (relative owner 0 is the opponent)."""
    if cells is None or zones is None:
        cells, zones = hidden_mask_constants(batch["kgrid"].device)
    tokens = batch["tok"]
    kind = tokens[..., 0]
    opponent = tokens[..., 3] == 0
    known = (kind == 2) & opponent & batch["tmask"]
    grid = batch["kgrid"]
    rows = torch.arange(grid.shape[1], device=grid.device)
    # The slot field is a grid row only for knowledge tokens; other kinds reuse it with other meanings.
    # Comparisons instead of scatter/gather stay valid for any row count, including zero rows.
    match = ((tokens[..., 4].long() - 1).unsqueeze(-1) == rows) & known.unsqueeze(-1)
    opponent_rows = match.any(1)
    grid = grid.masked_fill(opponent_rows.unsqueeze(-1) & cells, 0)
    asserted = grid.view(*grid.shape[:2], 51, 3)[..., 2].amax(-1) > 0
    remaining = (match & asserted.unsqueeze(1)).any(-1)
    order = (kind == 5) & opponent & (tokens[..., 2].unsqueeze(-1) == zones).any(-1)
    drop = (known & ~remaining) | order
    return {**batch, "kgrid": grid, "tmask": batch["tmask"] & ~drop}


# Plan critic-v2, input_scaling "v2": the p99 of each raw input over collect-s2b (critic-probes P-1 census,
# reports/data/critic-probes/census.json). Flags and signs stay at unit scale; my/their pairs share one scale.
V2_NUMERIC = (1.0, 44.0, 1.0, 1.0, 14.0)  # count, pile index b, zone owner c (+-1), inserted d (0/1), location slot
V2_CONTEXT_INDEX = 3.0
V2_CONTEXT_VALUE = 56.0
V2_POSITION = 39.0
V2_GRID_BOUNDS = 43.0  # lower and upper bounds of the same pile counts
V2_LOCATION_ICONS = 3.0
V2_LOCATION_STATE = (17.0, 17.0, 17.0, 17.0, 2.0, 2.0)  # power us/them, ability us/them, drain us/them
V2_QUESTION_REFS = 1.0
# Plan option-inputs-and-balance-probe: integer option values share the pile-count scale of the globals (Force Pile,
# Reserve Deck: V2_GLOB), so "activate 4" compares directly with them; the embedding covers 0..OPTION_VALUES-1.
V2_OPTION_VALUE = 51.0
OPTION_VALUES = 32
# glob = head[4:22]: turn number us/them, phase, my-turn flag, Life Force us/them, reserve, Force pile, used pile, lost
# pile, hand (us/them each), battle flag, drain flag. Phase gets weight 0: the phase embedding carries it.
V2_GLOB = (23.0, 23.0, 0.0, 1.0, 51.0, 51.0, 51.0, 51.0, 17.0, 17.0, 15.0, 15.0, 49.0, 49.0, 26.0, 26.0, 1.0, 1.0)
V2_GLOB_GAIN = tuple(1.0 / scale if scale else 0.0 for scale in V2_GLOB)

QUESTION_COMPONENTS = ("question_template", "dtype", "glob", "phase", "side")
ENTITY_COMPONENTS = ("card", "kind", "zone", "owner", "numeric", "entity_flags", "retained", "reference_flags",
                     "position", "grid", "degree")
# Amendment 10/10c scopes plus the D15 synthetic hypotheses (H1): which parts are left un-normalized.
NORM_EXCLUSIONS = {"all": (), "all_but_labels": LABEL_COMPONENTS, "all_but_entity": ENTITY_COMPONENTS,
                   "all_but_grid": ("grid", "degree")}


class BoundedEncoderLayer(nn.TransformerEncoderLayer):
    """Plan dilution-names-ab, fix 1: a pre-norm encoder layer whose two sublayer outputs are each layer-normalised and
    scaled by a learnable per-channel gain before they are added to the residual stream. Same parameter names as
    nn.TransformerEncoderLayer plus out_norm1/2 (no parameters) and gain1/2; always the explicit (non-fast) path."""

    def __init__(self, *args, gain=1.25, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.linear2.out_features
        self.out_norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.out_norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.gain1 = nn.Parameter(torch.full((width,), float(gain)))
        self.gain2 = nn.Parameter(torch.full((width,), float(gain)))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self.gain1 * self.out_norm1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
        return x + self.gain2 * self.out_norm2(self._ff_block(self.norm2(x)))


def names_pairs(ids, path):
    """Plan dilution-names-ab, fix 2: a symmetric [cards, cards] boolean matrix of the combo map's (payoff, enabler)
    pairs, indexed like the card table."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    combo = json.loads((Path(path) if Path(path).is_absolute() else root / path).read_text())
    index = {card: position for position, card in enumerate(ids)}
    matrix = torch.zeros((len(ids), len(ids)), dtype=torch.bool)
    for deck in combo["decks"].values():
        for pair in deck:
            a, b = index.get(pair["payoff"]), index.get(pair["enabler"])
            if a is not None and b is not None:
                matrix[a, b] = matrix[b, a] = True
    return matrix


def counter_pairs(ids, path):
    """Plan links-round2: a directed [cards, cards] boolean matrix, [own card, opponent card] true where the own card's
    text names the opponent card (the counter map), indexed like the card table."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    counter = json.loads((Path(path) if Path(path).is_absolute() else root / path).read_text())
    index = {card: position for position, card in enumerate(ids)}
    matrix = torch.zeros((len(ids), len(ids)), dtype=torch.bool)
    for pair in counter["pairs"]:
        a, b = index.get(pair["card"]), index.get(pair["target"])
        if a is not None and b is not None:
            matrix[a, b] = True
    return matrix


class LinkStep(nn.Module):
    """Plan links-round2: mean of the linked tokens' hidden states, through a small MLP, layer-normalised and scaled by
    a per-channel gain that starts at 0 (the repaired form of fix 2's names step)."""

    def __init__(self, width):
        super().__init__()
        self.inp = nn.Linear(2 * width, width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.gain = nn.Parameter(torch.zeros(width))

    def forward(self, hidden, linked):
        weights = linked.to(hidden.dtype)
        degree = weights.sum(-1, keepdim=True)
        message = torch.bmm(weights, hidden) / degree.clamp_min(1.0)
        out = self.gain * self.norm(self.out(F.gelu(self.inp(torch.cat((hidden, message), -1)))))
        return out * (degree > 0).to(out.dtype)


IN_PLAY_ZONES = (0, 1, 2, 3)  # Gemp Zone ordinals: AT_LOCATION, SIDE_OF_TABLE, LOCATIONS, ATTACHED


class EfficientPolicy(BasePolicy):
    def __init__(self, cfg, tables, actor=True):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.table_sha = tables.sha
        self.actor = actor
        width = cfg.d
        cards = tables.cards.cards
        dimensions = (len(tables.cards.ids), cards["features"].shape[1], cards["text_emb"].shape[1],
                      *(len(tables.vocabulary.payload[family]) for family in ("q", "o", "context")))
        if dimensions != (cfg.n_cards, cfg.n_card_features, cfg.text_dim, cfg.n_qtmpl, cfg.n_otmpl, cfg.n_context):
            raise ValueError("Model configuration disagrees with owned table dimensions")
        self.cards = GrowingPrintedCards(cfg, cards["features"], cards["text_emb"])
        self.kind = nn.Embedding(11, width)
        self.zone = nn.Embedding(64, width)
        self.owner = nn.Embedding(2, width)
        self.entity_flags = nn.Linear(16, width, bias=False)
        self.scope = nn.Linear(3, width, bias=False)
        self.fact_flag = nn.Embedding(2, width)
        self.sentinel = nn.Embedding(8, width)
        self.context_role = nn.Linear(4, width, bias=False)
        self.reference_flags = nn.Linear(2, width, bias=False)
        self.numeric = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.context_value = nn.Linear(1, width)
        self.context_index = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.position = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.grid = nn.Sequential(nn.Linear(306 if cfg.grid_indicators else 153, width),
                                  nn.GELU(), nn.Linear(width, width))
        def template(family):
            return TextPriorEmbedding(tables.vocabulary.priors[family], width,
                                      keys=tables.vocabulary.payload[family], family=family,
                                      seed=cfg.template_identity_seed, prior_scale=cfg.template_prior_scale,
                                      identity_std=cfg.template_identity_std,
                                      oov=buckets(tables.vocabulary.payload, family))

        self.context = template("context")
        self.qtmpl = template("q")
        self.relation = nn.Embedding(17, width)
        self.degree = nn.Linear(2, width, bias=False)
        self.messages = nn.ModuleList(DirectedRound(width) for _ in range(cfg.relation_layers))
        self.bind_edge = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.dtype = nn.Embedding(7, width)
        self.question_pool = ReferencePool(width)
        if cfg.input_scaling == "v2":
            self.glob = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
            self.question_pool.count_scale = V2_QUESTION_REFS
        else:
            self.glob = nn.Linear(18, width)
        self.phase = nn.Embedding(32, width)
        if cfg.residual_norm:
            layer = BoundedEncoderLayer(width, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                        activation="gelu", gain=cfg.residual_gain)
        else:
            layer = nn.TransformerEncoderLayer(width, cfg.heads, cfg.ff, cfg.dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        # Amendment 10: one LayerNorm per input component so every part enters at the same scale.
        names = list(COMPONENTS) + (list(OPTION_COMPONENTS) if actor else [])
        if cfg.component_norm_scope == "options_context_question":
            # D15 step 2c: Amendment 10c plus the question-side parts.
            names = [name for name in names if name in OPTION_COMPONENTS or name in CONTEXT_COMPONENTS
                     or name in QUESTION_COMPONENTS]
        elif cfg.component_norm_scope == "options_context":
            # Amendment 10c: normalize only option bases and context sub-parts; entity/knowledge/question
            # token composition stays as in Amendment 9.
            names = [name for name in names if name in OPTION_COMPONENTS or name in CONTEXT_COMPONENTS]
        elif cfg.component_norm_scope in NORM_EXCLUSIONS:
            names = [name for name in names if name not in NORM_EXCLUSIONS[cfg.component_norm_scope]]
        else:
            raise ValueError("component_norm_scope must be 'options_context', 'options_context_question' or one of "
                             + ", ".join(NORM_EXCLUSIONS))
        self.component_norms = nn.ModuleDict({name: nn.LayerNorm(width) for name in names}) if cfg.component_norm else None
        if cfg.knowledge_mask == "opp_hidden":
            # Persistent: traced modules keep every buffer, and update_traced needs identical key sets on both
            # sides. Only masked (B) models carry these keys; every other checkpoint is unchanged.
            cells, zones = hidden_mask_constants()
            self.register_buffer("hidden_cells", cells)
            self.register_buffer("hidden_zones", zones)
        if self.component_norms is not None:
            # Amendment 10b: near-constant categorical labels start quiet so varying content (grid, card,
            # templates, values) dominates each token; gains are learned.
            with torch.no_grad():
                for name in LABEL_COMPONENTS:
                    if name in self.component_norms:
                        self.component_norms[name].weight.fill_(cfg.component_label_gain)
        if actor:
            if cfg.fact_value:
                self.fact_value = nn.Sequential(nn.Linear(153, width, bias=False), nn.LayerNorm(width))
                self.fact_gain = nn.Parameter(torch.tensor(0.1))
            if cfg.post_degree:
                self.retrieval_degree = nn.Linear(2, width, bias=False)
            self.reference_match = nn.Linear(1, width, bias=False)
            self.otmpl = template("o")
            self.option_pool = ReferencePool(width)
            self.option = nn.Sequential(nn.Linear(6 * width + 2, width), nn.GELU(), nn.LayerNorm(width))
            self.decoder = nn.ModuleList(OptionAttention(cfg) for _ in range(cfg.option_layers))
            for decoder in self.decoder:
                with torch.no_grad():
                    decoder.blueprint_bias[:4].zero_()
                    decoder.blueprint_bias[4:].fill_(4.0)
            self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            nn.init.orthogonal_(self.score[-1].weight, gain=0.01)
            nn.init.zeros_(self.score[-1].bias)
        # Explicit categorical list excludes card deltas and text-prior residuals.
        if cfg.embedding_init_std != 1.0:
            for module in self.categorical_embeddings().values():
                nn.init.normal_(module.weight, std=cfg.embedding_init_std)
        if cfg.location_icons:
            # Registered last, so an extended checkpoint keeps its parent's parameter order (optimizer state).
            self.location_icons = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.location_icons[-1].weight)
            nn.init.zeros_(self.location_icons[-1].bias)
        if cfg.location_state:
            self.location_state = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.location_state[-1].weight)
            nn.init.zeros_(self.location_state[-1].bias)
        if actor and cfg.option_inputs == "v2":
            # Registered last (after location_state), zero outputs: see EfficientConfig.option_inputs.
            self.option_value = nn.Embedding(OPTION_VALUES + 1, width)
            self.option_scalars = nn.Linear(3, width, bias=False)
            self.option_source_card = nn.Linear(width, width, bias=False)
            self.option_source_location = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
            self.option_reference_log = nn.Linear(1, width, bias=False)
            for module in (self.option_value, self.option_scalars, self.option_source_card,
                           self.option_source_location[-1], self.option_reference_log):
                nn.init.zeros_(module.weight)
            nn.init.zeros_(self.option_source_location[-1].bias)
        if cfg.names_round:
            # Registered last, zero output: an extended checkpoint initially computes exactly what its parent did.
            self.register_buffer("names_pairs", names_pairs(tables.cards.ids, cfg.names_map))
            if cfg.names_norm:
                self.names_step = LinkStep(width)
            else:
                self.names_in = nn.Linear(2 * width, width)
                self.names_out = nn.Linear(width, width)
                nn.init.zeros_(self.names_out.weight)
                nn.init.zeros_(self.names_out.bias)
        if cfg.counter_round:
            self.register_buffer("counter_pairs", counter_pairs(tables.cards.ids, cfg.counter_map))
            self.counter_step = LinkStep(width)
        if actor and cfg.option_selected:
            # Registered last, zero output: see EfficientConfig.option_selected.
            self.option_selected = nn.Linear(1, width, bias=False)
            nn.init.zeros_(self.option_selected.weight)

    def names_message(self, batch, hidden):
        """Fix 2: mean of the hidden states of the own card tokens linked by the combo map, through a small MLP."""
        tokens = batch["tok"].long()
        kind, blueprint, owner = tokens[..., 0], tokens[..., 1], tokens[..., 3]
        card = (kind == 0) & batch["tmask"] & (blueprint > 3)
        linked = self.names_pairs[blueprint.unsqueeze(-1), blueprint.unsqueeze(-2)]
        linked = linked & card.unsqueeze(-1) & card.unsqueeze(-2) & (owner.unsqueeze(-1) == owner.unsqueeze(-2))
        if self.cfg.names_norm:
            return self.names_step(hidden, linked)
        weights = linked.to(hidden.dtype)
        degree = weights.sum(-1, keepdim=True)
        message = torch.bmm(weights, hidden) / degree.clamp_min(1.0)
        out = self.names_out(F.gelu(self.names_in(torch.cat((hidden, message), -1))))
        return out * (degree > 0).to(out.dtype)

    def counter_message(self, batch, hidden):
        """Counter links: each card token reads the other side's card tokens in play that its text names."""
        tokens = batch["tok"].long()
        kind, blueprint, zone, owner = tokens[..., 0], tokens[..., 1], tokens[..., 2], tokens[..., 3]
        card = (kind == 0) & batch["tmask"] & (blueprint > 3)
        in_play = card & (zone <= IN_PLAY_ZONES[-1]) & (zone >= IN_PLAY_ZONES[0])
        linked = self.counter_pairs[blueprint.unsqueeze(-1), blueprint.unsqueeze(-2)]
        linked = linked & card.unsqueeze(-1) & in_play.unsqueeze(-2) & (owner.unsqueeze(-1) != owner.unsqueeze(-2))
        return self.counter_step(hidden, linked)

    def categorical_embeddings(self):
        names = ("kind", "zone", "owner", "fact_flag", "sentinel", "relation", "dtype", "phase")
        result = {name: getattr(self, name) for name in names}
        result.update({f"messages.{index}.direction": layer.direction
                       for index, layer in enumerate(self.messages)})
        return result

    def cache_card_table(self):
        self._card_table = self.cards.table().detach()
        self._context_table = self.context.table().detach()
        self._question_table = self.qtmpl.table().detach()
        if self.actor:
            self._option_table = self.otmpl.table().detach()

    def clear_tables(self):
        for name in ("_card_table", "_context_table", "_question_table", "_option_table"):
            setattr(self, name, None)

    def grow(self, old, new):
        if self.table_sha != old.sha:
            raise ValueError("Model does not own supplied parent table")
        old.assert_prefix_of(new)
        self.cards.grow(new.cards.cards["features"], new.cards.cards["text_emb"])
        self.context.grow(new.vocabulary.priors["context"], new.vocabulary.payload["context"])
        self.qtmpl.grow(new.vocabulary.priors["q"], new.vocabulary.payload["q"])
        if self.actor:
            self.otmpl.grow(new.vocabulary.priors["o"], new.vocabulary.payload["o"])
        if self.cfg.names_round:
            grown = names_pairs(new.cards.ids, self.cfg.names_map).to(self.names_pairs.device)
            grown[:self.names_pairs.shape[0], :self.names_pairs.shape[1]] |= self.names_pairs
            self.names_pairs = grown
        if self.cfg.counter_round:
            grown = counter_pairs(new.cards.ids, self.cfg.counter_map).to(self.counter_pairs.device)
            grown[:self.counter_pairs.shape[0], :self.counter_pairs.shape[1]] |= self.counter_pairs
            self.counter_pairs = grown
        self.table_sha = new.sha
        self.cfg.n_cards = len(new.cards.ids)
        self.cfg.n_qtmpl = len(new.vocabulary.payload["q"])
        self.cfg.n_otmpl = len(new.vocabulary.payload["o"])
        self.cfg.n_context = len(new.vocabulary.payload["context"])
        self.clear_tables()

    def part(self, name, value):
        if self.component_norms is None or name not in self.component_norms:
            return value
        return self.component_norms[name](value)

    def template(self, name, ids):
        module, cache_name = {"q": (self.qtmpl, "_question_table"),
                              "context": (self.context, "_context_table")}[name]
        cached = getattr(self, cache_name, None)
        return module(ids) if cached is None else F.embedding(ids, cached)

    def encode(self, batch):
        if self.cfg.knowledge_mask == "opp_hidden":
            batch = mask_opponent_hidden(batch, self.hidden_cells, self.hidden_zones)
        table = getattr(self, "_card_table", None)
        table = self.cards.table() if table is None else table
        tokens = batch["tok"].long()
        wide = batch["tokf"].float()
        # Columns 0-3 are the ABI-4 floats; ABI 5 adds the location state in columns 4-7.
        floats = wide[..., :4]
        kind, blueprint, zone, owner, slot, _, count, flags = tokens.unbind(-1)
        v2 = self.cfg.input_scaling == "v2"
        context = kind == 7
        known = kind == 2
        retained = (kind == 3) | (kind == 5)
        reference = (kind == 9) | (kind == 10)
        numeric = torch.cat((count.unsqueeze(-1).float(), floats[..., 1:],
                             ((slot - 1) * (kind == 1)).unsqueeze(-1).float()), -1)
        numeric = numeric / (numeric.new_tensor(V2_NUMERIC) if v2 else 60.0)
        numeric = numeric * (~context & ~known).unsqueeze(-1)
        card = self.part("card", F.embedding(blueprint, table)) * (blueprint > 0).unsqueeze(-1)
        hidden = card + self.part("kind", self.kind(kind)) + self.part("zone", self.zone(zone))
        hidden = hidden + self.part("owner", self.owner(owner))
        hidden = hidden + self.part("numeric", self.numeric(numeric)) * (~context & ~known).unsqueeze(-1)
        if self.cfg.location_icons:
            hidden = hidden + self.location_icons(unpack_location_icons(floats[..., 0], kind == 1,
                                                                        V2_LOCATION_ICONS if v2 else 60.0))
        if self.cfg.location_state:
            scale = wide.new_tensor(V2_LOCATION_STATE) if v2 else 60.0
            hidden = hidden + self.location_state(unpack_location_state(wide[..., 4:8], kind == 1, scale))
        hidden = hidden + self.part("entity_flags", self.entity_flags(self.bits(flags, 16))) * (kind <= 1).unsqueeze(-1)
        hidden = hidden + self.part("retained", self.scope(self.bits(flags // 2, 3))
                                    + self.fact_flag(flags.remainder(2))) * retained.unsqueeze(-1)
        indices = torch.stack((floats[..., 0], floats[..., 2], floats[..., 3], count.float()), -1)
        index_scale = (self.cfg.context_index_scale or V2_CONTEXT_INDEX) if v2 else 60.0
        value_scale = (self.cfg.context_value_scale or V2_CONTEXT_VALUE) if v2 else 60.0
        context_hidden = self.part("context_index", self.context_index((indices + 1) / index_scale))
        context_hidden = context_hidden + self.part("context_value", self.context_value(floats[..., 1:2] / value_scale))
        context_hidden = context_hidden + self.part("sentinel", self.sentinel(flags.remainder(8)))
        context_hidden = context_hidden + self.part("context_role", self.context_role(self.bits(flags // 8, 4)))
        context_ids = torch.where(context, slot, 0)
        context_hidden = context_hidden + self.part("context_template", self.template("context", context_ids))
        hidden = hidden + context_hidden * context.unsqueeze(-1)
        reference_bits = torch.stack((flags.remainder(2), (flags // 128).remainder(2)), -1).float()
        hidden = hidden + self.part("reference_flags", self.reference_flags(reference_bits)) * reference.unsqueeze(-1)
        hidden = hidden + self.part("position", self.position((slot - 1).unsqueeze(-1).float()
                                                              / (V2_POSITION if v2 else 60.0))) * (
            reference | (kind == 5)).unsqueeze(-1)
        grid_index = torch.where(known, slot - 1, 0)
        raw_grid = batch["kgrid"].gather(1, grid_index.unsqueeze(-1).expand(-1, -1, 153)).float()
        grid, projected_grid = grid_features(raw_grid, self.cfg.grid_indicators, V2_GRID_BOUNDS if v2 else 60.0)
        grid = grid * known.unsqueeze(-1)
        hidden = hidden + self.part("grid", self.grid(projected_grid)) * known.unsqueeze(-1)
        batch_size, length, width = hidden.shape
        flat = hidden.flatten(0, 1)
        source, target, relation_ids, binding, valid = sparse_edges(batch)
        relation = self.relation(relation_ids)
        degree = degrees(flat, source, target, valid)
        degree = torch.log1p(degree) if v2 else degree / 60.0
        flat = flat + self.part("degree", self.degree(degree))
        for layer in self.messages:
            flat = layer(flat, source, target, relation, valid)
        bound = self.bind_edge(torch.cat((flat[source], flat[target], relation), -1))
        flat = flat.index_add(0, binding, (bound * valid.unsqueeze(-1)).to(flat.dtype))
        hidden = flat.reshape(batch_size, length, width)
        if self.cfg.names_round or self.cfg.counter_round:
            update = 0
            if self.cfg.names_round:
                update = update + self.names_message(batch, hidden)
            if self.cfg.counter_round:
                update = update + self.counter_message(batch, hidden)
            hidden = hidden + update
        question = (self.part("question_template", self.template("q", batch["qtmpl"].long()))
                    + self.part("dtype", self.dtype(batch["dtype"].long())))
        rows, valid_rows = self.rows(batch, "question_rows", (kind == 9) & batch["tmask"])
        question = question + self.question_pool(flat[rows], question,
                                                torch.div(rows, length, rounding_mode="floor"), valid_rows)
        globals_ = batch["glob"].float()
        globals_scaled = globals_ * globals_.new_tensor(V2_GLOB_GAIN) if v2 else globals_ / 60.0
        global_hidden = (self.part("glob", self.glob(globals_scaled))
                         + self.part("phase", self.phase(globals_[:, 2].long().clamp(0, 31)))
                         + self.part("side", self.owner(batch["side"].long())))
        sequence = torch.cat((question[:, None], global_hidden[:, None], hidden), 1)
        padding = torch.cat((torch.zeros(batch_size, 2, dtype=torch.bool, device=hidden.device), ~batch["tmask"]), 1)
        if getattr(self, "packed_attention", False) and not torch.jit.is_tracing():
            encoded = self.packed_encode(sequence, padding)
        else:
            encoded = self.norm(self.encoder(sequence, src_key_padding_mask=padding))
        extra = torch.zeros_like(encoded[:, 2:])
        if self.cfg.fact_value and self.actor:
            extra = extra + self.fact_gain * self.fact_value(grid) * known.unsqueeze(-1)
        if self.cfg.post_degree and self.actor:
            extra = extra + self.retrieval_degree(degree.reshape(batch_size, length, 2))
        return torch.cat((encoded[:, :2], encoded[:, 2:] + extra), 1), table

    def queries(self, batch, hidden, table):
        template, selected, source, blueprint, value, none = batch["opt"].long().unbind(-1)
        batch_size, options = template.shape
        length = batch["tok"].shape[1]
        cached = getattr(self, "_option_table", None)
        base = self.part("option_template", self.otmpl(template) if cached is None else F.embedding(template, cached))
        base = base + self.part("option_card", F.embedding(blueprint, table)) * (blueprint > 0).unsqueeze(-1)
        source_hidden = hidden[:, 2:].gather(1, (source - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, self.cfg.d))
        source_hidden = source_hidden * (source > 0).unsqueeze(-1)
        rows, valid = self.rows(batch, "option_rows", (batch["tok"][..., 0] == 10) & batch["tmask"])
        owners = torch.where(valid, batch["tokf"][..., 0].long().flatten()[rows] - 1
                             + torch.div(rows, length, rounding_mode="floor") * options, 0)
        pooled = self.option_pool(hidden[:, 2:].flatten(0, 1)[rows], base.flatten(0, 1), owners, valid)
        pooled = pooled.reshape(batch_size, options, self.cfg.d)
        card_ids = batch["tok"][..., 1].long()
        key = (card_ids + torch.arange(batch_size, device=card_ids.device)[:, None] * table.shape[0]).flatten()
        question_ref = (batch["tok"][..., 0] == 9) & batch["tmask"] & (card_ids > 1)
        counts = hidden.new_zeros(batch_size * table.shape[0])
        counts = counts.index_add(0, key, question_ref.flatten().to(hidden.dtype))
        matches = counts[key[rows]] * (card_ids.flatten()[rows] > 1) * valid
        overlap = hidden.new_zeros(batch_size * options).index_add(0, owners, matches).reshape(batch_size, options, 1)
        pooled = pooled + self.reference_match(overlap / 60.0)
        if self.cfg.option_inputs == "v2":
            base = base + self.option_extras(batch, value, source, overlap, table)
        if self.cfg.option_selected:
            base = base + self.option_selected((selected > 0).unsqueeze(-1).to(base.dtype))
        question = hidden[:, :1].expand(-1, options, -1)
        return self.option(torch.cat((base, source_hidden, pooled, question, question * pooled,
                                      question * source_hidden, value.unsqueeze(-1).float() / 60.0,
                                      none.unsqueeze(-1).float()), -1))

    def option_extras(self, batch, value, source, overlap, table):
        """Plan option-inputs-and-balance-probe (option_inputs "v2"): what makes integer and card choices visible."""
        valid = batch["omask"] & (value > 0)
        number = (value - 1).clamp_min(0)
        floats = number.float()
        big = torch.finfo(floats.dtype).max
        highest = torch.where(valid, floats, torch.full_like(floats, -big)).amax(-1, keepdim=True)
        lowest = torch.where(valid, floats, torch.full_like(floats, big)).amin(-1, keepdim=True)
        scalars = torch.stack((floats / V2_OPTION_VALUE, (floats == highest).float(), (floats == lowest).float()), -1)
        index = torch.where(valid, number.clamp_max(OPTION_VALUES - 1), torch.full_like(number, OPTION_VALUES))
        extras = self.option_value(index) * valid.unsqueeze(-1) + self.option_scalars(scalars * valid.unsqueeze(-1))
        pointer = (source - 1).clamp_min(0)
        has_source = source > 0
        cards = batch["tok"][..., 1].long().gather(1, pointer)
        card = F.embedding(cards, table) * (has_source & (cards > 1)).unsqueeze(-1)
        extras = extras + self.option_source_card(card)
        if self.cfg.location_state:
            kinds = batch["tok"][..., 0].long().gather(1, pointer)
            packed = batch["tokf"][..., 4:8].gather(1, pointer.unsqueeze(-1).expand(-1, -1, 4))
            location = has_source & (kinds == 1)
            state = unpack_location_state(packed, location, packed.new_tensor(V2_LOCATION_STATE))
            extras = extras + self.option_source_location(state)
        return extras + self.option_reference_log(torch.log1p(overlap))

    def forward(self, b, return_hidden=False):
        if self.cfg.knowledge_mask == "opp_hidden":
            # Before anything reads the batch: the decoder memory mask also comes from b["tmask"].
            b = mask_opponent_hidden(b, self.hidden_cells, self.hidden_zones)
        output = super().forward(b, return_hidden)
        cap = self.cfg.logit_cap
        if not cap:
            return output
        logits, hidden = output if return_hidden else (output, None)
        valid = b["omask"]
        raw = logits.masked_fill(~valid, 0.0)
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            # Diagnostics only: shows whether the network still pushes past the cap.
            self.raw_max_abs_logit = raw.detach().abs().max()
        capped = torch.where(valid, cap * torch.tanh(raw / cap), logits)
        return (capped, hidden) if return_hidden else capped


class EfficientCritic(nn.Module):
    def __init__(self, cfg, tables):
        super().__init__()
        self.cfg = cfg
        self.encoder = EfficientPolicy(cfg, tables, actor=False)
        self.value = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, 1))
        if cfg.win_head:
            # Plan critic-v2: P(this seat wins) from the same Q' || G' summary (a logit); trained on the game result.
            self.win = nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, 1))

    def heads(self, batch):
        """(value, win logit or None)."""
        hidden, _ = self.encoder.encode(batch)
        summary = torch.cat((hidden[:, 0], hidden[:, 1]), -1)
        win = self.win(summary).squeeze(-1).float() if self.cfg.win_head else None
        return self.value(summary).squeeze(-1).float(), win

    def forward(self, batch):
        return self.heads(batch)[0]


def unpack_location_icons(packed, location, scale=60.0):
    """BL-20(a): unpack printed Light, printed Dark, current Light, current Dark icons (base 16), divided by scale."""
    value = torch.where(location, packed, torch.zeros_like(packed)).round().long()
    digits = torch.stack([(value // 16 ** index).remainder(16) for index in range(4)], -1).float()
    return digits / scale * location.unsqueeze(-1)


def unpack_location_state(packed, location, scale=60.0):
    """Amendment 2: power us/them, ability us/them (packed h(us) + 512 h(them), h = 2x), drain us/them; divided by
    scale (a number or a per-column tensor)."""
    value = torch.where(location.unsqueeze(-1), packed, torch.zeros_like(packed))
    halves = value[..., :2].round().long()
    us, them = halves.remainder(512).float() / 2, torch.div(halves, 512, rounding_mode="floor").float() / 2
    state = torch.stack((us[..., 0], them[..., 0], us[..., 1], them[..., 1], value[..., 2], value[..., 3]), -1)
    return state / scale * location.unsqueeze(-1)


def grid_features(raw, indicators=True, bound_scale=60.0):
    """Bounds divided by bound_scale; append asserted presence, exclusion and exactness at unit scale."""
    scaled = raw * raw.new_tensor([1 / bound_scale, 1 / bound_scale, 1.0] * 51)
    if not indicators:
        return scaled, scaled
    lower = raw[..., 0::3]
    upper = raw[..., 1::3]
    asserted = raw[..., 2::3]
    bits = torch.stack((lower > 0, upper == 0, lower == upper), dim=-1).to(raw.dtype)
    bits = bits * asserted.unsqueeze(-1)
    return scaled, torch.cat((scaled, bits.flatten(-2)), dim=-1)
