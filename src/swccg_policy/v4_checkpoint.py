"""Self-contained ABI-4 checkpoints with explicit prefix-extension loading."""

import copy
import os
from pathlib import Path

import numpy as np
import torch

from .efficient import EfficientConfig, EfficientPolicy, EfficientCritic
from .soft_vocabulary import SoftVocabulary
from .table_bundle import TableBundle
from .v4_tables import CardTable


def pack_tables(tables):
    return {"cards": tables.cards.cards, "card_parent": tables.cards.parent_sha,
            "card_ancestors": tables.cards.ancestors, "vocabulary": tables.vocabulary.payload,
            "priors": tables.vocabulary.priors, "parent_sha": tables.parent_sha,
            "ancestors": tables.ancestors, "sha": tables.sha}


def unpack_tables(payload):
    cards = CardTable(payload["cards"], payload["card_parent"], payload["card_ancestors"])
    vocabulary = SoftVocabulary(payload["vocabulary"], payload["priors"])
    tables = TableBundle(cards, vocabulary, payload["parent_sha"], payload["ancestors"])
    if tables.sha != payload["sha"]:
        raise ValueError("Checkpoint table SHA mismatch")
    return tables


def new_models(tables, seed=904000, layers=6, critic_layers=6, **ablations):
    torch.manual_seed(seed)
    cards = tables.cards.cards
    numeric = [index for index, name in enumerate(cards["feature_names"])
               if str(name).startswith(("num:", "has:"))]
    cfg = EfficientConfig(len(tables.cards.ids), cards["features"].shape[1], 768,
                          len(tables.vocabulary.payload["q"]), len(tables.vocabulary.payload["o"]),
                          len(tables.vocabulary.payload["context"]), numeric, layers=layers,
                          template_identity_seed=seed, **ablations)
    critic_cfg = copy.deepcopy(cfg)
    critic_cfg.layers = critic_layers
    critic_cfg.knowledge_mask = "none"  # D16: the critic keeps its privileged view; it never acts.
    return EfficientPolicy(cfg, tables), EfficientCritic(critic_cfg, tables)


def save(path, actor, critic, tables, metadata=None, optimizers=None, extra=None):
    """Plan q-critic: `extra` adds top-level entries (e.g. qnet_state, qnet_optimizer); load() ignores them."""
    if actor.table_sha != tables.sha or critic.encoder.table_sha != tables.sha:
        raise ValueError("Checkpoint model/table identity mismatch")
    state = {"cfg": actor.cfg.to_dict(), "critic_cfg": critic.cfg.to_dict(), "tables": pack_tables(tables),
             "vocab": tables.vocabulary.payload, "table_sha": tables.sha,
             "state": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
             "value_state": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
             "rl_meta": metadata or {}}
    if optimizers is not None:
        state["optimizer_state"] = [optimizer.state_dict() for optimizer in optimizers]
    if extra:
        if set(extra) & set(state):
            raise ValueError(f"Checkpoint extra entries clash with {sorted(set(extra) & set(state))}")
        state.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def verify_buffers(model, tables):
    expected = {"cards.features": tables.cards.cards["features"], "cards.text": tables.cards.cards["text_emb"],
                "qtmpl.priors": tables.vocabulary.priors["q"],
                "context.priors": tables.vocabulary.priors["context"]}
    if model.actor:
        expected["otmpl.priors"] = tables.vocabulary.priors["o"]
    state = model.state_dict()
    for name, array in expected.items():
        if not np.array_equal(state[name].detach().cpu().numpy(), array):
            raise ValueError(f"Checkpoint buffer disagrees with owned tables: {name}")


def load(path, device="cpu", tables=None, payload=None):
    checkpoint = payload if payload is not None else torch.load(path, map_location="cpu", weights_only=False)
    owned = unpack_tables(checkpoint["tables"])
    if checkpoint["table_sha"] != owned.sha or checkpoint["vocab"] != owned.vocabulary.payload:
        raise ValueError("Checkpoint table/vocabulary identity mismatch")
    if tables is not None:
        owned.assert_prefix_of(tables)
    # Checkpoints predating Amendment 5 used only the original 153 grid inputs.
    legacy = {"grid_indicators": False, "post_degree": False, "embedding_init_std": 1.0,
              "template_prior_scale": 1.0, "template_identity_std": 0.0, "template_identity_seed": 0,
              "component_norm": False,
              "component_label_gain": 1.0, "component_norm_scope": "all"}
    actor_cfg = {**legacy, **checkpoint["cfg"]}
    critic_cfg = {**legacy, **checkpoint["critic_cfg"]}
    for cfg in (actor_cfg, critic_cfg):
        if cfg.pop("per_scope_tokens", False):
            raise ValueError("The withdrawn probe-only per-scope architecture is not a production checkpoint")
    actor = EfficientPolicy(EfficientConfig(**actor_cfg), owned)
    critic = EfficientCritic(EfficientConfig(**critic_cfg), owned)
    actor.load_state_dict(checkpoint["state"])
    critic.load_state_dict(checkpoint["value_state"])
    verify_buffers(actor, owned)
    verify_buffers(critic.encoder, owned)
    if tables is not None and tables.sha != owned.sha:
        actor.grow(owned, tables)
        critic.encoder.grow(owned, tables)
    return actor.to(device).eval(), critic.to(device).eval(), tables or owned, checkpoint
