"""Shared BC/probe optimization; smoothing is a probe-only confidence control."""

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    lr: float = 3e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    delta_weight_decay: float | None = None
    clip_norm: float = 1.0
    precision: str = "fp32"
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.optimizer not in ("adam", "adamw") or self.precision not in ("fp32", "bf16"):
            raise ValueError("Unsupported optimizer or training precision")
        if self.lr <= 0 or self.clip_norm != 1.0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Training requires positive LR, clipping at 1.0 and valid label smoothing")

    def to_dict(self):
        return asdict(self)


class TrainingStep:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        parameters = list(model.parameters())
        if config.delta_weight_decay is not None:
            parameters = [
                {"params": [p for name, p in model.named_parameters() if name != "cards.delta.weight"]},
                {"params": [model.cards.delta.weight], "weight_decay": config.delta_weight_decay},
            ]
        optimizer = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
        self.optimizer = optimizer(parameters, lr=config.lr, betas=config.betas,
                                   eps=config.eps, weight_decay=config.weight_decay)

    def __call__(self, batch, loss_call):
        device_type = next(self.model.parameters()).device.type
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=self.config.precision == "bf16"):
            logits = self.model(batch)
        loss = loss_call(logits, self.config.label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        valid = logits[batch["omask"]] if isinstance(batch, dict) and "omask" in batch else logits
        return {"loss": loss.detach(), "logits": logits.detach(), "grad_norm": float(norm),
                "max_abs_logit": float(valid.detach().abs().max())}


def probe_loss(labels):
    return lambda logits, smoothing: F.cross_entropy(logits.float(), labels, label_smoothing=smoothing)


def probe_config(lr=3e-4):
    return TrainingConfig(lr=lr, label_smoothing=0.05)
