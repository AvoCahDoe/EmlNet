"""Shared training loop + metrics for moons and benchmark scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


@dataclass
class RunHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)


def bce_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        return (pred.eq(y).float().mean()).item()


def sanity_forward(model: nn.Module, x: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        out = model(x)
    ok = torch.isfinite(out).all().item()
    logger.info("Sanity forward all finite: %s | logits shape=%s", ok, tuple(out.shape))


def train_one(
    name: str,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_max_norm: float,
    log_every: int | None = None,
) -> RunHistory:
    hist = RunHistory()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    if log_every is None:
        log_every = max(1, epochs // 10)
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        total_norm = clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        opt.step()

        hist.grad_norm.append(float(total_norm))

        model.eval()
        with torch.no_grad():
            tl = loss_fn(model(x_train), y_train).item()
            vl = loss_fn(model(x_val), y_val).item()
            ta = bce_accuracy(model(x_train), y_train)
            va = bce_accuracy(model(x_val), y_val)
        model.train()

        hist.train_loss.append(tl)
        hist.val_loss.append(vl)
        hist.train_acc.append(ta)
        hist.val_acc.append(va)

        if ep == 0 or ep == epochs - 1 or (ep + 1) % log_every == 0:
            logger.info(
                "[%s] epoch %5d | train_loss=%.4f val_loss=%.4f | train_acc=%.3f val_acc=%.3f | grad_norm=%.4f",
                name,
                ep + 1,
                tl,
                vl,
                ta,
                va,
                hist.grad_norm[-1],
            )

    return hist
