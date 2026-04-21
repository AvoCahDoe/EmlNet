"""Stabilized EML forward with clamped outgoing gradients (custom autograd)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EMLFunctionConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    eps: float = 1e-6
    grad_x_clip: float = 1.0
    grad_y_clip: float = 1.0


class EMLFunction(torch.autograd.Function):
    """exp(clamp(x)) - log(|y| + eps) with masked clamp-grad and clipped partials."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, cfg: EMLFunctionConfig) -> torch.Tensor:
        xc = torch.clamp(x, cfg.x_min, cfg.x_max)
        ya = torch.abs(y) + cfg.eps
        ctx.cfg = cfg
        ctx.save_for_backward(x, y, xc, ya)
        return torch.exp(xc) - torch.log(ya)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        cfg: EMLFunctionConfig = ctx.cfg
        x, y, xc, ya = ctx.saved_tensors
        g = grad_output
        grad_xc = g * torch.exp(xc)
        in_range = ((x >= cfg.x_min) & (x <= cfg.x_max)).to(grad_xc.dtype)
        grad_x = torch.clamp(grad_xc * in_range, -cfg.grad_x_clip, cfg.grad_x_clip)
        grad_y = g * (-1.0 / ya) * torch.sign(y)
        grad_y = torch.clamp(grad_y, -cfg.grad_y_clip, cfg.grad_y_clip)
        return grad_x, grad_y, None
