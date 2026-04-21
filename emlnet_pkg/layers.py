"""Learnable EML layer: two linear projections into (x, y) then custom EML."""

from __future__ import annotations

import torch
from torch import nn

from emlnet_pkg.eml_function import EMLFunction, EMLFunctionConfig


class EMLLayer(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, cfg: EMLFunctionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else EMLFunctionConfig()
        self.lin_x = nn.Linear(in_features, hidden_features, bias=True)
        self.lin_y = nn.Linear(in_features, hidden_features, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.lin_x(inp)
        y = self.lin_y(inp)
        return EMLFunction.apply(x, y, self.cfg)
