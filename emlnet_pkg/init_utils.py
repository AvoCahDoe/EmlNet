"""Conservative weight initialization for EML-style networks."""

from __future__ import annotations

import torch.nn as nn


def init_weights_small_(module: nn.Module, std: float = 0.01, zero_bias: bool = True) -> None:
    """Initialize Linear weights from a tight Gaussian; biases zero or small."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                if zero_bias:
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.normal_(m.bias, mean=0.0, std=std)
