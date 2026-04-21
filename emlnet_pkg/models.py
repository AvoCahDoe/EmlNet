"""Two-layer EML classifier vs two-layer ReLU MLP (moons-sized inputs)."""

from __future__ import annotations

import torch.nn as nn

from emlnet_pkg.eml_function import EMLFunctionConfig
from emlnet_pkg.layers import EMLLayer


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TwoLayerEMLClassifier(nn.Module):
    """EMLLayer -> EMLLayer -> linear logits."""

    def __init__(self, in_features: int = 2, hidden: int = 32, cfg: EMLFunctionConfig | None = None) -> None:
        super().__init__()
        self.eml1 = EMLLayer(in_features, hidden, cfg)
        self.eml2 = EMLLayer(hidden, hidden, cfg)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.eml1(x)
        z = self.eml2(z)
        return self.head(z)


class TwoLayerMLP(nn.Module):
    """Linear -> ReLU -> Linear -> logits."""

    def __init__(self, in_features: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepEMLClassifier(nn.Module):
    """Stack `n_eml` EMLLayers (first maps in->hidden, rest hidden->hidden) + logits."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_eml: int,
        cfg: EMLFunctionConfig | None = None,
    ) -> None:
        super().__init__()
        if n_eml < 1:
            raise ValueError("n_eml must be >= 1")
        cfg = cfg if cfg is not None else EMLFunctionConfig()
        blocks: list[nn.Module] = [EMLLayer(in_features, hidden, cfg)]
        for _ in range(1, n_eml):
            blocks.append(EMLLayer(hidden, hidden, cfg))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(x))


class DeepMLPClassifier(nn.Module):
    """`(Linear -> ReLU)` repeated `n_hidden` times, then linear to logit."""

    def __init__(self, in_features: int, hidden: int, n_hidden: int) -> None:
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        layers: list[nn.Module] = []
        layers += [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(1, n_hidden):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
