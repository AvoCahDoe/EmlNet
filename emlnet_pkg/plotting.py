"""Figure output under EMLnet/plots (resolved from package location)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plots_dir() -> Path:
    """`EMLnet/plots` (one level above the `emlnet_pkg` package)."""
    return Path(__file__).resolve().parents[1] / "plots"


def ensure_plots_dir() -> Path:
    p = plots_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
