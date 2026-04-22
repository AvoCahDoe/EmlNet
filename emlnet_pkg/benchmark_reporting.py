"""Figures + JSON export for benchmark and sweep runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn as nn

from emlnet_pkg.plotting import save_current_figure
from emlnet_pkg.training_core import RunHistory


def mesh_probs(model: nn.Module, lim: float, n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(-lim, lim, n, dtype=np.float32)
    gy = np.linspace(-lim, lim, n, dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    t = torch.tensor(grid)
    model.eval()
    with torch.no_grad():
        z = torch.sigmoid(model(t)).numpy().reshape(xx.shape)
    return xx, yy, z


def plot_losses(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str, logy: bool) -> None:
    epochs = np.arange(1, len(h_eml.train_loss) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.train_loss, label="EML train", linewidth=1.1)
    plt.plot(epochs, h_eml.val_loss, label="EML val", linewidth=1.1)
    plt.plot(epochs, h_mlp.train_loss, label="MLP train", linewidth=1.1)
    plt.plot(epochs, h_mlp.val_loss, label="MLP val", linewidth=1.1)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    save_current_figure(path)


def plot_val_gap(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
    epochs = np.arange(1, len(h_eml.val_loss) + 1)
    gap = np.array(h_mlp.val_loss) - np.array(h_eml.val_loss)
    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs, gap, color="tab:purple", linewidth=1.2)
    plt.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MLP val BCE − EML val BCE")
    plt.title(title + " (positive ⇒ EML lower val loss)")
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def plot_val_loss_delta_eml_minus_mlp(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
    epochs = np.arange(1, len(h_eml.val_loss) + 1)
    delta = np.array(h_eml.val_loss) - np.array(h_mlp.val_loss)
    plt.figure(figsize=(9, 4.5))
    plt.fill_between(epochs, delta, 0.0, where=(delta >= 0), alpha=0.35, color="tab:green", label="EML better (lower val loss)")
    plt.fill_between(epochs, delta, 0.0, where=(delta < 0), alpha=0.35, color="tab:red", label="MLP better")
    plt.plot(epochs, delta, color="tab:blue", linewidth=1.0)
    plt.axhline(0.0, color="k", linewidth=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("EML val BCE − MLP val BCE")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def plot_accuracy(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
    epochs = np.arange(1, len(h_eml.train_acc) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.train_acc, label="EML train", linewidth=1.1)
    plt.plot(epochs, h_eml.val_acc, label="EML val", linewidth=1.1)
    plt.plot(epochs, h_mlp.train_acc, label="MLP train", linewidth=1.1)
    plt.plot(epochs, h_mlp.val_acc, label="MLP val", linewidth=1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def plot_grad_norms(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
    epochs = np.arange(1, len(h_eml.grad_norm) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.grad_norm, label="EML (post clip)", linewidth=1.0)
    plt.plot(epochs, h_mlp.grad_norm, label="MLP (post clip)", linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Global grad L2 norm")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def plot_decision_pair(
    eml: nn.Module,
    mlp: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lim: float,
    path: Path,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, model, t in zip(axes, (eml, mlp), ("EML", "MLP"), strict=True):
        xx, yy, z = mesh_probs(model, lim=lim)
        ax.contourf(xx, yy, z, levels=28, cmap="RdBu_r", alpha=0.92)
        ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np, s=10, cmap="Spectral", edgecolors="k", linewidths=0.15)
        ax.set_title(t)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle(suptitle)
    fig.tight_layout()
    save_current_figure(path)


def plot_summary_grid(h_eml: RunHistory, h_mlp: RunHistory, path: Path, suptitle: str) -> None:
    epochs = np.arange(1, len(h_eml.train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(epochs, h_eml.train_loss, label="EML tr")
    ax.plot(epochs, h_eml.val_loss, label="EML val")
    ax.plot(epochs, h_mlp.train_loss, label="MLP tr")
    ax.plot(epochs, h_mlp.val_loss, label="MLP val")
    ax.set_title("Loss (linear)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, h_eml.train_loss, label="EML tr")
    ax.plot(epochs, h_eml.val_loss, label="EML val")
    ax.plot(epochs, h_mlp.train_loss, label="MLP tr")
    ax.plot(epochs, h_mlp.val_loss, label="MLP val")
    ax.set_yscale("log")
    ax.set_title("Loss (log y)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, h_eml.train_acc, label="EML tr")
    ax.plot(epochs, h_eml.val_acc, label="EML val")
    ax.plot(epochs, h_mlp.train_acc, label="MLP tr")
    ax.plot(epochs, h_mlp.val_acc, label="MLP val")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acc")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, h_eml.grad_norm, label="EML")
    ax.plot(epochs, h_mlp.grad_norm, label="MLP")
    ax.set_title("Grad norm (post clip)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle)
    fig.tight_layout()
    save_current_figure(path)


def plot_final_val_bars(
    eml_val_loss: float,
    mlp_val_loss: float,
    eml_val_acc: float,
    mlp_val_acc: float,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(["EML", "MLP"], [eml_val_loss, mlp_val_loss], color=["#4dabf7", "#ffa94d"])
    axes[0].set_ylabel("Final val BCE")
    axes[0].set_title(title + " — val loss")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(["EML", "MLP"], [eml_val_acc, mlp_val_acc], color=["#4dabf7", "#ffa94d"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Final val acc")
    axes[1].set_title(title + " — val acc")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_current_figure(path)


def save_histories_json(path: Path, h_eml: RunHistory, h_mlp: RunHistory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"eml": h_eml.to_dict(), "mlp": h_mlp.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_figure_bundle(
    out_dir: Path,
    title: str,
    eml: nn.Module,
    mlp: nn.Module,
    h_eml: RunHistory,
    h_mlp: RunHistory,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lim: float,
    *,
    prefix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = prefix
    plot_losses(h_eml, h_mlp, out_dir / f"{p}loss_linear.png", title, logy=False)
    plot_losses(h_eml, h_mlp, out_dir / f"{p}loss_logy.png", title, logy=True)
    plot_val_gap(h_eml, h_mlp, out_dir / f"{p}val_loss_gap.png", title)
    plot_val_loss_delta_eml_minus_mlp(h_eml, h_mlp, out_dir / f"{p}val_delta_eml_minus_mlp.png", title)
    plot_accuracy(h_eml, h_mlp, out_dir / f"{p}accuracy.png", title)
    plot_grad_norms(h_eml, h_mlp, out_dir / f"{p}grad_norm.png", title)
    plot_decision_pair(eml, mlp, x_np, y_np, lim, out_dir / f"{p}decision.png", title)
    plot_summary_grid(h_eml, h_mlp, out_dir / f"{p}summary_grid.png", title)
    plot_final_val_bars(
        h_eml.val_loss[-1],
        h_mlp.val_loss[-1],
        h_eml.val_acc[-1],
        h_mlp.val_acc[-1],
        out_dir / f"{p}final_bars.png",
        title,
    )
    save_histories_json(out_dir / f"{p}histories.json", h_eml, h_mlp)


def plot_heatmap_gap(
    depths: list[int],
    hiddens: list[int],
    matrix: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """matrix[i,j] = gap at depth[i], hidden[j] (rows = depth, cols = hidden).

    Uses a diverging norm around 0. Matplotlib's default ``Normalize`` breaks when
    ``vmin == vmax`` (e.g. a 1×1 sweep), which yields a blank / white heatmap cell.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if not np.any(np.isfinite(mat)):
        mat = np.zeros((len(depths), len(hiddens)), dtype=np.float64)

    v = float(np.nanmax(np.abs(mat)))
    if not np.isfinite(v) or v < 1e-12:
        v = 1e-6
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    fw = max(6.0, len(hiddens) * 1.15 + 2.5)
    fh = max(4.0, len(depths) * 0.85 + 2.0)
    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.imshow(
        mat,
        aspect="equal",
        cmap="RdBu_r",
        origin="lower",
        norm=norm,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MLP val BCE − EML val BCE")
    ax.set_xticks(np.arange(len(hiddens)))
    ax.set_xticklabels([str(h) for h in hiddens])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel("hidden width")
    ax.set_ylabel("depth")
    ax.set_title(title)

    ncells = max(1, mat.shape[0] * mat.shape[1])
    fs = max(5, min(11, 200 // ncells))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = "nan" if not np.isfinite(val) else f"{val:.3f}"
            t = ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white")
            t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    fig.tight_layout()
    save_current_figure(path)
