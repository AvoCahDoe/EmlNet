"""Train 2-layer EML vs 2-layer MLP on make_moons; save loss / acc / grad / decision plots."""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from emlnet_pkg.eml_function import EMLFunctionConfig
from emlnet_pkg.init_utils import init_weights_small_
from emlnet_pkg.models import TwoLayerEMLClassifier, TwoLayerMLP, count_params
from emlnet_pkg.plotting import ensure_plots_dir, save_current_figure
from emlnet_pkg.training_core import RunHistory, sanity_forward, train_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _mesh_probs(model: nn.Module, lim: float = 2.0, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(-lim, lim, n, dtype=np.float32)
    gy = np.linspace(-lim, lim, n, dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    t = torch.tensor(grid)
    model.eval()
    with torch.no_grad():
        z = torch.sigmoid(model(t)).numpy().reshape(xx.shape)
    return xx, yy, z


def _plot_decision_pair(eml: nn.Module, mlp: nn.Module, x_np: np.ndarray, y_np: np.ndarray, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, model, title in zip(axes, (eml, mlp), ("EML (2-layer)", "MLP (2-layer)"), strict=True):
        xx, yy, z = _mesh_probs(model)
        ax.contourf(xx, yy, z, levels=28, cmap="RdBu_r", alpha=0.9)
        ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np, s=12, cmap="Spectral", edgecolors="k", linewidths=0.2)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.suptitle("Two moons: decision regions after training")
    fig.tight_layout()
    save_current_figure(path)


def _plot_losses(h_eml: RunHistory, h_mlp: RunHistory, path, logy: bool = False) -> None:
    epochs = np.arange(1, len(h_eml.train_loss) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.train_loss, label="EML train", linewidth=1.2)
    plt.plot(epochs, h_eml.val_loss, label="EML val", linewidth=1.2)
    plt.plot(epochs, h_mlp.train_loss, label="MLP train", linewidth=1.2)
    plt.plot(epochs, h_mlp.val_loss, label="MLP val", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training dynamics (make_moons)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    save_current_figure(path)


def _plot_accuracy(h_eml: RunHistory, h_mlp: RunHistory, path) -> None:
    epochs = np.arange(1, len(h_eml.train_acc) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.train_acc, label="EML train acc", linewidth=1.2)
    plt.plot(epochs, h_eml.val_acc, label="EML val acc", linewidth=1.2)
    plt.plot(epochs, h_mlp.train_acc, label="MLP train acc", linewidth=1.2)
    plt.plot(epochs, h_mlp.val_acc, label="MLP val acc", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("Classification accuracy (threshold 0.5 on sigmoid)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def _plot_grad_norms(h_eml: RunHistory, h_mlp: RunHistory, path) -> None:
    epochs = np.arange(1, len(h_eml.grad_norm) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, h_eml.grad_norm, label="EML grad norm (post clip)", linewidth=1.0)
    plt.plot(epochs, h_mlp.grad_norm, label="MLP grad norm (post clip)", linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Global grad L2 norm")
    plt.title("Gradient norms after clip_grad_norm_")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(path)


def _plot_summary_grid(h_eml: RunHistory, h_mlp: RunHistory, path) -> None:
    epochs = np.arange(1, len(h_eml.train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(epochs, h_eml.train_loss, label="EML tr")
    ax.plot(epochs, h_eml.val_loss, label="EML val")
    ax.plot(epochs, h_mlp.train_loss, label="MLP tr")
    ax.plot(epochs, h_mlp.val_loss, label="MLP val")
    ax.set_title("Loss (linear y)")
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

    fig.suptitle("EML vs MLP — summary panel")
    fig.tight_layout()
    save_current_figure(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EML vs MLP on make_moons and save plots.")
    p.add_argument("--epochs", type=int, default=15000, help="Long default for unattended CPU runs.")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--n-samples", type=int, default=800)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_np, y_np = make_moons(n_samples=args.n_samples, noise=0.1, random_state=0)
    x_tr_np, x_val_np, y_tr_np, y_val_np = train_test_split(
        x_np, y_np, test_size=0.2, random_state=args.seed, stratify=y_np
    )
    x_train = torch.tensor(x_tr_np, dtype=torch.float32)
    y_train = torch.tensor(y_tr_np, dtype=torch.float32).view(-1, 1)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

    cfg = EMLFunctionConfig()
    eml = TwoLayerEMLClassifier(in_features=2, hidden=args.hidden, cfg=cfg)
    mlp = TwoLayerMLP(in_features=2, hidden=args.hidden)

    init_weights_small_(eml, std=args.init_std, zero_bias=True)
    init_weights_small_(mlp, std=args.init_std, zero_bias=True)

    logger.info("EML parameters: %d", count_params(eml))
    logger.info("MLP parameters: %d", count_params(mlp))

    x0 = x_train[: min(32, x_train.shape[0])]
    sanity_forward(eml, x0)
    sanity_forward(mlp, x0)

    logger.info("Training EML for %d epochs...", args.epochs)
    h_eml = train_one(
        "EML",
        eml,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_max_norm=args.clip_max_norm,
    )

    init_weights_small_(mlp, std=args.init_std, zero_bias=True)
    logger.info("Training MLP for %d epochs...", args.epochs)
    h_mlp = train_one(
        "MLP",
        mlp,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_max_norm=args.clip_max_norm,
    )

    if args.no_plots:
        logger.info("--no-plots set; skipping figures.")
        return

    out = ensure_plots_dir()
    _plot_losses(h_eml, h_mlp, out / "moons_loss_linear.png", logy=False)
    _plot_losses(h_eml, h_mlp, out / "moons_loss_logy.png", logy=True)
    _plot_accuracy(h_eml, h_mlp, out / "moons_accuracy.png")
    _plot_grad_norms(h_eml, h_mlp, out / "moons_grad_norm.png")
    _plot_decision_pair(eml, mlp, x_np, y_np, out / "moons_decision_eml_mlp.png")
    _plot_summary_grid(h_eml, h_mlp, out / "moons_summary_grid.png")

    for name in (
        "moons_loss_linear.png",
        "moons_loss_logy.png",
        "moons_accuracy.png",
        "moons_grad_norm.png",
        "moons_decision_eml_mlp.png",
        "moons_summary_grid.png",
    ):
        logger.info("Wrote %s", out / name)


if __name__ == "__main__":
    main()
