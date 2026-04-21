"""Multi-scenario EML vs MLP benchmarks (harder 2D sets, deeper stacks)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from emlnet_pkg.datasets_scenarios import SCENARIOS, load_scenario
from emlnet_pkg.eml_function import EMLFunctionConfig
from emlnet_pkg.init_utils import init_weights_small_
from emlnet_pkg.models import DeepEMLClassifier, DeepMLPClassifier, count_params
from emlnet_pkg.plotting import ensure_plots_dir, save_current_figure
from emlnet_pkg.training_core import RunHistory, sanity_forward, train_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _mesh_probs(model: nn.Module, lim: float, n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(-lim, lim, n, dtype=np.float32)
    gy = np.linspace(-lim, lim, n, dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    t = torch.tensor(grid)
    model.eval()
    with torch.no_grad():
        z = torch.sigmoid(model(t)).numpy().reshape(xx.shape)
    return xx, yy, z


def _plot_losses(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str, logy: bool) -> None:
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


def _plot_val_gap(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
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


def _plot_accuracy(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
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


def _plot_grad_norms(h_eml: RunHistory, h_mlp: RunHistory, path: Path, title: str) -> None:
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


def _plot_decision_pair(
    eml: nn.Module,
    mlp: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lim: float,
    path: Path,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, model, title in zip(axes, (eml, mlp), ("EML", "MLP"), strict=True):
        xx, yy, z = _mesh_probs(model, lim=lim)
        ax.contourf(xx, yy, z, levels=28, cmap="RdBu_r", alpha=0.92)
        ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np, s=10, cmap="Spectral", edgecolors="k", linewidths=0.15)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle(suptitle)
    fig.tight_layout()
    save_current_figure(path)


def _plot_summary_grid(h_eml: RunHistory, h_mlp: RunHistory, path: Path, suptitle: str) -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EML vs MLP on harder 2D scenarios (deeper nets supported).")
    p.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help=f"Comma-separated keys or 'all'. Keys: {', '.join(sorted(SCENARIOS))}",
    )
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--hidden", type=int, default=56)
    p.add_argument("--depth", type=int, default=3, help="EML blocks and MLP (Linear,ReLU) depth (matched).")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--init-std", type=float, default=0.008)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.scenarios.strip().lower() == "all":
        keys = sorted(SCENARIOS.keys())
    else:
        keys = [k.strip() for k in args.scenarios.split(",") if k.strip()]

    base = ensure_plots_dir() / "benchmarks"
    base.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for key in keys:
        if key not in SCENARIOS:
            raise SystemExit(f"Unknown scenario {key!r}. Pick from: {sorted(SCENARIOS)}")
        bundle = load_scenario(key)
        x_np, y_np = bundle.x, bundle.y
        strat = y_np if len(np.unique(y_np)) > 1 else None
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_np, y_np, test_size=args.test_size, random_state=args.seed, stratify=strat
        )
        x_train = torch.tensor(x_tr, dtype=torch.float32)
        y_train = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
        x_val = torch.tensor(x_va, dtype=torch.float32)
        y_val = torch.tensor(y_va, dtype=torch.float32).view(-1, 1)

        cfg = EMLFunctionConfig()
        eml = DeepEMLClassifier(in_features=2, hidden=args.hidden, n_eml=args.depth, cfg=cfg)
        mlp = DeepMLPClassifier(in_features=2, hidden=args.hidden, n_hidden=args.depth)

        init_weights_small_(eml, std=args.init_std, zero_bias=True)
        init_weights_small_(mlp, std=args.init_std, zero_bias=True)

        logger.info("=== Scenario %s | lim=%.3f ===", bundle.name, bundle.lim)
        logger.info("EML params: %d | MLP params: %d | depth=%d hidden=%d", count_params(eml), count_params(mlp), args.depth, args.hidden)

        x0 = x_train[: min(64, x_train.shape[0])]
        sanity_forward(eml, x0)
        sanity_forward(mlp, x0)

        log_every = max(1, args.epochs // 8)
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
            log_every=log_every,
        )
        init_weights_small_(mlp, std=args.init_std, zero_bias=True)
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
            log_every=log_every,
        )

        row = {
            "scenario": bundle.name,
            "eml_params": count_params(eml),
            "mlp_params": count_params(mlp),
            "depth": args.depth,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "eml_final_val_loss": h_eml.val_loss[-1],
            "mlp_final_val_loss": h_mlp.val_loss[-1],
            "eml_final_val_acc": h_eml.val_acc[-1],
            "mlp_final_val_acc": h_mlp.val_acc[-1],
            "val_loss_gap_mlp_minus_eml": h_mlp.val_loss[-1] - h_eml.val_loss[-1],
        }
        summary_rows.append(row)

        if args.no_plots:
            continue

        out = base / bundle.name
        out.mkdir(parents=True, exist_ok=True)
        tag = f"d{args.depth}_h{args.hidden}"
        title = f"{bundle.name} ({tag})"
        _plot_losses(h_eml, h_mlp, out / f"{tag}_loss_linear.png", title, logy=False)
        _plot_losses(h_eml, h_mlp, out / f"{tag}_loss_logy.png", title, logy=True)
        _plot_val_gap(h_eml, h_mlp, out / f"{tag}_val_loss_gap.png", title)
        _plot_accuracy(h_eml, h_mlp, out / f"{tag}_accuracy.png", title)
        _plot_grad_norms(h_eml, h_mlp, out / f"{tag}_grad_norm.png", title)
        _plot_decision_pair(eml, mlp, x_np, y_np, bundle.lim, out / f"{tag}_decision.png", title)
        _plot_summary_grid(h_eml, h_mlp, out / f"{tag}_summary_grid.png", title)
        logger.info("Wrote figures under %s", out)

    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    csv_path = base / "summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("Wrote %s and %s", summary_path, csv_path)
    else:
        logger.info("Wrote %s (no rows for CSV)", summary_path)


if __name__ == "__main__":
    main()
