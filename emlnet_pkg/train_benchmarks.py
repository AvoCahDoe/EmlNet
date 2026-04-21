"""Multi-scenario EML vs MLP benchmarks (harder 2D sets, deeper stacks)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from emlnet_pkg.benchmark_reporting import save_run_figure_bundle
from emlnet_pkg.datasets_scenarios import SCENARIOS, load_scenario
from emlnet_pkg.eml_function import EMLFunctionConfig
from emlnet_pkg.init_utils import init_weights_small_
from emlnet_pkg.models import DeepEMLClassifier, DeepMLPClassifier, count_params
from emlnet_pkg.plotting import ensure_plots_dir
from emlnet_pkg.training_core import sanity_forward, train_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        logger.info(
            "EML params: %d | MLP params: %d | depth=%d hidden=%d",
            count_params(eml),
            count_params(mlp),
            args.depth,
            args.hidden,
        )

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
        save_run_figure_bundle(
            out,
            title,
            eml,
            mlp,
            h_eml,
            h_mlp,
            x_np,
            y_np,
            bundle.lim,
            prefix=f"{tag}_",
        )
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
