"""Grid sweep: scenarios × depths × hidden widths → plots/sweep/ + manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from emlnet_pkg.benchmark_reporting import plot_heatmap_gap, save_run_figure_bundle
from emlnet_pkg.datasets_scenarios import SCENARIOS, load_scenario
from emlnet_pkg.eml_function import EMLFunctionConfig
from emlnet_pkg.init_utils import init_weights_small_
from emlnet_pkg.models import DeepEMLClassifier, DeepMLPClassifier, count_params
from emlnet_pkg.plotting import ensure_plots_dir
from emlnet_pkg.training_core import sanity_forward, train_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep EML vs MLP over scenarios × depth × hidden.")
    p.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help=f"Comma-separated keys or 'all'. Keys: {', '.join(sorted(SCENARIOS))}",
    )
    p.add_argument("--depths", type=str, default="2,3,4", help="Comma-separated depths, e.g. 2,3,4")
    p.add_argument("--hiddens", type=str, default="32,56,96", help="Comma-separated hidden widths")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--init-std", type=float, default=0.008)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override sweep root under plots/ (default: plots/sweep)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.scenarios.strip().lower() == "all":
        keys = sorted(SCENARIOS.keys())
    else:
        keys = [k.strip() for k in args.scenarios.split(",") if k.strip()]

    depths = _parse_int_list(args.depths)
    hiddens = _parse_int_list(args.hiddens)

    plots = ensure_plots_dir()
    root = Path(args.out_root) if args.out_root else plots / "sweep"
    scen_root = root / "scenarios"
    scen_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []
    heatmap_entries: list[dict[str, str]] = []

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

        gap_matrix = np.full((len(depths), len(hiddens)), np.nan, dtype=np.float64)

        for di, depth in enumerate(depths):
            for hi, hidden in enumerate(hiddens):
                cfg = EMLFunctionConfig()
                eml = DeepEMLClassifier(in_features=2, hidden=hidden, n_eml=depth, cfg=cfg)
                mlp = DeepMLPClassifier(in_features=2, hidden=hidden, n_hidden=depth)

                init_weights_small_(eml, std=args.init_std, zero_bias=True)
                init_weights_small_(mlp, std=args.init_std, zero_bias=True)

                tag = f"d{depth}_h{hidden}"
                logger.info("=== %s | %s | lim=%.3f ===", bundle.name, tag, bundle.lim)
                logger.info(
                    "EML params: %d | MLP params: %d",
                    count_params(eml),
                    count_params(mlp),
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

                gap = float(h_mlp.val_loss[-1] - h_eml.val_loss[-1])
                gap_matrix[di, hi] = gap

                row = {
                    "scenario": bundle.name,
                    "eml_params": count_params(eml),
                    "mlp_params": count_params(mlp),
                    "depth": depth,
                    "hidden": hidden,
                    "epochs": args.epochs,
                    "eml_final_val_loss": h_eml.val_loss[-1],
                    "mlp_final_val_loss": h_mlp.val_loss[-1],
                    "eml_final_val_acc": h_eml.val_acc[-1],
                    "mlp_final_val_acc": h_mlp.val_acc[-1],
                    "val_loss_gap_mlp_minus_eml": gap,
                }
                all_rows.append(row)

                rel_dir = f"scenarios/{bundle.name}/{tag}"
                run_dir = root / rel_dir
                if not args.no_plots:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    title = f"{bundle.name} ({tag})"
                    save_run_figure_bundle(
                        run_dir,
                        title,
                        eml,
                        mlp,
                        h_eml,
                        h_mlp,
                        x_np,
                        y_np,
                        bundle.lim,
                        prefix="",
                    )

                manifest_runs.append(
                    {
                        **row,
                        "rel_dir": rel_dir.replace("\\", "/"),
                        "figures": {
                            "loss_linear": f"{rel_dir}/loss_linear.png".replace("\\", "/"),
                            "loss_logy": f"{rel_dir}/loss_logy.png".replace("\\", "/"),
                            "val_loss_gap": f"{rel_dir}/val_loss_gap.png".replace("\\", "/"),
                            "val_delta_eml_minus_mlp": f"{rel_dir}/val_delta_eml_minus_mlp.png".replace("\\", "/"),
                            "accuracy": f"{rel_dir}/accuracy.png".replace("\\", "/"),
                            "grad_norm": f"{rel_dir}/grad_norm.png".replace("\\", "/"),
                            "decision": f"{rel_dir}/decision.png".replace("\\", "/"),
                            "summary_grid": f"{rel_dir}/summary_grid.png".replace("\\", "/"),
                            "final_bars": f"{rel_dir}/final_bars.png".replace("\\", "/"),
                            "histories": f"{rel_dir}/histories.json".replace("\\", "/"),
                        },
                    }
                )

        hm_name = "heatmap_val_loss_gap_mlp_minus_eml.png"
        hm_rel = f"scenarios/{bundle.name}/{hm_name}"
        hm_path = root / hm_rel
        if not args.no_plots:
            plot_heatmap_gap(
                depths,
                hiddens,
                gap_matrix,
                hm_path,
                f"{bundle.name} — val loss gap (MLP − EML)",
            )
        heatmap_entries.append({"scenario": bundle.name, "path": hm_rel.replace("\\", "/")})

    manifest = {
        "depths": depths,
        "hiddens": hiddens,
        "scenarios": keys,
        "epochs_default": args.epochs,
        "runs": manifest_runs,
        "heatmaps": heatmap_entries,
    }
    man_path = root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path = root / "summary_all.json"
    summary_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    csv_path = root / "summary_all.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    logger.info("Wrote manifest %s, summary %s, %s", man_path, summary_path, csv_path)


if __name__ == "__main__":
    main()
