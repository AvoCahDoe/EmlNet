"""Copy sweep outputs (plots/sweep) into site/ for static hosting."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bundle plots/sweep into site/sweep + site/data.")
    p.add_argument("--sweep-root", type=str, default="", help="Path to sweep folder (default: EMLnet/plots/sweep)")
    p.add_argument("--site-root", type=str, default="", help="Path to site/ (default: sibling of plots/)")
    p.add_argument("--clean", action="store_true", help="Remove existing site/sweep before copy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent.parent
    sweep = Path(args.sweep_root) if args.sweep_root else here / "plots" / "sweep"
    site = Path(args.site_root) if args.site_root else here / "site"
    if not sweep.is_dir():
        raise SystemExit(f"Sweep folder not found: {sweep}")
    man = sweep / "manifest.json"
    if not man.is_file():
        raise SystemExit(f"Missing manifest.json under {sweep}. Run train_sweep first.")

    dest_sweep = site / "sweep"
    if args.clean and dest_sweep.exists():
        shutil.rmtree(dest_sweep)
    dest_sweep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sweep, dest_sweep, dirs_exist_ok=True)

    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(man, data_dir / "sweep_manifest.json")
    summary = sweep / "summary_all.json"
    if summary.is_file():
        shutil.copy2(summary, data_dir / "sweep_summary_all.json")

    meta = {
        "sweep_manifest_url": "data/sweep_manifest.json",
        "sweep_summary_url": "data/sweep_summary_all.json",
        "asset_base": "sweep",
    }
    (data_dir / "site_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Copied {sweep} -> {dest_sweep}")
    print(f"Wrote {data_dir / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
