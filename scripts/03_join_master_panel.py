#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.data_prep import join_master_panel, build_item_coverage_summary
from cs2_trading_rebuild.utils import load_table, save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    daily = load_table(resolve_path(cfg, "raw_daily_market"))
    master = load_table(resolve_path(cfg, "raw_item_master"))

    panel, join_report = join_master_panel(daily, master)
    panel_path = resolve_path(cfg, "market_panel_master")
    save_table(panel, panel_path)

    coverage = build_item_coverage_summary(panel)
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_table(coverage, reports_dir / "03_item_coverage_summary.parquet")
    write_json(reports_dir / "03_join_report.json", join_report)
    print(f"Saved master panel to {panel_path}")


if __name__ == "__main__":
    main()