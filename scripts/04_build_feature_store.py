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
from cs2_trading_rebuild.features import build_feature_store
from cs2_trading_rebuild.utils import load_table, save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_table(resolve_path(cfg, "market_panel_master"))
    feature_path = resolve_path(cfg, "feature_store")
    market_path = resolve_path(cfg, "market_series")

    feature_df, market_series = build_feature_store(
        panel=panel,
        sell_cost_rate=float(cfg["research"]["sell_cost_rate"]),
        alpha_horizons=[int(x) for x in cfg["research"]["alpha_horizons_days"]],
    )
    save_table(feature_df, feature_path)
    save_table(market_series, market_path)

    reports_dir = resolve_path(cfg, "reports_dir")
    write_json(reports_dir / "04_feature_store_report.json", {
        "n_feature_rows": int(len(feature_df)),
        "n_market_rows": int(len(market_series)),
        "date_min": str(feature_df["date"].min()),
        "date_max": str(feature_df["date"].max()),
    })
    print(f"Saved feature store to {feature_path}")


if __name__ == "__main__":
    main()