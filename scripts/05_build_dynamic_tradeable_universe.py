#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.universe import (
    REQUIRED_UNIVERSE_INPUT_COLS,
    build_dynamic_tradeable_universe,
)
from cs2_trading_rebuild.utils import save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    feature_path = resolve_path(cfg, "feature_store")
    out_path = resolve_path(cfg, "universe_daily")

    # Read only the columns needed by step 05.
    feature_df = pd.read_parquet(feature_path, columns=REQUIRED_UNIVERSE_INPUT_COLS)

    universe_df = build_dynamic_tradeable_universe(feature_df, cfg)
    save_table(universe_df, out_path)

    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "05_universe_report.json", {
        "n_rows": int(len(universe_df)),
        "tradeable_share": float(universe_df["tradeable_today"].mean()) if len(universe_df) else 0.0,
        "avg_tradeable_items_per_day": float(
            universe_df.groupby("date", sort=False)["tradeable_today"].sum().mean()
        ) if len(universe_df) else 0.0,
    })
    print(f"Saved daily universe to {out_path}")


if __name__ == "__main__":
    main()