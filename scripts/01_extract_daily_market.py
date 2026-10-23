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
from cs2_trading_rebuild.data_prep import export_daily_market_from_db, validate_daily_market
from cs2_trading_rebuild.utils import save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = resolve_path(cfg, "raw_daily_market")
    report_dir = resolve_path(cfg, "reports_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    daily = export_daily_market_from_db(cfg)
    save_table(daily, out_path)
    write_json(report_dir / "01_daily_market_report.json", validate_daily_market(daily))
    print(f"Saved daily market to {out_path}")


if __name__ == "__main__":
    main()