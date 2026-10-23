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
from cs2_trading_rebuild.policy import POLICY_PANEL_COLS, build_policy_panel_from_alpha_predictions, calibrate_policy
from cs2_trading_rebuild.utils import save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    models_dir = resolve_path(cfg, "models_dir")
    alpha_pred_path = models_dir / "alpha_predictions.parquet"

    # Load only the columns needed for calibration/live/test.
    alpha_pred_df = pd.read_parquet(alpha_pred_path, columns=POLICY_PANEL_COLS)

    panel = build_policy_panel_from_alpha_predictions(alpha_pred_df)

    signals_dir = resolve_path(cfg, "signals_dir")
    signals_dir.mkdir(parents=True, exist_ok=True)
    save_table(panel, signals_dir / "policy_panel.parquet")

    result = calibrate_policy(panel, cfg)

    save_table(result.search_table, models_dir / "policy_grid_search.parquet")
    write_json(models_dir / "policy.json", result.best_policy)
    write_json(models_dir / "policy_validation_summary.json", result.validation_summary)

    print(f"Saved calibrated policy to {models_dir / 'policy.json'}")


if __name__ == "__main__":
    main()