#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import gc
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.models import (
    CATEGORICAL_FEATURES,
    FEATURE_NUMERIC_COLS,
    save_train_result,
    train_alpha_model,
)
from cs2_trading_rebuild.utils import save_table, write_json


def _downcast_in_place(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _resolve_alpha_target_spec(cfg: dict) -> tuple[str, str]:
    target_col = str(cfg["research"]["primary_target"])
    h = int(cfg["research"]["primary_alpha_horizon_days"])
    if "net" in target_col:
        base_future_col = f"future_net_return_{h}d"
    else:
        base_future_col = f"future_gross_return_{h}d"
    return target_col, base_future_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_col, base_future_col = _resolve_alpha_target_spec(cfg)

    split_min = min(
        pd.Timestamp(cfg["splits"]["train_start"]),
        pd.Timestamp(cfg["splits"]["val_start"]),
        pd.Timestamp(cfg["splits"]["test_start"]),
    )
    split_max = max(
        pd.Timestamp(cfg["splits"]["train_end"]),
        pd.Timestamp(cfg["splits"]["val_end"]),
        pd.Timestamp(cfg["splits"]["test_end"]),
    )

    # IMPORTANT FIX:
    # base_future_col must be included explicitly for target reconstruction,
    # and the full column list must be deduplicated.
    feature_cols = ["date", "market_hash_name", target_col, base_future_col] + FEATURE_NUMERIC_COLS + CATEGORICAL_FEATURES
    feature_cols = list(dict.fromkeys(feature_cols))

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    regime_path = resolve_path(cfg, "models_dir") / "regime_predictions.parquet"

    universe_df = pd.read_parquet(
        universe_path,
        columns=["date", "market_hash_name", "tradeable_today", "tradability_score"],
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    universe_df = _downcast_in_place(universe_df)

    feature_df = pd.read_parquet(
        feature_path,
        columns=feature_cols,
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    # Safety: parquet reads can still surface duplicate columns if the source file
    # already contains them. Keep first occurrence only.
    if feature_df.columns.has_duplicates:
        feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()].copy()
    feature_df = _downcast_in_place(feature_df)

    regime_pred_df = pd.read_parquet(
        regime_path,
        columns=["date", "regime_prob"],
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    regime_pred_df = _downcast_in_place(regime_pred_df)

    result = train_alpha_model(feature_df, universe_df, regime_pred_df, cfg)

    del feature_df
    del universe_df
    del regime_pred_df
    gc.collect()

    models_dir = resolve_path(cfg, "models_dir")
    save_train_result(models_dir, "alpha_model", result)
    save_table(result.predictions, models_dir / "alpha_predictions.parquet")
    write_json(models_dir / "alpha_report.json", result.train_report)

    print(f"Saved alpha model to {models_dir}")


if __name__ == "__main__":
    main()
