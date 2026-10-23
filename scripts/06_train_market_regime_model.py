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
from cs2_trading_rebuild.models import (
    REGIME_FEATURES,
    build_tradeable_market_series,
    save_train_result,
    train_regime_model,
)
from cs2_trading_rebuild.utils import save_table, write_json


def _required_feature_cols() -> list[str]:
    return [
        "date",
        "market_hash_name",
        "price_rub",
        "ret_1d",
        "ret_7d",
        "ret_30d",
        "vol_30d",
        "turnover_rub",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

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

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")

    feature_df = pd.read_parquet(
        feature_path,
        columns=_required_feature_cols(),
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    universe_df = pd.read_parquet(
        universe_path,
        columns=["date", "market_hash_name", "tradeable_today"],
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )

    market_df = build_tradeable_market_series(feature_df, universe_df, cfg)
    result = train_regime_model(market_df, cfg)

    models_dir = resolve_path(cfg, "models_dir")
    save_train_result(models_dir, "regime_model", result)
    save_table(market_df, models_dir / "tradable_market_series.parquet")
    save_table(result.predictions, models_dir / "regime_predictions.parquet")
    write_json(models_dir / "regime_report.json", result.train_report)

    print(f"Saved regime model to {models_dir}")


if __name__ == "__main__":
    main()
