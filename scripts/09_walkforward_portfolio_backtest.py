#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.backtest_portfolio import BACKTEST_PANEL_COLS, run_portfolio_backtest
from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.utils import save_table, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    signals_path = resolve_path(cfg, "signals_dir") / "policy_panel.parquet"

    panel = pd.read_parquet(signals_path, columns=BACKTEST_PANEL_COLS)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    splits = cfg["splits"]
    test_mask = (
        (panel["date"] >= pd.Timestamp(splits["test_start"]))
        & (panel["date"] <= pd.Timestamp(splits["test_end"]))
    )
    test_panel = panel.loc[test_mask].copy().reset_index(drop=True)

    policy_path = resolve_path(cfg, "models_dir") / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    bt = run_portfolio_backtest(
        panel_df=test_panel,
        policy=policy,
        initial_capital_rub=float(cfg["research"]["initial_capital_rub"]),
        sell_cost_rate=float(cfg["research"]["sell_cost_rate"]),
        execute_next_day=bool(cfg["research"]["execute_next_day"]),
        store_details=True,
    )

    backtests_dir = resolve_path(cfg, "backtests_dir")
    backtests_dir.mkdir(parents=True, exist_ok=True)

    save_table(bt.equity_curve, backtests_dir / "portfolio_backtest_test_equity.parquet")
    save_table(bt.trades, backtests_dir / "portfolio_backtest_test_trades.parquet")
    save_table(bt.daily_holdings, backtests_dir / "portfolio_backtest_test_holdings.parquet")
    write_json(backtests_dir / "portfolio_backtest_test_summary.json", bt.summary)

    print(f"Saved test portfolio backtest to {backtests_dir}")


if __name__ == "__main__":
    main()