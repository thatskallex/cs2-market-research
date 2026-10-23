#!/usr/bin/env python
"""
Alpha-rotation friction sweep for the CS2 trading rebuild.

Purpose
-------
This is a research script, not a production strategy script. It follows the
zero-cost proof in scripts/18_zero_cost_signal_profit_proof.py and asks the
next empirical-finance question:

    "At what level of market friction does the model's alpha-rotation edge
     disappear?"

It reruns the same long-only alpha-rotation strategies across a grid of sell
costs and compares them with passive baselines at the same cost. The output is
intended to support research-style tables and figures: cost sensitivity,
break-even costs, turnover, drawdown, and split-by-period robustness.

Cost model
----------
The default friction is a sell-side proportional cost, matching the existing
repo's Steam/market-fee style accounting. For a sale worth P * units, proceeds
are:

    proceeds = P * units * (1 - sell_cost_rate)

Buys are not charged by default. Final equity is reported in two forms:

    final_equity_rub
        Cash plus marked value of remaining holdings without forced liquidation.
        This answers: "What is the portfolio worth on paper?"

    final_equity_liquidated_rub
        Cash plus remaining holdings after the same sell cost is applied.
        This answers: "What cash would we have if we fully exited?"

The main ROI and break-even tables use liquidated equity by default, because it
is the most conservative comparison across turnover profiles.

Outputs
-------
  data/processed/reports/19_alpha_rotation_friction_sweep_summary.json
  data/processed/reports/19_strategy_cost_grid.csv
  data/processed/reports/19_strategy_cost_grid.parquet
  data/processed/reports/19_break_even_table.csv
  data/processed/reports/19_break_even_table.parquet
  data/processed/reports/19_split_robustness_table.csv
  data/processed/reports/19_research_tables.md
  data/processed/reports/19_fig_*.png                  # if matplotlib exists
  backtests/friction_<cost>_<strategy>_equity.parquet  # optional details
  backtests/friction_<cost>_<strategy>_trades.parquet  # optional details

Run
---
  python scripts/19_alpha_rotation_friction_sweep.py --config configs/cs2_trading_rebuilt.yaml

Useful variants
---------------
  # Finer cost grid around likely break-even regions
  python scripts/19_alpha_rotation_friction_sweep.py --config configs/cs2_trading_rebuilt.yaml \
    --costs 0 0.01 0.02 0.03 0.05 0.07 0.10 0.12 0.15 0.20

  # Focus on the more defensible low-turnover candidates
  python scripts/19_alpha_rotation_friction_sweep.py --config configs/cs2_trading_rebuilt.yaml \
    --rebalance-days 90 180 365 --top-pcts 0.10 0.20 0.30
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.utils import save_table, write_json


DEFAULT_REBALANCE_DAYS = [30, 90, 180, 365]
DEFAULT_TOP_PCTS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_COSTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]


META_COLS = ["quality", "stattrak", "souvenir", "type", "rarity", "metatype"]
OPTIONAL_ALPHA_DIAGNOSTIC_COLS = [
    "future_excess_gross_return_30d",
    "future_gross_return_30d",
    "future_net_return_30d",
    "alpha_rank_score_raw",
    "alpha_prob_top",
    "alpha_prob_bottom",
]


def _available_parquet_columns(path: Path) -> set[str]:
    """Return parquet column names without loading the full file."""
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        # Fallback: pandas can read an empty slice for many parquet engines.
        try:
            return set(pd.read_parquet(path).head(0).columns)
        except Exception:
            return set()


def _read_parquet_existing(path: Path, wanted: list[str], **kwargs) -> pd.DataFrame:
    """Read only columns that exist, preserving requested order."""
    available = _available_parquet_columns(path)
    cols = [c for c in wanted if c in available]
    if not cols:
        raise KeyError(f"None of the requested columns exist in {path}: {wanted}")
    return pd.read_parquet(path, columns=cols, **kwargs)


def _safe_name(name: str) -> str:
    name = name.lower().replace("%", "pct").replace(".", "_")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _cost_label(cost: float) -> str:
    return f"cost_{int(round(float(cost) * 10000)):04d}bp"


def _roi_pct(initial_capital: float, final_equity: float | None) -> float | None:
    if final_equity is None or initial_capital <= 0:
        return None
    return float((float(final_equity) / initial_capital - 1.0) * 100.0)


def _sharpe(equity: pd.Series, annualization_days: int = 365) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    r = eq.pct_change().dropna()
    if len(r) < 2:
        return None
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(math.sqrt(annualization_days) * float(r.mean()) / sd)


def _max_drawdown_pct(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) == 0:
        return None
    running_max = eq.cummax().replace(0, np.nan)
    dd = (eq - running_max) / running_max
    if not dd.notna().any():
        return None
    return float(dd.min() * 100.0)


def _per_split_roi(equity_df: pd.DataFrame, splits: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split_name, (start_key, end_key) in [
        ("train", ("train_start", "train_end")),
        ("val", ("val_start", "val_end")),
        ("test", ("test_start", "test_end")),
    ]:
        s = pd.Timestamp(splits[start_key]).normalize()
        e = pd.Timestamp(splits[end_key]).normalize()
        seg = equity_df[(equity_df["date"] >= s) & (equity_df["date"] <= e)]
        if len(seg) < 2:
            out[split_name] = {"n_days": int(len(seg))}
            continue
        start_eq = float(seg["liquidated_equity_rub"].iloc[0])
        end_eq = float(seg["liquidated_equity_rub"].iloc[-1])
        out[split_name] = {
            "n_days": int(len(seg)),
            "start_equity_rub": start_eq,
            "end_equity_rub": end_eq,
            "roi_pct": _roi_pct(start_eq, end_eq),
        }
    return out


def _load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    alpha_path = resolve_path(cfg, "models_dir") / "alpha_predictions.parquet"
    if not alpha_path.exists():
        raise FileNotFoundError(f"alpha_predictions.parquet not found at {alpha_path}")

    print("Loading universe...")
    universe_cols = ["date", "market_hash_name", "tradeable_today"] + META_COLS
    universe = _read_parquet_existing(universe_path, universe_cols)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["tradeable_today"] = universe["tradeable_today"].fillna(False).astype(bool)
    universe = universe.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    print("Loading prices...")
    prices = pd.read_parquet(feature_path, columns=["date", "market_hash_name", "price_rub"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["price_rub"] = pd.to_numeric(prices["price_rub"], errors="coerce").astype("float32")
    prices = prices.dropna(subset=["price_rub"])
    prices = prices.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    print("Loading alpha predictions...")
    alpha_cols = ["date", "market_hash_name", "alpha_pred"] + META_COLS + OPTIONAL_ALPHA_DIAGNOSTIC_COLS
    alpha = _read_parquet_existing(alpha_path, alpha_cols)
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha["alpha_pred"] = pd.to_numeric(alpha["alpha_pred"], errors="coerce").astype("float32")
    for c in OPTIONAL_ALPHA_DIAGNOSTIC_COLS:
        if c in alpha.columns:
            alpha[c] = pd.to_numeric(alpha[c], errors="coerce").astype("float32")
    alpha = alpha.dropna(subset=["alpha_pred"])
    alpha = alpha.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    price_wide = prices.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    price_wide_mtm = price_wide.ffill()
    print(
        f"  price grid: {price_wide.index.min().date()} to {price_wide.index.max().date()}, "
        f"{len(price_wide.columns):,} items"
    )
    print(f"  universe rows: {len(universe):,}; alpha rows: {len(alpha):,}")
    return universe, alpha, price_wide, price_wide_mtm

def _make_sim_dates(price_wide: pd.DataFrame, universe: pd.DataFrame, splits: dict[str, Any]) -> pd.DatetimeIndex:
    universe_dates = set(universe.loc[universe["tradeable_today"], "date"].unique())
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & universe_dates))
    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    if len(sim_dates) == 0:
        raise ValueError("No simulation dates after intersecting price, universe, and split ranges.")
    return sim_dates


def _build_daily_tradeable_scores(
    universe: pd.DataFrame,
    alpha: pd.DataFrame,
    sim_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.DataFrame]:
    df = universe.loc[universe["tradeable_today"], ["date", "market_hash_name"]].merge(
        alpha[["date", "market_hash_name", "alpha_pred"]],
        on=["date", "market_hash_name"],
        how="left",
    )
    df = df[df["date"].isin(sim_dates)].copy()
    df["alpha_pred"] = pd.to_numeric(df["alpha_pred"], errors="coerce")
    return {pd.Timestamp(d): g[["market_hash_name", "alpha_pred"]].copy() for d, g in df.groupby("date", sort=True)}


def _target_all(g: pd.DataFrame) -> list[str]:
    return g["market_hash_name"].dropna().astype(str).tolist()


def _target_top_pct(g: pd.DataFrame, top_pct: float, min_names: int = 1) -> list[str]:
    scored = g.dropna(subset=["alpha_pred"]).copy()
    if len(scored) == 0:
        return []
    scored = scored.sort_values(["alpha_pred", "market_hash_name"], ascending=[False, True])
    k = max(int(math.ceil(len(scored) * float(top_pct))), int(min_names))
    return scored.head(k)["market_hash_name"].astype(str).tolist()


def _next_exec_date(sim_dates: pd.DatetimeIndex, current_idx: int, execute_next_day: bool) -> pd.Timestamp | None:
    if execute_next_day:
        if current_idx + 1 >= len(sim_dates):
            return None
        return pd.Timestamp(sim_dates[current_idx + 1])
    return pd.Timestamp(sim_dates[current_idx])


def simulate_rebalance_with_costs(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    daily_scores: dict[pd.Timestamp, pd.DataFrame],
    sim_dates: pd.DatetimeIndex,
    initial_capital: float,
    rebalance_days: int,
    selection_mode: str,
    sell_cost_rate: float,
    top_pct: float | None = None,
    execute_next_day: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebalance simulator with proportional sell costs.

    At each scheduled signal date, choose a target set from that date's tradeable
    universe. Execution occurs on the next simulation date by default. The
    portfolio is adjusted toward equal target capital weights. When sell_cost_rate
    > 0, trimming winners is costly; this script intentionally measures how much
    that hurts the alpha-rotation edge.
    """
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    pending_targets: dict[pd.Timestamp, list[str]] = {}
    last_signal_date: pd.Timestamp | None = None

    equity_rows: list[dict[str, Any]] = []
    trades_rows: list[dict[str, Any]] = []
    total_cost_paid = 0.0
    gross_sell_turnover = 0.0
    gross_buy_turnover = 0.0
    rebalance_count = 0

    def mark_values(d: pd.Timestamp) -> tuple[float, float]:
        if not positions:
            return 0.0, 0.0
        held = [it for it in positions if it in price_cols]
        if not held:
            return 0.0, 0.0
        row = price_wide_mtm.loc[d, held]
        gross_mv = 0.0
        for it in held:
            px = row[it]
            if pd.notna(px) and px > 0:
                gross_mv += float(positions[it]) * float(px)
        liquidated_mv = gross_mv * (1.0 - sell_cost_rate)
        return gross_mv, liquidated_mv

    def execute_target(d: pd.Timestamp, target_items: list[str]) -> None:
        nonlocal cash, positions, total_cost_paid, gross_sell_turnover, gross_buy_turnover, rebalance_count
        candidate_items = list(set(target_items) | set(positions.keys()))
        candidate_items = [it for it in candidate_items if it in price_cols]
        if not candidate_items:
            return

        px_row_exec = price_wide.loc[d, [it for it in candidate_items if it in price_wide.columns]]
        px_row_mtm = price_wide_mtm.loc[d, candidate_items]

        def px_for_sell(it: str) -> float | None:
            px = px_row_mtm[it]
            if pd.notna(px) and px > 0:
                return float(px)
            return None

        def px_for_buy(it: str) -> float | None:
            if it not in px_row_exec.index:
                return None
            px = px_row_exec[it]
            if pd.notna(px) and px > 0:
                return float(px)
            return None

        current_mv_gross = 0.0
        for it, units in positions.items():
            px = px_for_sell(it)
            if px is not None:
                current_mv_gross += int(units) * px
        equity_gross = cash + current_mv_gross
        if equity_gross <= 0:
            return

        buyable_targets = [it for it in target_items if it in price_cols and px_for_buy(it) is not None]
        buyable_targets = sorted(set(buyable_targets))
        if not buyable_targets:
            return

        target_budget = equity_gross / len(buyable_targets)
        desired_units: dict[str, int] = {}
        for it in buyable_targets:
            px = px_for_buy(it)
            if px is None:
                continue
            desired_units[it] = int(target_budget // px)

        traded_this_rebalance = False

        # Sells first: exits or trims.
        for it in list(positions.keys()):
            old_units = int(positions[it])
            new_units = int(desired_units.get(it, 0))
            if new_units >= old_units:
                continue
            units_to_sell = old_units - new_units
            px = px_for_sell(it)
            if px is None:
                continue
            gross_value = units_to_sell * px
            cost_paid = gross_value * sell_cost_rate
            proceeds = gross_value - cost_paid
            cash += proceeds
            total_cost_paid += cost_paid
            gross_sell_turnover += gross_value
            traded_this_rebalance = True
            if new_units > 0:
                positions[it] = new_units
            else:
                del positions[it]
            trades_rows.append({
                "date": d,
                "side": "sell",
                "market_hash_name": it,
                "price_rub": px,
                "units": int(units_to_sell),
                "gross_value_rub": float(gross_value),
                "net_value_rub": float(proceeds),
                "cost_paid_rub": float(cost_paid),
                "reason": "friction_rebalance_trim_or_exit",
            })

        # Buys second: enter or top up. No buy-side fee by default.
        for it, desired in desired_units.items():
            old_units = int(positions.get(it, 0))
            if desired <= old_units:
                continue
            units_to_buy = desired - old_units
            px = px_for_buy(it)
            if px is None:
                continue
            affordable = int(cash // px)
            units_to_buy = min(units_to_buy, affordable)
            if units_to_buy <= 0:
                continue
            cost = units_to_buy * px
            cash -= cost
            positions[it] = old_units + units_to_buy
            gross_buy_turnover += cost
            traded_this_rebalance = True
            trades_rows.append({
                "date": d,
                "side": "buy",
                "market_hash_name": it,
                "price_rub": px,
                "units": int(units_to_buy),
                "gross_value_rub": float(cost),
                "net_value_rub": float(cost),
                "cost_paid_rub": 0.0,
                "reason": "friction_rebalance_buy_or_topup",
            })

        if traded_this_rebalance:
            rebalance_count += 1

    for i, d0 in enumerate(sim_dates):
        d = pd.Timestamp(d0)

        for target_items in pending_targets.pop(d, []):
            execute_target(d, target_items)

        gross_mv, liquidated_mv = mark_values(d)
        equity_rows.append({
            "date": d,
            "cash_rub": float(cash),
            "market_value_rub": float(gross_mv),
            "liquidated_market_value_rub": float(liquidated_mv),
            "equity_rub": float(cash + gross_mv),
            "liquidated_equity_rub": float(cash + liquidated_mv),
            "n_positions": int(len(positions)),
        })

        due = last_signal_date is None or (rebalance_days > 0 and (d - last_signal_date).days >= rebalance_days)
        if not due:
            continue
        scores_today = daily_scores.get(d)
        if scores_today is None or len(scores_today) == 0:
            continue
        if selection_mode == "all":
            target = _target_all(scores_today)
        elif selection_mode == "top_pct":
            if top_pct is None:
                raise ValueError("top_pct must be supplied for selection_mode='top_pct'")
            target = _target_top_pct(scores_today, top_pct=top_pct, min_names=1)
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")
        if not target:
            continue
        exec_date = _next_exec_date(sim_dates, i, execute_next_day)
        if exec_date is None:
            continue
        pending_targets.setdefault(exec_date, []).append(target)
        last_signal_date = d

    summary = {
        "total_cost_paid_rub": float(total_cost_paid),
        "gross_sell_turnover_rub": float(gross_sell_turnover),
        "gross_buy_turnover_rub": float(gross_buy_turnover),
        "gross_total_turnover_rub": float(gross_sell_turnover + gross_buy_turnover),
        "rebalance_count_with_trades": int(rebalance_count),
    }
    return pd.DataFrame(equity_rows), pd.DataFrame(trades_rows), summary


def _strategy_summary(
    strategy_name: str,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    splits: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(equity_df) == 0:
        final_equity = initial_capital
        final_liq = initial_capital
        avg_pos = 0.0
        max_pos = 0
    else:
        final_equity = float(equity_df["equity_rub"].iloc[-1])
        final_liq = float(equity_df["liquidated_equity_rub"].iloc[-1])
        avg_pos = float(equity_df["n_positions"].mean())
        max_pos = int(equity_df["n_positions"].max())

    total_cost = float(trades_df["cost_paid_rub"].sum()) if len(trades_df) and "cost_paid_rub" in trades_df.columns else 0.0
    sell_turnover = float(trades_df.loc[trades_df["side"] == "sell", "gross_value_rub"].sum()) if len(trades_df) else 0.0
    buy_turnover = float(trades_df.loc[trades_df["side"] == "buy", "gross_value_rub"].sum()) if len(trades_df) else 0.0

    summary: dict[str, Any] = {
        "strategy": strategy_name,
        "initial_capital_rub": float(initial_capital),
        "final_equity_rub": float(final_equity),
        "final_equity_liquidated_rub": float(final_liq),
        "roi_pct": _roi_pct(initial_capital, final_liq),
        "roi_marked_pct": _roi_pct(initial_capital, final_equity),
        "sharpe": _sharpe(equity_df["liquidated_equity_rub"]) if len(equity_df) else None,
        "max_drawdown_pct": _max_drawdown_pct(equity_df["liquidated_equity_rub"]) if len(equity_df) else None,
        "n_days": int(len(equity_df)),
        "n_trades": int(len(trades_df)),
        "n_buys": int((trades_df["side"] == "buy").sum()) if len(trades_df) else 0,
        "n_sells": int((trades_df["side"] == "sell").sum()) if len(trades_df) else 0,
        "avg_n_positions": avg_pos,
        "max_n_positions": max_pos,
        "total_cost_paid_rub": total_cost,
        "gross_sell_turnover_rub": sell_turnover,
        "gross_buy_turnover_rub": buy_turnover,
        "gross_total_turnover_rub": sell_turnover + buy_turnover,
        "turnover_to_initial_capital": (sell_turnover + buy_turnover) / initial_capital if initial_capital > 0 else None,
        "sell_turnover_to_initial_capital": sell_turnover / initial_capital if initial_capital > 0 else None,
        "by_split": _per_split_roi(equity_df, splits),
    }
    if extra:
        summary.update(extra)
    return summary


def _interpolate_crossing(costs: list[float], values: list[float], target: float = 0.0) -> float | None:
    """Return the first x where values cross from >= target to < target.

    Linear interpolation is used between grid points. Returns None when the
    crossing is outside the supplied grid.
    """
    pairs = sorted((float(c), float(v)) for c, v in zip(costs, values) if pd.notna(c) and pd.notna(v))
    if len(pairs) == 0:
        return None
    if pairs[0][1] < target:
        return pairs[0][0]
    for (c0, v0), (c1, v1) in zip(pairs[:-1], pairs[1:]):
        if v0 >= target and v1 < target:
            if v1 == v0:
                return c1
            frac = (target - v0) / (v1 - v0)
            return float(c0 + frac * (c1 - c0))
    return None


def _annualized_return_pct(roi_pct: Any, n_days: Any, annualization_days: int = 365) -> float | None:
    """Convert total-period ROI into annualized return/CAGR-like percentage."""
    try:
        roi = float(roi_pct)
        days = float(n_days)
    except Exception:
        return None
    if not np.isfinite(roi) or not np.isfinite(days) or days <= 0:
        return None
    wealth = 1.0 + roi / 100.0
    if wealth <= 0:
        return None
    return float((wealth ** (annualization_days / days) - 1.0) * 100.0)


def _build_tables(grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Passive benchmark at each cost: best passive liquidated ROI at that exact cost.
    grid_df = grid_df.copy()
    grid_df["annualized_roi_pct"] = [
        _annualized_return_pct(r, n) for r, n in zip(grid_df["roi_pct"], grid_df["n_days"])
    ]

    passive = grid_df[grid_df["strategy_type"].isin(["passive", "passive_rebalanced"])].copy()
    best_passive_by_cost = (
        passive.sort_values(["sell_cost_rate", "roi_pct"], ascending=[True, False])
        .groupby("sell_cost_rate", as_index=False)
        .first()[["sell_cost_rate", "strategy", "roi_pct", "annualized_roi_pct"]]
        .rename(columns={
            "strategy": "best_passive_strategy",
            "roi_pct": "best_passive_roi_pct",
            "annualized_roi_pct": "best_passive_annualized_roi_pct",
        })
    )
    grid = grid_df.merge(best_passive_by_cost, on="sell_cost_rate", how="left")
    grid["excess_roi_vs_best_passive_pct"] = grid["roi_pct"] - grid["best_passive_roi_pct"]
    grid["annualized_excess_roi_vs_best_passive_pct"] = (
        grid["annualized_roi_pct"] - grid["best_passive_annualized_roi_pct"]
    )

    break_rows = []
    for strategy, g in grid.groupby("strategy", sort=False):
        g = g.sort_values("sell_cost_rate")
        costs = g["sell_cost_rate"].astype(float).tolist()
        rois = g["roi_pct"].astype(float).tolist()
        excess = g["excess_roi_vs_best_passive_pct"].astype(float).tolist()
        annualized = g["annualized_roi_pct"].astype(float).tolist()
        annualized_excess = g["annualized_excess_roi_vs_best_passive_pct"].astype(float).tolist()
        zero_profit_cost = _interpolate_crossing(costs, rois, target=0.0)
        passive_break_even_cost = _interpolate_crossing(costs, excess, target=0.0)
        annualized_zero_profit_cost = _interpolate_crossing(costs, annualized, target=0.0)
        annualized_passive_break_even_cost = _interpolate_crossing(costs, annualized_excess, target=0.0)
        row0 = g.iloc[0]
        row_max_cost = g.iloc[-1]
        break_rows.append({
            "strategy": strategy,
            "strategy_type": row0.get("strategy_type"),
            "selection_mode": row0.get("selection_mode"),
            "top_pct": row0.get("top_pct"),
            "rebalance_days": row0.get("rebalance_days"),
            "roi_0pct_cost": row0.get("roi_pct"),
            "annualized_roi_0pct_cost": row0.get("annualized_roi_pct"),
            "roi_max_tested_cost": row_max_cost.get("roi_pct"),
            "annualized_roi_max_tested_cost": row_max_cost.get("annualized_roi_pct"),
            "excess_roi_0pct_vs_passive": row0.get("excess_roi_vs_best_passive_pct"),
            "annualized_excess_roi_0pct_vs_passive": row0.get("annualized_excess_roi_vs_best_passive_pct"),
            "excess_roi_max_tested_vs_passive": row_max_cost.get("excess_roi_vs_best_passive_pct"),
            "annualized_excess_roi_max_tested_vs_passive": row_max_cost.get("annualized_excess_roi_vs_best_passive_pct"),
            "zero_profit_break_even_cost": zero_profit_cost,
            "passive_relative_break_even_cost": passive_break_even_cost,
            "annualized_zero_profit_break_even_cost": annualized_zero_profit_cost,
            "annualized_passive_relative_break_even_cost": annualized_passive_break_even_cost,
            "avg_trades_across_costs": float(g["n_trades"].mean()),
            "avg_sell_turnover_to_initial": float(g["sell_turnover_to_initial_capital"].mean()),
            "avg_positions_at_0pct": row0.get("avg_n_positions"),
            "max_drawdown_0pct": row0.get("max_drawdown_pct"),
            "sharpe_0pct": row0.get("sharpe"),
        })
    break_even = pd.DataFrame(break_rows).sort_values(
        ["strategy_type", "passive_relative_break_even_cost", "roi_0pct_cost"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    split_rows = []
    for _, row in grid.iterrows():
        by_split = row.get("by_split")
        if not isinstance(by_split, dict):
            continue
        for split_name in ["train", "val", "test"]:
            split = by_split.get(split_name, {})
            split_rows.append({
                "strategy": row["strategy"],
                "strategy_type": row["strategy_type"],
                "sell_cost_rate": row["sell_cost_rate"],
                "split": split_name,
                "split_roi_pct": split.get("roi_pct"),
                "split_annualized_roi_pct": _annualized_return_pct(split.get("roi_pct"), split.get("n_days")),
                "n_days": split.get("n_days"),
            })
    split_table = pd.DataFrame(split_rows)
    return grid, break_even, split_table

def _fmt_pct(x: Any, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.{decimals}f}%"


def _fmt_cost(x: Any) -> str:
    if x is None or pd.isna(x):
        return "> max grid / not crossed"
    return f"{float(x) * 100:.2f}%"


def _write_research_tables_md(path: Path, grid: pd.DataFrame, break_even: pd.DataFrame) -> None:
    # Compact academic-style tables for copy/paste into reports.
    best0 = grid[grid["sell_cost_rate"] == grid["sell_cost_rate"].min()].copy()
    best0 = best0.sort_values("roi_pct", ascending=False).head(12)
    cols_perf = [
        "strategy", "strategy_type", "top_pct", "rebalance_days", "roi_pct",
        "sharpe", "max_drawdown_pct", "n_trades", "avg_n_positions",
        "excess_roi_vs_best_passive_pct",
    ]
    perf = best0[cols_perf].copy()

    be_cols = [
        "strategy", "strategy_type", "top_pct", "rebalance_days", "roi_0pct_cost",
        "passive_relative_break_even_cost", "zero_profit_break_even_cost",
        "avg_sell_turnover_to_initial", "avg_trades_across_costs",
    ]
    be = break_even[break_even["strategy_type"] == "alpha_rotation"].head(16)[be_cols].copy()

    def to_md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    text = []
    text.append("# Research Tables — Alpha Rotation Friction Sweep\n")
    text.append("## Table 1. Zero-cost performance ranking\n")
    text.append(to_md(perf))
    text.append("\n\n## Table 2. Break-even transaction-cost analysis\n")
    text.append(to_md(be))
    text.append("\n")
    path.write_text("\n".join(text), encoding="utf-8")


def _human_strategy_name(strategy: str) -> str:
    """Readable names for legends/tables."""
    if strategy == "buy_and_hold_all":
        return "Passive buy-and-hold"
    if strategy.startswith("all_universe_rebalance_"):
        days = strategy.split("_")[-1].replace("d", "")
        return f"Passive full-universe, rebalance {days}d"
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d", str(strategy))
    if m:
        top, days = m.groups()
        return f"Alpha top {top}%, rebalance {days}d"
    return str(strategy).replace("_", " ").title()



def _build_alpha_metadata_panel(universe: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    """Daily tradeable universe joined to model scores and metadata for research plots."""
    uni_cols = ["date", "market_hash_name", "tradeable_today"] + [c for c in META_COLS if c in universe.columns]
    a_cols = ["date", "market_hash_name", "alpha_pred"] + [c for c in META_COLS if c in alpha.columns] + [
        c for c in OPTIONAL_ALPHA_DIAGNOSTIC_COLS if c in alpha.columns
    ]
    panel = universe.loc[universe["tradeable_today"], uni_cols].merge(
        alpha.loc[:, list(dict.fromkeys(a_cols))],
        on=["date", "market_hash_name"],
        how="inner",
        suffixes=("_universe", "_alpha"),
    )
    # Consolidate metadata from alpha first, then universe.
    for c in META_COLS:
        a = f"{c}_alpha"
        u = f"{c}_universe"
        if a in panel.columns and u in panel.columns:
            panel[c] = panel[a].where(panel[a].notna(), panel[u])
        elif a in panel.columns:
            panel[c] = panel[a]
        elif u in panel.columns:
            panel[c] = panel[u]
        if c in panel.columns:
            panel[c] = panel[c].astype("string").fillna("Unknown")
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["alpha_pred"] = pd.to_numeric(panel["alpha_pred"], errors="coerce")
    panel = panel.dropna(subset=["alpha_pred"])
    return panel


def _assign_daily_alpha_deciles(df: pd.DataFrame, score_col: str = "alpha_pred", n: int = 10) -> pd.Series:
    def _one_day(s: pd.Series) -> pd.Series:
        ranks = pd.to_numeric(s, errors="coerce").rank(method="first")
        try:
            return pd.qcut(ranks, n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)
    return df.groupby("date", group_keys=False)[score_col].transform(_one_day)


def _split_label(dates: pd.Series, splits: dict[str, Any]) -> pd.Series:
    d = pd.to_datetime(dates).dt.normalize()
    out = pd.Series("other", index=dates.index, dtype="object")
    for name, s_key, e_key in [
        ("train", "train_start", "train_end"),
        ("validation", "val_start", "val_end"),
        ("test", "test_start", "test_end"),
    ]:
        out[(d >= pd.Timestamp(splits[s_key])) & (d <= pd.Timestamp(splits[e_key]))] = name
    return out


def _target_diagnostic_col(alpha_panel: pd.DataFrame) -> str | None:
    for c in ["future_excess_gross_return_30d", "future_gross_return_30d", "future_net_return_30d"]:
        if c in alpha_panel.columns and pd.to_numeric(alpha_panel[c], errors="coerce").notna().any():
            return c
    return None


def _shorten_category_label(x: Any, max_len: int = 26) -> str:
    s = "Unknown" if pd.isna(x) else str(x)
    s = s.replace("Classified", "Cls.").replace("Covert", "Covert")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _make_figures(
    reports_dir: Path,
    grid: pd.DataFrame,
    break_even: pd.DataFrame,
    alpha: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    splits: dict[str, Any] | None = None,
) -> list[str]:
    """Create research-friendly grouped figures.

    v3 plotting philosophy:
      - Avoid the unreadable "all curves in one legend" chart as the main exhibit.
      - Use small multiples so each figure answers one research question.
      - Highlight the practical candidate family: top 10% alpha baskets across holding/rebalance horizons.
      - Keep passive benchmark visible in every panel.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipping figures ({exc})")
        return []

    fig_paths: list[str] = []

    def savefig(name: str) -> None:
        out = reports_dir / name
        plt.tight_layout()
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        fig_paths.append(str(out))

    def _best_passive_series(metric: str = "annualized_roi_pct") -> pd.DataFrame:
        passive = grid[grid["strategy_type"].isin(["passive", "passive_rebalanced"])].copy()
        return (
            passive.sort_values(["sell_cost_rate", metric], ascending=[True, False])
            .groupby("sell_cost_rate", as_index=False)
            .first()
            .sort_values("sell_cost_rate")
        )

    def _plot_passive(ax, passive: pd.DataFrame, metric: str, label: str = "Best passive") -> None:
        ax.plot(
            passive["sell_cost_rate"] * 100,
            passive[metric],
            color="black",
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=label,
            zorder=4,
        )

    def _strategy_style(strategy: str) -> tuple[str, str]:
        """Consistent color by rebalance horizon and line style by top bucket."""
        m_days = re.search(r"_(\d+)d$", str(strategy))
        m_top = re.search(r"top_(\d+)pct", str(strategy))
        days = int(m_days.group(1)) if m_days else -1
        top = int(m_top.group(1)) if m_top else -1
        color_map = {
            30: "tab:blue",
            90: "tab:green",
            180: "tab:purple",
            365: "tab:gray",
        }
        style_map = {
            5: "-",
            10: "-",
            20: "--",
            30: ":",
        }
        return color_map.get(days, "tab:orange"), style_map.get(top, "-")

    def _plot_strategy(ax, strategy: str, metric: str, label: str | None = None, alpha: float = 1.0, lw: float = 2.0) -> None:
        g = grid[grid["strategy"] == strategy].sort_values("sell_cost_rate")
        if g.empty:
            return
        color, linestyle = _strategy_style(strategy)
        ax.plot(
            g["sell_cost_rate"] * 100,
            g[metric],
            marker="o",
            markersize=4,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            label=label or _human_strategy_name(strategy),
        )

    alpha_grid_all = grid[grid["strategy_type"] == "alpha_rotation"].copy()
    passive_ann = _best_passive_series("annualized_roi_pct")
    passive_total = _best_passive_series("roi_pct")
    costs_pct = sorted(grid["sell_cost_rate"].dropna().unique() * 100)

    top_values = sorted([int(round(float(x) * 100)) for x in alpha_grid_all["top_pct"].dropna().unique()])
    day_values = sorted([int(x) for x in alpha_grid_all["rebalance_days"].dropna().unique()])

    # ------------------------------------------------------------------
    # Figure 1: Small multiples by TOP BUCKET.
    # Each panel keeps top_pct fixed and compares rebalance horizons.
    # This shows how turnover/holding period changes cost robustness.
    # ------------------------------------------------------------------
    n_panels = len(top_values)
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    handles = None
    labels = None
    for ax, top in zip(axes, top_values):
        for days in day_values:
            strategy = f"alpha_top_{top}pct_rebalance_{days}d"
            _plot_strategy(ax, strategy, "annualized_roi_pct", label=f"Rebalance {days}d", lw=2.2)
        _plot_passive(ax, passive_ann, "annualized_roi_pct", label="Best passive")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Alpha top {top}%")
        ax.set_xlabel("Sell-side cost (%)")
        ax.set_ylabel("Annualized liquidated return (%)")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
    for ax in axes[n_panels:]:
        ax.axis("off")
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=True)
        fig.subplots_adjust(bottom=0.12)
    fig.suptitle("Cost sensitivity grouped by alpha-selection width", y=0.995)
    savefig("19_fig_cost_sensitivity_by_top_bucket.png")

    # ------------------------------------------------------------------
    # Figure 2: Small multiples by REBALANCE HORIZON.
    # Each panel keeps the holding/rebalance horizon fixed and compares top cutoffs.
    # This shows whether the 10% bucket is actually the best compromise.
    # ------------------------------------------------------------------
    n_panels = len(day_values)
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    handles = None
    labels = None
    top_color_map = {5: "tab:blue", 10: "tab:orange", 20: "tab:green", 30: "tab:red"}
    for ax, days in zip(axes, day_values):
        for top in top_values:
            strategy = f"alpha_top_{top}pct_rebalance_{days}d"
            g = grid[grid["strategy"] == strategy].sort_values("sell_cost_rate")
            if g.empty:
                continue
            ax.plot(
                g["sell_cost_rate"] * 100,
                g["annualized_roi_pct"],
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=top_color_map.get(top),
                label=f"Top {top}%",
            )
        _plot_passive(ax, passive_ann, "annualized_roi_pct", label="Best passive")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Rebalance every {days} days")
        ax.set_xlabel("Sell-side cost (%)")
        ax.set_ylabel("Annualized liquidated return (%)")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
    for ax in axes[n_panels:]:
        ax.axis("off")
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=True)
        fig.subplots_adjust(bottom=0.12)
    fig.suptitle("Cost sensitivity grouped by rebalance horizon", y=0.995)
    savefig("19_fig_cost_sensitivity_by_rebalance_horizon.png")

    # ------------------------------------------------------------------
    # Figure 3: FINAL presentation graph — Top 10% only, different rebalance horizons.
    # This is the cleanest chart for the current research story.
    # ------------------------------------------------------------------
    focus_top = 10
    plt.figure(figsize=(11, 6.5))
    for days in day_values:
        strategy = f"alpha_top_{focus_top}pct_rebalance_{days}d"
        label = f"Alpha top {focus_top}%, rebalance {days}d"
        _plot_strategy(plt.gca(), strategy, "annualized_roi_pct", label=label, lw=2.7)
    _plot_passive(plt.gca(), passive_ann, "annualized_roi_pct", label="Best passive benchmark")
    plt.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    plt.xlabel("Sell-side cost (%)")
    plt.ylabel("Annualized liquidated return (%)")
    plt.title("Practical candidate family: top 10% alpha basket across rebalance horizons")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9, frameon=True)
    savefig("19_fig_focus_top10_by_rebalance_horizon.png")

    # ------------------------------------------------------------------
    # Figure 4: Same final graph but in full-sample total ROI terms.
    # Useful because the original zero-cost proof was stated in total ROI.
    # ------------------------------------------------------------------
    plt.figure(figsize=(11, 6.5))
    for days in day_values:
        strategy = f"alpha_top_{focus_top}pct_rebalance_{days}d"
        label = f"Alpha top {focus_top}%, rebalance {days}d"
        _plot_strategy(plt.gca(), strategy, "roi_pct", label=label, lw=2.7)
    _plot_passive(plt.gca(), passive_total, "roi_pct", label="Best passive benchmark")
    plt.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    plt.xlabel("Sell-side cost (%)")
    plt.ylabel("Liquidated ROI over full sample (%)")
    plt.title("Total ROI: top 10% alpha basket across rebalance horizons")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9, frameon=True)
    savefig("19_fig_focus_top10_total_roi_by_rebalance_horizon.png")

    # ------------------------------------------------------------------
    # Figure 5: Best-at-10%-cost gap, one alpha curve only.
    # This is the cleanest "does it survive 10% cost?" chart.
    # ------------------------------------------------------------------
    target_cost = 0.10
    available_costs = sorted(grid["sell_cost_rate"].dropna().unique())
    chosen_cost = min(available_costs, key=lambda c: abs(float(c) - target_cost))
    alpha_at_target = alpha_grid_all[alpha_grid_all["sell_cost_rate"] == chosen_cost].copy()
    if len(alpha_at_target):
        best_strategy = alpha_at_target.sort_values("roi_pct", ascending=False).iloc[0]["strategy"]
        best_curve = grid[grid["strategy"] == best_strategy].sort_values("sell_cost_rate")
        passive_curve = passive_ann.sort_values("sell_cost_rate")

        best_point = best_curve.loc[best_curve["sell_cost_rate"] == chosen_cost].iloc[0]
        passive_point = passive_curve.loc[passive_curve["sell_cost_rate"] == chosen_cost].iloc[0]
        gap_total = float(best_point["roi_pct"] - passive_point["roi_pct"])
        gap_ann = float(best_point["annualized_roi_pct"] - passive_point["annualized_roi_pct"])

        plt.figure(figsize=(11, 6.5))
        _plot_strategy(
            plt.gca(),
            str(best_strategy),
            "annualized_roi_pct",
            label=_human_strategy_name(str(best_strategy)),
            lw=3.0,
        )
        _plot_passive(plt.gca(), passive_ann, "annualized_roi_pct", label="Best passive benchmark")
        x = float(chosen_cost) * 100
        y0 = float(passive_point["annualized_roi_pct"])
        y1 = float(best_point["annualized_roi_pct"])
        plt.vlines(x, y0, y1, color="tab:red", linewidth=3.0, alpha=0.9)
        plt.scatter([x, x], [y0, y1], color="tab:red", s=70, zorder=5)
        plt.annotate(
            f"Positive gap at {x:.0f}% sell cost\n"
            f"Annualized excess: {gap_ann:.1f} pp\n"
            f"Total ROI excess: {gap_total:.1f} pp",
            xy=(x, (y0 + y1) / 2),
            xytext=(x + 1.2, (y0 + y1) / 2),
            arrowprops={"arrowstyle": "->", "lw": 1.2},
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.5", "alpha": 0.95},
        )
        plt.axhline(0, color="black", linewidth=0.9, alpha=0.5)
        plt.xlabel("Sell-side cost (%)")
        plt.ylabel("Annualized liquidated return (%)")
        plt.title(f"Best alpha strategy at {x:.0f}% cost vs passive benchmark")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9, frameon=True)
        savefig("19_fig_best_at_10pct_cost_gap.png")

    # ------------------------------------------------------------------
    # Figure 6: Academic heatmap of ANNUALIZED EXCESS return at the selected
    # cost, versus the passive buy-and-hold benchmark.
    # Positive cells are where alpha rotation adds value over buy-and-hold.
    # ------------------------------------------------------------------
    cost_for_matrix = chosen_cost
    m = alpha_grid = alpha_grid_all[alpha_grid_all["sell_cost_rate"] == cost_for_matrix].copy()
    bh_mask = (grid["strategy"] == "buy_and_hold_all") & (grid["sell_cost_rate"] == cost_for_matrix)
    bh_rows = grid.loc[bh_mask].copy()
    if len(m) and len(bh_rows):
        from matplotlib import colors as mcolors
        from matplotlib.patches import Rectangle

        bh_ann = float(bh_rows["annualized_roi_pct"].iloc[0])
        alpha_grid["annualized_excess_vs_buy_hold_pct"] = alpha_grid["annualized_roi_pct"].astype(float) - bh_ann

        pivot = alpha_grid.pivot_table(
            index="top_pct",
            columns="rebalance_days",
            values="annualized_excess_vs_buy_hold_pct",
            aggfunc="first",
        ).sort_index()

        values = pivot.values.astype(float)
        finite = values[np.isfinite(values)] if values.size else np.array([])
        vmax = max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1.0)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)

        x_labels = [f"{int(c)}d" for c in pivot.columns]
        y_labels = [f"Top {int(round(float(i) * 100))}%" for i in pivot.index]
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.set_xlabel("Rebalance interval", fontsize=12)
        ax.set_ylabel("Alpha-selection width", fontsize=12)
        ax.set_title(
            f"Annualized excess return vs buy-and-hold at {cost_for_matrix * 100:.0f}% sell-side cost\n"
            "alpha annualized return minus buy-and-hold annualized return",
            fontsize=14,
            pad=12,
        )
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Annualized excess return (percentage points)", fontsize=11)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = values[i, j]
                if not np.isfinite(val):
                    continue
                positive = val > 0
                txt_color = "black" if abs(val) < 0.55 * vmax else "white"
                label = f"{val:.1f} pp" if val >= 0 else f"−{abs(val):.1f} pp"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="normal",
                    color=txt_color,
                    bbox={"boxstyle": "round,pad=0.20", "fc": (1, 1, 1, 0.14), "ec": (1, 1, 1, 0.0)},
                )
                if positive:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.9))

        fig.text(
            0.012,
            0.012,
            f"Baseline buy-and-hold annualized return at {cost_for_matrix * 100:.0f}% cost: {bh_ann:.1f}%",
            fontsize=10,
            color="0.25",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        savefig("19_fig_matrix_excess_annualized_vs_buy_hold_10pct_cost.png")

    # ------------------------------------------------------------------
    # Figure 7: Break-even cost bars for alpha strategies, grouped by rebalance horizon.
    # ------------------------------------------------------------------
    be_alpha = break_even[break_even["strategy_type"] == "alpha_rotation"].copy()
    be_alpha = be_alpha.sort_values("passive_relative_break_even_cost", ascending=False, na_position="last").head(16)
    plt.figure(figsize=(11, 7))
    labels = [_human_strategy_name(s) for s in be_alpha["strategy"].astype(str)]
    values = be_alpha["passive_relative_break_even_cost"].astype(float).mul(100).fillna(0.0)
    plt.barh(range(len(be_alpha)), values)
    plt.yticks(range(len(be_alpha)), labels, fontsize=8)
    plt.xlabel("Passive-relative break-even sell cost (%)")
    plt.title("Estimated cost tolerance by alpha strategy")
    plt.grid(True, axis="x", alpha=0.25)
    plt.gca().invert_yaxis()
    savefig("19_fig_break_even_costs_readable.png")

    # Keep a low-priority full spaghetti chart for audit/debug only, not as the main exhibit.
    plt.figure(figsize=(14, 7))
    for strategy in sorted(alpha_grid_all["strategy"].unique()):
        _plot_strategy(plt.gca(), strategy, "annualized_roi_pct", alpha=0.55, lw=1.4)
    _plot_passive(plt.gca(), passive_ann, "annualized_roi_pct", label="Best passive benchmark")
    plt.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    plt.xlabel("Sell-side cost (%)")
    plt.ylabel("Annualized liquidated return (%)")
    plt.title("Audit view: all alpha-rotation strategies")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    savefig("19_fig_audit_all_alpha_curves.png")



    # ------------------------------------------------------------------
    # Figure 8: Academic portfolio-sort plot. Within each day, items are
    # sorted into alpha deciles. The figure shows whether higher-ranked
    # deciles realize higher forward returns.
    # ------------------------------------------------------------------
    if alpha is not None and universe is not None and splits is not None:
        signal_panel = _build_alpha_metadata_panel(universe, alpha)
        target_col = _target_diagnostic_col(signal_panel)
        if target_col is not None and len(signal_panel) > 0:
            dec = signal_panel.dropna(subset=["alpha_pred", target_col]).copy()
            dec["alpha_decile"] = _assign_daily_alpha_deciles(dec, "alpha_pred", 10)
            dec = dec.dropna(subset=["alpha_decile"])
            dec["alpha_decile"] = dec["alpha_decile"].astype(int) + 1
            dec["split"] = _split_label(dec["date"], splits)
            dec[target_col] = pd.to_numeric(dec[target_col], errors="coerce") * 100.0
            agg = (
                dec[dec["split"].isin(["train", "validation", "test"])]
                .groupby(["split", "alpha_decile"], as_index=False)[target_col]
                .mean()
            )
            order = ["train", "validation", "test"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
            for ax, split_name in zip(axes, order):
                g = agg[agg["split"] == split_name]
                ax.bar(g["alpha_decile"], g[target_col])
                ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
                ax.set_title(split_name.title())
                ax.set_xlabel("Alpha decile (10 = highest)")
                ax.grid(True, axis="y", alpha=0.25)
                if len(g):
                    top = float(g.loc[g["alpha_decile"] == 10, target_col].mean())
                    bot = float(g.loc[g["alpha_decile"] == 1, target_col].mean())
                    ax.text(
                        0.02,
                        0.96,
                        f"Top-bottom: {top - bot:+.1f} pp",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.6", "alpha": 0.9},
                    )
            axes[0].set_ylabel("Mean realized forward return (%)")
            fig.suptitle("Alpha-sorted portfolio returns by split", fontsize=14, y=1.02)
            savefig("19_fig_alpha_decile_realized_forward_return.png")

            # Rolling top-minus-bottom spread, a compact stability diagnostic.
            spreads = []
            for d, g in dec.groupby("date", sort=True):
                if g["alpha_decile"].nunique() < 5:
                    continue
                top_mean = g.loc[g["alpha_decile"] == 10, target_col].mean()
                bot_mean = g.loc[g["alpha_decile"] == 1, target_col].mean()
                if pd.notna(top_mean) and pd.notna(bot_mean):
                    spreads.append({"date": pd.Timestamp(d), "top_bottom_spread_pp": float(top_mean - bot_mean)})
            spread_df = pd.DataFrame(spreads).sort_values("date")
            if len(spread_df) >= 20:
                spread_df["rolling_90d_spread_pp"] = spread_df["top_bottom_spread_pp"].rolling(90, min_periods=20).mean()
                plt.figure(figsize=(11, 5.2))
                plt.plot(spread_df["date"], spread_df["rolling_90d_spread_pp"], linewidth=2.2)
                plt.axhline(0, color="black", linewidth=0.9, alpha=0.7)
                for label, start_key in [("Validation", "val_start"), ("Test", "test_start")]:
                    if start_key in splits:
                        plt.axvline(pd.Timestamp(splits[start_key]), color="0.4", linestyle="--", linewidth=1.0, alpha=0.8)
                        plt.text(pd.Timestamp(splits[start_key]), plt.ylim()[1], f" {label}", va="top", fontsize=9)
                plt.title("Rolling alpha spread: top decile minus bottom decile")
                plt.ylabel("90-day rolling mean spread (percentage points)")
                plt.xlabel("Signal date")
                plt.grid(True, alpha=0.25)
                savefig("19_fig_rolling_top_bottom_alpha_spread.png")

        # ------------------------------------------------------------------
        # Figure 9: Metadata preference maps.
        # Instead of a simple bar chart, these figures show how the alpha model
        # changes the composition of the item universe across the full ranking.
        # ------------------------------------------------------------------
        meta_panel = _build_alpha_metadata_panel(universe, alpha)
        if len(meta_panel) > 0:
            meta_panel["rank_pct"] = meta_panel.groupby("date")["alpha_pred"].rank(method="average", pct=True)
            meta_panel["alpha_decile"] = _assign_daily_alpha_deciles(meta_panel, "alpha_pred", 10)
            meta_panel = meta_panel.dropna(subset=["alpha_decile"]).copy()
            meta_panel["alpha_decile"] = meta_panel["alpha_decile"].astype(int) + 1
            meta_panel["is_top10"] = meta_panel["rank_pct"] >= 0.90
            meta_panel["is_top20"] = meta_panel["rank_pct"] >= 0.80
            cats = [c for c in ["quality", "rarity", "type", "metatype"] if c in meta_panel.columns]

            # Save a compact composition table for audit and report tables.
            comp_rows = []
            for col in cats:
                for bucket, mask in [("Universe", pd.Series(True, index=meta_panel.index)), ("Top 20%", meta_panel["is_top20"]), ("Top 10%", meta_panel["is_top10"])]:
                    s = meta_panel.loc[mask, col].value_counts(normalize=True).head(8)
                    for k, v in s.items():
                        comp_rows.append({"dimension": col, "bucket": bucket, "category": str(k), "share_pct": float(v * 100.0)})
            comp_df = pd.DataFrame(comp_rows)
            if len(comp_df):
                comp_df.to_csv(reports_dir / "19_metadata_composition_top_alpha_buckets.csv", index=False)

            # 9A. Bubble heatmap: rows are metadata levels, columns are alpha deciles.
            # Color = enrichment versus the whole tradeable universe; size = category share inside each decile.
            pref_rows = []
            for col in cats:
                base_share = meta_panel[col].value_counts(normalize=True)
                # Pick categories that are common enough or strongly enriched in the top decile.
                top10_share = meta_panel.loc[meta_panel["is_top10"], col].value_counts(normalize=True)
                candidates = pd.concat([base_share.rename("base"), top10_share.rename("top10")], axis=1).fillna(0.0)
                candidates["abs_enrichment"] = (candidates["top10"] - candidates["base"]).abs()
                candidates = candidates[(candidates["base"] >= 0.01) | (candidates["top10"] >= 0.01)]
                keep = candidates.sort_values(["abs_enrichment", "top10"], ascending=False).head(6).index.tolist()
                for cat in keep:
                    base = float(base_share.get(cat, 0.0))
                    for decile, g_dec in meta_panel.groupby("alpha_decile"):
                        share = float((g_dec[col] == cat).mean()) if len(g_dec) else 0.0
                        pref_rows.append({
                            "dimension": col,
                            "category": str(cat),
                            "label": f"{col}: {_shorten_category_label(cat, 22)}",
                            "alpha_decile": int(decile),
                            "share_pct": share * 100.0,
                            "enrichment_pp": (share - base) * 100.0,
                        })
            pref_df = pd.DataFrame(pref_rows)
            if len(pref_df):
                # Order rows by dimension and by top-decile enrichment inside each dimension.
                top_enrich = pref_df[pref_df["alpha_decile"] == 10].set_index("label")["enrichment_pp"].to_dict()
                dim_order = {d: i for i, d in enumerate(cats)}
                labels = sorted(pref_df["label"].unique(), key=lambda x: (dim_order.get(x.split(":", 1)[0], 99), -top_enrich.get(x, 0.0)))
                y_map = {lab: i for i, lab in enumerate(labels)}
                pref_df["y"] = pref_df["label"].map(y_map)

                fig_h = max(7.0, 0.34 * len(labels) + 2.3)
                fig, ax = plt.subplots(figsize=(12.0, fig_h))
                finite = pref_df["enrichment_pp"].replace([np.inf, -np.inf], np.nan).dropna()
                vmax = max(float(finite.abs().quantile(0.98)) if len(finite) else 1.0, 1.0)
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
                sizes = 18 + pref_df["share_pct"].clip(lower=0.0, upper=60.0) * 9.0
                sc = ax.scatter(
                    pref_df["alpha_decile"],
                    pref_df["y"],
                    s=sizes,
                    c=pref_df["enrichment_pp"],
                    cmap="RdYlGn",
                    norm=norm,
                    edgecolor="0.25",
                    linewidth=0.25,
                    alpha=0.90,
                )
                ax.axvline(9.5, color="0.2", linewidth=1.0, linestyle="--", alpha=0.55)
                ax.set_xlim(0.5, 10.5)
                ax.set_xticks(range(1, 11))
                ax.set_xlabel("Alpha decile (10 = highest-ranked)", fontsize=12)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.invert_yaxis()
                ax.set_title("Metadata preference map across alpha-ranked portfolios", fontsize=14, pad=12)
                ax.grid(True, axis="x", alpha=0.18)
                cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
                cbar.set_label("Enrichment versus universe (percentage points)", fontsize=10)
                # Bubble-size legend using invisible handles.
                for share in [5, 15, 30]:
                    ax.scatter([], [], s=18 + share * 9.0, c="white", edgecolor="0.35", label=f"{share}% share")
                ax.legend(title="Bubble size", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
                fig.text(
                    0.012,
                    0.012,
                    "Color shows over/under-representation relative to all tradeable items; dashed line separates the top decile.",
                    fontsize=9,
                    color="0.30",
                )
                savefig("19_fig_metadata_preference_map_alpha_deciles.png")

            # 9B. Archetype map: which metadata combinations become concentrated in the top alpha bucket?
            combo_cols = [c for c in ["quality", "rarity", "type", "metatype"] if c in meta_panel.columns]
            if len(combo_cols) >= 2:
                combo = meta_panel.copy()
                combo["archetype"] = combo[combo_cols].astype("string").agg(" | ".join, axis=1)
                base = combo["archetype"].value_counts(normalize=True).rename("universe_share")
                top = combo.loc[combo["is_top10"], "archetype"].value_counts(normalize=True).rename("top10_share")
                counts = combo.loc[combo["is_top10"], "archetype"].value_counts().rename("top10_count")
                arche = pd.concat([base, top, counts], axis=1).fillna(0.0)
                arche["universe_share_pct"] = arche["universe_share"] * 100.0
                arche["top10_share_pct"] = arche["top10_share"] * 100.0
                arche["enrichment_pp"] = arche["top10_share_pct"] - arche["universe_share_pct"]
                arche["lift"] = np.where(arche["universe_share"] > 0, arche["top10_share"] / arche["universe_share"], np.nan)
                arche = arche[(arche["top10_count"] >= 20) | (arche["top10_share_pct"] >= 0.5)].copy()
                arche = arche.sort_values(["enrichment_pp", "top10_share_pct"], ascending=False).head(18)
                if len(arche):
                    fig, ax = plt.subplots(figsize=(10.8, 7.0))
                    finite_lift = arche["lift"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
                    # TwoSlopeNorm requires vmin < vcenter < vmax. In some runs all
                    # plotted archetypes are enriched (lift > 1), so the data minimum
                    # can be above the neutral lift of 1.0. Anchor the lower side below
                    # 1.0 to keep the neutral no-enrichment point meaningful and avoid
                    # Matplotlib's ascending-order ValueError.
                    lift_min = float(np.nanmin(finite_lift.to_numpy(dtype=float)))
                    lift_max = float(np.nanmax(finite_lift.to_numpy(dtype=float)))
                    vmin_lift = min(lift_min, 0.5)
                    vmax_lift = max(lift_max, 1.5)
                    norm_lift = mcolors.TwoSlopeNorm(vmin=vmin_lift, vcenter=1.0, vmax=vmax_lift)
                    sizes = 55 + np.sqrt(arche["top10_count"].astype(float).clip(lower=1.0)) * 22
                    sc = ax.scatter(
                        arche["universe_share_pct"],
                        arche["top10_share_pct"],
                        s=sizes,
                        c=finite_lift,
                        cmap="PuOr_r",
                        norm=norm_lift,
                        alpha=0.86,
                        edgecolor="0.25",
                        linewidth=0.45,
                    )
                    max_axis = max(float(arche["universe_share_pct"].max()), float(arche["top10_share_pct"].max()), 1.0) * 1.16
                    ax.plot([0, max_axis], [0, max_axis], color="0.25", linestyle="--", linewidth=1.0, alpha=0.7)
                    ax.set_xlim(0, max_axis)
                    ax.set_ylim(0, max_axis)
                    ax.set_xlabel("Share of full tradeable universe (%)", fontsize=12)
                    ax.set_ylabel("Share of top-10% alpha basket (%)", fontsize=12)
                    ax.set_title("Top-alpha item archetypes: concentration versus the universe", fontsize=14, pad=12)
                    ax.grid(True, alpha=0.22)
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.035)
                    cbar.set_label("Lift: top-10% share / universe share", fontsize=10)
                    # Annotate only the clearest outliers to avoid clutter.
                    label_df = arche.sort_values(["enrichment_pp", "top10_share_pct"], ascending=False).head(8)
                    for name, row in label_df.iterrows():
                        ax.annotate(
                            _shorten_category_label(name, 42),
                            xy=(row["universe_share_pct"], row["top10_share_pct"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                            bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "0.75", "alpha": 0.88},
                        )
                    fig.text(
                        0.012,
                        0.012,
                        "Points above the dashed line are over-represented in the top-alpha basket; size reflects top-basket count.",
                        fontsize=9,
                        color="0.30",
                    )
                    savefig("19_fig_metadata_archetype_lift_top10.png")

        # ------------------------------------------------------------------
        # Figure 10: Return-turnover frontier at the selected cost.
        # This is a compact way to show the implementation trade-off.
        # ------------------------------------------------------------------
        chosen = grid[(grid["sell_cost_rate"] == chosen_cost) & (grid["strategy_type"] == "alpha_rotation")].copy()
        if len(chosen):
            plt.figure(figsize=(10.5, 6))
            for days, g in chosen.groupby("rebalance_days"):
                plt.scatter(
                    g["sell_turnover_to_initial_capital"],
                    g["annualized_excess_roi_vs_best_passive_pct"],
                    s=70 + 4 * g["top_pct"].fillna(0).astype(float) * 100,
                    label=f"Rebalance {int(days)}d",
                    alpha=0.85,
                )
                for _, r in g.iterrows():
                    plt.text(
                        r["sell_turnover_to_initial_capital"],
                        r["annualized_excess_roi_vs_best_passive_pct"],
                        f" {int(round(float(r['top_pct']) * 100))}%",
                        fontsize=8,
                        va="center",
                    )
            plt.axhline(0, color="black", linewidth=0.9, alpha=0.7)
            plt.xlabel("Sell turnover / initial capital")
            plt.ylabel("Annualized excess return vs best passive (pp)")
            plt.title(f"Return-turnover trade-off at {chosen_cost * 100:.0f}% sell-side cost")
            plt.grid(True, alpha=0.25)
            plt.legend(title="Holding horizon", fontsize=9)
            savefig("19_fig_return_turnover_frontier_10pct_cost.png")

    return fig_paths

def _print_table(grid: pd.DataFrame) -> None:
    z = grid[grid["sell_cost_rate"] == grid["sell_cost_rate"].min()].sort_values("roi_pct", ascending=False).head(12)
    print("\nTop zero-cost strategies")
    print("=" * 110)
    print(f"{'Strategy':<36} | {'ROI':>9} | {'Excess':>9} | {'Sharpe':>7} | {'DD':>8} | {'Trades':>7} | {'Avg Pos':>7}")
    print("-" * 110)
    for _, r in z.iterrows():
        print(
            f"{r['strategy']:<36} | "
            f"{_fmt_pct(r['roi_pct']):>9} | "
            f"{_fmt_pct(r['excess_roi_vs_best_passive_pct']):>9} | "
            f"{(f'{r['sharpe']:.2f}' if pd.notna(r['sharpe']) else '—'):>7} | "
            f"{_fmt_pct(r['max_drawdown_pct']):>8} | "
            f"{int(r['n_trades']):>7} | "
            f"{float(r['avg_n_positions']):>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--costs", nargs="+", type=float, default=DEFAULT_COSTS)
    parser.add_argument("--rebalance-days", nargs="+", type=int, default=DEFAULT_REBALANCE_DAYS)
    parser.add_argument("--top-pcts", nargs="+", type=float, default=DEFAULT_TOP_PCTS)
    parser.add_argument("--same-day-execution", action="store_true")
    parser.add_argument("--save-details", action="store_true", help="Save all equity/trade parquet files. Can be large.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG figure generation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    initial_capital = float(cfg["research"]["initial_capital_rub"])
    execute_next_day = not bool(args.same_day_execution)

    reports_dir = resolve_path(cfg, "reports_dir")
    backtests_dir = resolve_path(cfg, "backtests_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    universe, alpha, price_wide, price_wide_mtm = _load_inputs(cfg)
    sim_dates = _make_sim_dates(price_wide, universe, splits)
    daily_scores = _build_daily_tradeable_scores(universe, alpha, sim_dates)

    costs = sorted(set(float(c) for c in args.costs))
    print(
        f"Simulating {len(sim_dates):,} dates from {sim_dates[0].date()} to {sim_dates[-1].date()} "
        f"with {'next-day' if execute_next_day else 'same-day'} execution."
    )
    print("Cost grid:", ", ".join(f"{c * 100:.2f}%" for c in costs))

    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}

    strategy_specs: list[dict[str, Any]] = [
        {
            "strategy": "buy_and_hold_all",
            "strategy_type": "passive",
            "selection_mode": "all",
            "rebalance_days": 10**9,
            "reported_rebalance_days": 0,
            "top_pct": None,
        }
    ]
    for days in args.rebalance_days:
        strategy_specs.append({
            "strategy": f"all_universe_rebalance_{days}d",
            "strategy_type": "passive_rebalanced",
            "selection_mode": "all",
            "rebalance_days": int(days),
            "reported_rebalance_days": int(days),
            "top_pct": None,
        })
    for days in args.rebalance_days:
        for pct in args.top_pcts:
            pct_label = int(round(float(pct) * 100))
            strategy_specs.append({
                "strategy": f"alpha_top_{pct_label}pct_rebalance_{days}d",
                "strategy_type": "alpha_rotation",
                "selection_mode": "top_pct",
                "rebalance_days": int(days),
                "reported_rebalance_days": int(days),
                "top_pct": float(pct),
            })

    for cost in costs:
        print(f"\n=== Sell cost {cost * 100:.2f}% ===")
        nested[_cost_label(cost)] = {}
        for spec in strategy_specs:
            name = spec["strategy"]
            print(f"  {name}")
            eq, tr, cost_stats = simulate_rebalance_with_costs(
                price_wide=price_wide,
                price_wide_mtm=price_wide_mtm,
                daily_scores=daily_scores,
                sim_dates=sim_dates,
                initial_capital=initial_capital,
                rebalance_days=int(spec["rebalance_days"]),
                selection_mode=str(spec["selection_mode"]),
                sell_cost_rate=float(cost),
                top_pct=spec["top_pct"],
                execute_next_day=execute_next_day,
            )
            summary = _strategy_summary(
                strategy_name=name,
                equity_df=eq,
                trades_df=tr,
                initial_capital=initial_capital,
                splits=splits,
                extra={
                    "strategy_type": spec["strategy_type"],
                    "selection_mode": spec["selection_mode"],
                    "top_pct": spec["top_pct"],
                    "rebalance_days": spec["reported_rebalance_days"],
                    "sell_cost_rate": float(cost),
                    "execute_next_day": execute_next_day,
                    **cost_stats,
                },
            )
            nested[_cost_label(cost)][name] = summary
            rows.append(summary)

            if args.save_details:
                prefix = f"friction_{_cost_label(cost)}_{_safe_name(name)}"
                save_table(eq, backtests_dir / f"{prefix}_equity.parquet")
                save_table(tr, backtests_dir / f"{prefix}_trades.parquet")

    raw_grid = pd.DataFrame(rows)
    grid, break_even, split_table = _build_tables(raw_grid)

    save_table(grid, reports_dir / "19_strategy_cost_grid.parquet")
    grid.drop(columns=["by_split"], errors="ignore").to_csv(reports_dir / "19_strategy_cost_grid.csv", index=False)
    save_table(break_even, reports_dir / "19_break_even_table.parquet")
    break_even.to_csv(reports_dir / "19_break_even_table.csv", index=False)
    split_table.to_csv(reports_dir / "19_split_robustness_table.csv", index=False)
    _write_research_tables_md(reports_dir / "19_research_tables.md", grid, break_even)

    fig_paths: list[str] = []
    if not args.no_plots:
        fig_paths = _make_figures(reports_dir, grid, break_even, alpha=alpha, universe=universe, splits=splits)

    best_alpha_0 = (
        grid[(grid["strategy_type"] == "alpha_rotation") & (grid["sell_cost_rate"] == min(costs))]
        .sort_values("roi_pct", ascending=False)
        .head(1)
    )
    best_alpha_name = str(best_alpha_0["strategy"].iloc[0]) if len(best_alpha_0) else None
    best_alpha_be = None
    if best_alpha_name is not None:
        be_row = break_even[break_even["strategy"] == best_alpha_name]
        if len(be_row):
            best_alpha_be = be_row.iloc[0].to_dict()

    output = {
        "config": {
            "initial_capital_rub": initial_capital,
            "cost_model": "sell_cost_rate_only",
            "costs": costs,
            "rebalance_days": [int(x) for x in args.rebalance_days],
            "top_pcts": [float(x) for x in args.top_pcts],
            "execute_next_day": execute_next_day,
            "sim_start": sim_dates[0].date().isoformat(),
            "sim_end": sim_dates[-1].date().isoformat(),
            "n_sim_dates": int(len(sim_dates)),
            "main_roi_field": "final_equity_liquidated_rub",
        },
        "best_zero_cost_alpha_strategy": best_alpha_name,
        "best_zero_cost_alpha_break_even": best_alpha_be,
        "figures": fig_paths,
        "tables": {
            "strategy_cost_grid_csv": str(reports_dir / "19_strategy_cost_grid.csv"),
            "break_even_table_csv": str(reports_dir / "19_break_even_table.csv"),
            "split_robustness_table_csv": str(reports_dir / "19_split_robustness_table.csv"),
            "research_tables_md": str(reports_dir / "19_research_tables.md"),
        },
        "strategies_by_cost": nested,
        "break_even_table": break_even.to_dict(orient="records"),
    }
    write_json(reports_dir / "19_alpha_rotation_friction_sweep_summary.json", output)

    _print_table(grid)
    print("\nBreak-even highlights")
    print("=" * 110)
    for _, r in break_even[break_even["strategy_type"] == "alpha_rotation"].head(10).iterrows():
        print(
            f"{r['strategy']:<36} | passive-relative: {_fmt_cost(r['passive_relative_break_even_cost']):>22} | "
            f"zero-profit: {_fmt_cost(r['zero_profit_break_even_cost']):>22} | "
            f"0%-cost ROI: {_fmt_pct(r['roi_0pct_cost']):>9}"
        )

    print(f"\nSaved summary to {reports_dir / '19_alpha_rotation_friction_sweep_summary.json'}")
    print(f"Saved grid to    {reports_dir / '19_strategy_cost_grid.csv'}")
    print(f"Saved break-even table to {reports_dir / '19_break_even_table.csv'}")
    if fig_paths:
        print("Saved figures:")
        for p in fig_paths:
            print("  ", p)


if __name__ == "__main__":
    main()
