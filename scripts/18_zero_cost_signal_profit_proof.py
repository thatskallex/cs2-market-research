#!/usr/bin/env python
"""
Zero-cost signal-profit proof for the CS2 trading rebuild.

Purpose
-------
This is a research/proof script, not a production strategy script.
It answers the narrow question:

    "If market friction is removed, can the model's alpha signal produce
     profitable long-only portfolios, and can it beat simple passive baselines?"

This deliberately separates signal validity from real-market implementability.
All simulations here use:
  - 0% sell cost
  - no spread/slippage/Steam fee
  - no turnover penalty
  - no final liquidation haircut
  - next-day execution by default to avoid same-day lookahead

What it tests
-------------
1) Passive baselines:
   - buy_and_hold_all: buy the first tradeable universe and hold to end.
   - all_universe_rebalance_<N>d: periodically rebalance into all tradeable items.

2) Alpha rotation:
   - alpha_top_<PCT>_rebalance_<N>d: periodically hold the top PCT of tradeable
     items by alpha_pred.

3) Optional split ROI:
   Train/val/test ROI are reported using the configured split dates.

Outputs
-------
  data/processed/reports/18_zero_cost_signal_profit_proof_summary.json
  backtests/zero_cost_<strategy>_equity.parquet
  backtests/zero_cost_<strategy>_trades.parquet

Run
---
  python scripts/18_zero_cost_signal_profit_proof.py --config configs/cs2_trading_rebuilt.yaml

Optional examples
-----------------
  python scripts/18_zero_cost_signal_profit_proof.py --config configs/cs2_trading_rebuilt.yaml \
    --rebalance-days 30 90 180 365 --top-pcts 0.05 0.10 0.20 0.30

  python scripts/18_zero_cost_signal_profit_proof.py --config configs/cs2_trading_rebuilt.yaml \
    --same-day-execution
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import math
import re
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
ZERO_COST_RATE = 0.0


def _safe_name(name: str) -> str:
    name = name.lower().replace("%", "pct")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _roi_pct(initial_capital: float, final_equity: float) -> float | None:
    if initial_capital <= 0:
        return None
    return float((final_equity / initial_capital - 1.0) * 100.0)


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
        start_eq = float(seg["equity_rub"].iloc[0])
        end_eq = float(seg["equity_rub"].iloc[-1])
        out[split_name] = {
            "n_days": int(len(seg)),
            "start_equity_rub": start_eq,
            "end_equity_rub": end_eq,
            "roi_pct": _roi_pct(start_eq, end_eq),
        }
    return out


def _strategy_summary(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    splits: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(equity_df) == 0:
        final_equity = initial_capital
        avg_pos = 0.0
        max_pos = 0
    else:
        final_equity = float(equity_df["equity_rub"].iloc[-1])
        avg_pos = float(equity_df["n_positions"].mean())
        max_pos = int(equity_df["n_positions"].max())

    summary: dict[str, Any] = {
        "initial_capital_rub": float(initial_capital),
        "final_equity_rub": float(final_equity),
        "roi_pct": _roi_pct(initial_capital, final_equity),
        "sharpe": _sharpe(equity_df["equity_rub"]) if len(equity_df) else None,
        "max_drawdown_pct": _max_drawdown_pct(equity_df["equity_rub"]) if len(equity_df) else None,
        "n_days": int(len(equity_df)),
        "n_trades": int(len(trades_df)),
        "n_buys": int((trades_df["side"] == "buy").sum()) if len(trades_df) else 0,
        "n_sells": int((trades_df["side"] == "sell").sum()) if len(trades_df) else 0,
        "avg_n_positions": avg_pos,
        "max_n_positions": max_pos,
        "by_split": _per_split_roi(equity_df, splits),
    }
    if extra:
        summary.update(extra)
    return summary


def _load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    alpha_path = resolve_path(cfg, "models_dir") / "alpha_predictions.parquet"

    if not alpha_path.exists():
        raise FileNotFoundError(f"alpha_predictions.parquet not found at {alpha_path}")

    print("Loading universe...")
    universe = pd.read_parquet(
        universe_path,
        columns=["date", "market_hash_name", "tradeable_today"],
    )
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
    alpha = pd.read_parquet(alpha_path, columns=["date", "market_hash_name", "alpha_pred"])
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha["alpha_pred"] = pd.to_numeric(alpha["alpha_pred"], errors="coerce").astype("float32")
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


def _make_sim_dates(
    price_wide: pd.DataFrame,
    universe: pd.DataFrame,
    splits: dict[str, Any],
) -> pd.DatetimeIndex:
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
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for d, g in df.groupby("date", sort=True):
        out[pd.Timestamp(d)] = g[["market_hash_name", "alpha_pred"]].copy()
    return out


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


def simulate_zero_cost_rebalance(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    daily_scores: dict[pd.Timestamp, pd.DataFrame],
    sim_dates: pd.DatetimeIndex,
    initial_capital: float,
    rebalance_days: int,
    selection_mode: str,
    top_pct: float | None = None,
    execute_next_day: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full zero-cost rebalance simulator.

    At each scheduled signal date, choose a target set from that date's
    tradeable universe. Execution occurs on the next simulation date by default.
    On execution, the portfolio is adjusted to equal target capital weights.
    Existing holdings may be trimmed or topped up because zero transaction cost
    makes full rebalancing the cleanest signal test.
    """
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    equity_rows: list[dict[str, Any]] = []
    trades_rows: list[dict[str, Any]] = []
    pending_targets: dict[pd.Timestamp, list[str]] = {}
    last_signal_date: pd.Timestamp | None = None

    def mark_to_market(d: pd.Timestamp) -> float:
        if not positions:
            return 0.0
        held = [it for it in positions if it in price_cols]
        if not held:
            return 0.0
        row = price_wide_mtm.loc[d, held]
        mv = 0.0
        for it in held:
            px = row[it]
            if pd.notna(px) and px > 0:
                mv += float(positions[it]) * float(px)
        return mv

    def execute_target(d: pd.Timestamp, target_items: list[str]) -> None:
        nonlocal cash, positions
        # Use last-known prices for current holdings and executable prices for targets.
        # This keeps dormant positions valued/sellable at their last-known mark in the
        # zero-friction proof, avoiding artificial writeoffs.
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
            # Require an actual executable quote on the execution date for buys.
            if it not in px_row_exec.index:
                return None
            px = px_row_exec[it]
            if pd.notna(px) and px > 0:
                return float(px)
            return None

        # First compute current equity at executable/mark prices.
        current_mv = 0.0
        for it, units in positions.items():
            px = px_for_sell(it)
            if px is not None:
                current_mv += units * px
        equity = cash + current_mv
        if equity <= 0:
            return

        buyable_targets = [it for it in target_items if it in price_cols and px_for_buy(it) is not None]
        buyable_targets = sorted(set(buyable_targets))
        if not buyable_targets:
            # No target can be bought today. Do not force-liquidate into cash.
            return

        target_budget = equity / len(buyable_targets)
        desired_units: dict[str, int] = {}
        for it in buyable_targets:
            px = px_for_buy(it)
            if px is None:
                continue
            desired_units[it] = int(target_budget // px)

        # Sell or trim first, zero cost.
        for it in list(positions.keys()):
            old_units = int(positions[it])
            new_units = int(desired_units.get(it, 0))
            if new_units >= old_units:
                continue
            units_to_sell = old_units - new_units
            px = px_for_sell(it)
            if px is None:
                continue
            proceeds = units_to_sell * px * (1.0 - ZERO_COST_RATE)
            cash += proceeds
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
                "gross_value_rub": float(units_to_sell * px),
                "net_value_rub": float(proceeds),
                "reason": "zero_cost_rebalance_trim_or_exit",
            })

        # Buy or top up.
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
            trades_rows.append({
                "date": d,
                "side": "buy",
                "market_hash_name": it,
                "price_rub": px,
                "units": int(units_to_buy),
                "gross_value_rub": float(cost),
                "net_value_rub": float(cost),
                "reason": "zero_cost_rebalance_buy_or_topup",
            })

    for i, d0 in enumerate(sim_dates):
        d = pd.Timestamp(d0)

        # Execute any targets scheduled for today before marking today's end equity.
        for target_items in pending_targets.pop(d, []):
            execute_target(d, target_items)

        mv = mark_to_market(d)
        equity_rows.append({
            "date": d,
            "cash_rub": float(cash),
            "market_value_rub": float(mv),
            "equity_rub": float(cash + mv),
            "n_positions": int(len(positions)),
        })

        # Schedule new target from today's signal if rebalance is due.
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

    return pd.DataFrame(equity_rows), pd.DataFrame(trades_rows)


def simulate_zero_cost_buy_and_hold(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    daily_scores: dict[pd.Timestamp, pd.DataFrame],
    sim_dates: pd.DatetimeIndex,
    initial_capital: float,
    execute_next_day: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Equivalent to one rebalance into all names, then never rebalance.
    return simulate_zero_cost_rebalance(
        price_wide=price_wide,
        price_wide_mtm=price_wide_mtm,
        daily_scores=daily_scores,
        sim_dates=sim_dates,
        initial_capital=initial_capital,
        rebalance_days=10**9,
        selection_mode="all",
        top_pct=None,
        execute_next_day=execute_next_day,
    )


def _print_summary_table(summaries: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 118)
    print("Zero-cost strategy comparison")
    print("=" * 118)
    print(
        f"{'Strategy':<36} | {'ROI':>9} | {'Final Eq':>12} | {'Sharpe':>7} | "
        f"{'DD':>8} | {'Trades':>7} | {'Avg Pos':>7} | {'Train':>8} | {'Val':>8} | {'Test':>8}"
    )
    print("-" * 118)
    for name, s in sorted(summaries.items(), key=lambda kv: (kv[1].get("roi_pct") is None, -(kv[1].get("roi_pct") or -1e18))):
        by = s.get("by_split", {})
        tr = by.get("train", {}).get("roi_pct")
        va = by.get("val", {}).get("roi_pct")
        te = by.get("test", {}).get("roi_pct")
        print(
            f"{name:<36} | "
            f"{(f'{s.get('roi_pct'):+.1f}%' if s.get('roi_pct') is not None else '-'):>9} | "
            f"{s.get('final_equity_rub', float('nan')):>12,.0f} | "
            f"{(f'{s.get('sharpe'):.2f}' if s.get('sharpe') is not None else '-'):>7} | "
            f"{(f'{s.get('max_drawdown_pct'):+.1f}%' if s.get('max_drawdown_pct') is not None else '-'):>8} | "
            f"{s.get('n_trades', 0):>7} | "
            f"{s.get('avg_n_positions', 0.0):>7.1f} | "
            f"{(f'{tr:+.1f}%' if tr is not None else '-'):>8} | "
            f"{(f'{va:+.1f}%' if va is not None else '-'):>8} | "
            f"{(f'{te:+.1f}%' if te is not None else '-'):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebalance-days", nargs="+", type=int, default=DEFAULT_REBALANCE_DAYS)
    parser.add_argument("--top-pcts", nargs="+", type=float, default=DEFAULT_TOP_PCTS)
    parser.add_argument("--same-day-execution", action="store_true", help="Use same-day execution instead of next-day execution.")
    parser.add_argument("--no-save-details", action="store_true", help="Only save JSON summary, not equity/trade parquet files.")
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

    print(
        f"Simulating {len(sim_dates):,} dates from {sim_dates[0].date()} to {sim_dates[-1].date()} "
        f"with {'next-day' if execute_next_day else 'same-day'} execution and 0% cost."
    )

    summaries: dict[str, dict[str, Any]] = {}

    # Passive one-shot benchmark.
    name = "buy_and_hold_all"
    print(f"\n=== {name} ===")
    eq, tr = simulate_zero_cost_buy_and_hold(
        price_wide=price_wide,
        price_wide_mtm=price_wide_mtm,
        daily_scores=daily_scores,
        sim_dates=sim_dates,
        initial_capital=initial_capital,
        execute_next_day=execute_next_day,
    )
    summaries[name] = _strategy_summary(eq, tr, initial_capital, splits, extra={
        "strategy_type": "passive",
        "selection_mode": "all",
        "rebalance_days": 0,
        "sell_cost_rate": ZERO_COST_RATE,
        "execute_next_day": execute_next_day,
    })
    if not args.no_save_details:
        save_table(eq, backtests_dir / f"zero_cost_{_safe_name(name)}_equity.parquet")
        save_table(tr, backtests_dir / f"zero_cost_{_safe_name(name)}_trades.parquet")

    # Passive rebalance benchmarks.
    for days in args.rebalance_days:
        name = f"all_universe_rebalance_{days}d"
        print(f"\n=== {name} ===")
        eq, tr = simulate_zero_cost_rebalance(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            daily_scores=daily_scores,
            sim_dates=sim_dates,
            initial_capital=initial_capital,
            rebalance_days=int(days),
            selection_mode="all",
            top_pct=None,
            execute_next_day=execute_next_day,
        )
        summaries[name] = _strategy_summary(eq, tr, initial_capital, splits, extra={
            "strategy_type": "passive_rebalanced",
            "selection_mode": "all",
            "rebalance_days": int(days),
            "sell_cost_rate": ZERO_COST_RATE,
            "execute_next_day": execute_next_day,
        })
        if not args.no_save_details:
            save_table(eq, backtests_dir / f"zero_cost_{_safe_name(name)}_equity.parquet")
            save_table(tr, backtests_dir / f"zero_cost_{_safe_name(name)}_trades.parquet")

    # Alpha top-pct rotations.
    for days in args.rebalance_days:
        for pct in args.top_pcts:
            pct_label = int(round(float(pct) * 100))
            name = f"alpha_top_{pct_label}pct_rebalance_{days}d"
            print(f"\n=== {name} ===")
            eq, tr = simulate_zero_cost_rebalance(
                price_wide=price_wide,
                price_wide_mtm=price_wide_mtm,
                daily_scores=daily_scores,
                sim_dates=sim_dates,
                initial_capital=initial_capital,
                rebalance_days=int(days),
                selection_mode="top_pct",
                top_pct=float(pct),
                execute_next_day=execute_next_day,
            )
            summaries[name] = _strategy_summary(eq, tr, initial_capital, splits, extra={
                "strategy_type": "alpha_rotation",
                "selection_mode": "top_pct",
                "top_pct": float(pct),
                "rebalance_days": int(days),
                "sell_cost_rate": ZERO_COST_RATE,
                "execute_next_day": execute_next_day,
            })
            if not args.no_save_details:
                save_table(eq, backtests_dir / f"zero_cost_{_safe_name(name)}_equity.parquet")
                save_table(tr, backtests_dir / f"zero_cost_{_safe_name(name)}_trades.parquet")

    # Add deltas versus the strongest passive benchmark and versus buy-and-hold.
    passive_names = [n for n, s in summaries.items() if s.get("strategy_type") in {"passive", "passive_rebalanced"}]
    best_passive_name = max(passive_names, key=lambda n: summaries[n].get("roi_pct") if summaries[n].get("roi_pct") is not None else -1e18)
    bh_roi = summaries["buy_and_hold_all"].get("roi_pct")
    best_passive_roi = summaries[best_passive_name].get("roi_pct")
    for s in summaries.values():
        roi = s.get("roi_pct")
        s["delta_roi_vs_buy_and_hold_all_pct"] = float(roi - bh_roi) if roi is not None and bh_roi is not None else None
        s["delta_roi_vs_best_passive_pct"] = float(roi - best_passive_roi) if roi is not None and best_passive_roi is not None else None

    alpha_names = [n for n, s in summaries.items() if s.get("strategy_type") == "alpha_rotation"]
    best_alpha_name = max(alpha_names, key=lambda n: summaries[n].get("roi_pct") if summaries[n].get("roi_pct") is not None else -1e18) if alpha_names else None

    output = {
        "config": {
            "initial_capital_rub": initial_capital,
            "sell_cost_rate": ZERO_COST_RATE,
            "execute_next_day": execute_next_day,
            "rebalance_days": [int(x) for x in args.rebalance_days],
            "top_pcts": [float(x) for x in args.top_pcts],
            "sim_start": sim_dates[0].date().isoformat(),
            "sim_end": sim_dates[-1].date().isoformat(),
            "n_sim_dates": int(len(sim_dates)),
        },
        "best_passive_strategy": best_passive_name,
        "best_alpha_strategy": best_alpha_name,
        "proof_result": {
            "best_alpha_beats_buy_and_hold_all": (
                bool(summaries[best_alpha_name]["delta_roi_vs_buy_and_hold_all_pct"] > 0) if best_alpha_name else False
            ),
            "best_alpha_beats_best_passive": (
                bool(summaries[best_alpha_name]["delta_roi_vs_best_passive_pct"] > 0) if best_alpha_name else False
            ),
            "best_alpha_roi_pct": summaries[best_alpha_name]["roi_pct"] if best_alpha_name else None,
            "buy_and_hold_all_roi_pct": summaries["buy_and_hold_all"]["roi_pct"],
            "best_passive_roi_pct": summaries[best_passive_name]["roi_pct"],
        },
        "strategies": summaries,
    }

    _print_summary_table(summaries)
    print("\nBest passive:", best_passive_name, f"ROI={summaries[best_passive_name]['roi_pct']:+.2f}%")
    if best_alpha_name:
        print("Best alpha:  ", best_alpha_name, f"ROI={summaries[best_alpha_name]['roi_pct']:+.2f}%")
        print(
            "Delta best alpha vs buy_and_hold_all:",
            f"{summaries[best_alpha_name]['delta_roi_vs_buy_and_hold_all_pct']:+.2f} pp",
        )
        print(
            "Delta best alpha vs best passive:",
            f"{summaries[best_alpha_name]['delta_roi_vs_best_passive_pct']:+.2f} pp",
        )

    out_path = reports_dir / "18_zero_cost_signal_profit_proof_summary.json"
    write_json(out_path, output)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
