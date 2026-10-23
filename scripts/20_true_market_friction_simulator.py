#!/usr/bin/env python
"""
True-market alpha rotation friction simulator.

This is the realistic counterpart to the simplified zero-cost / friction sweep.
It keeps the same research question — does alpha rotation survive market
friction? — but makes execution materially closer to the CS2 market:

  - finite capital and integer item units
  - next-day execution after signal/rebalance day
  - no fills on zero-volume days
  - daily participation cap based on observed units_sold
  - unfilled orders can remain open for a limited number of days
  - sells execute before buys, so cash must actually be raised
  - buy premium, sell discount / fee, and optional impact from participation
  - mark-to-market uses last known prices, while execution requires volume
  - final liquidation is reported as an immediate liquidation estimate
  - outputs research tables and figures for paper-style diagnostics

Run:
  python scripts/20_true_market_friction_simulator.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  data/processed/reports/20_true_market_summary.json
  data/processed/reports/20_true_market_strategy_grid.csv
  data/processed/reports/20_true_market_break_even_table.csv
  data/processed/reports/20_true_market_research_tables.md
  data/processed/reports/20_fig_*.png
  backtests/20_true_market_<strategy>_cost_<bp>bp_equity.parquet
  backtests/20_true_market_<strategy>_cost_<bp>bp_trades.parquet
  backtests/20_true_market_<strategy>_cost_<bp>bp_orders.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Research grid
# ---------------------------------------------------------------------------

INITIAL_CAPITAL_DEFAULT = 50_000.0
COSTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
TOP_PCTS = [0.05, 0.10, 0.20, 0.30]
REBALANCE_DAYS = [30, 90, 180, 365]

# Market-realism knobs. These are deliberately visible in the summary so the
# result is interpretable as an execution-assumption experiment.
BUY_PREMIUM_RATE = 0.00          # set >0 if you want to model crossing the ask
BASE_SELL_DISCOUNT_RATE = 0.00   # non-fee forced-sale discount; cost sweep is separate
PARTICIPATION_RATE = 0.20        # max share of observed daily units_sold we can take
IMPACT_COEFF = 0.03              # impact = coeff * realized participation share
MAX_ORDER_LAG_DAYS = 14          # cancel unfilled orders after this many calendar days
ALLOW_ONE_UNIT_IF_VOLUME_POSITIVE = True
MIN_REBALANCE_CASH_RUB = 1.0


@dataclass
class Order:
    created_date: pd.Timestamp
    side: str
    item: str
    units_remaining: int
    reason: str
    strategy: str


def _safe_pct(initial: float, final: float) -> float:
    return float((final / initial - 1.0) * 100.0) if initial > 0 else np.nan


def _sharpe(equity: pd.Series) -> float:
    ret = equity.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2 or ret.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(365.0) * ret.mean() / ret.std(ddof=1))


def _max_drawdown_pct(equity: pd.Series) -> float:
    s = equity.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    peak = s.cummax().replace(0.0, np.nan)
    dd = (s - peak) / peak
    return float(dd.min() * 100.0)


def _split_summary(equity_df: pd.DataFrame, splits: dict[str, Any]) -> dict[str, Any]:
    out = {}
    if len(equity_df) == 0:
        return out
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for split, keys in {
        "train": ("train_start", "train_end"),
        "val": ("val_start", "val_end"),
        "test": ("test_start", "test_end"),
    }.items():
        s = pd.Timestamp(splits[keys[0]]).normalize()
        e = pd.Timestamp(splits[keys[1]]).normalize()
        seg = df[(df["date"] >= s) & (df["date"] <= e)]
        if len(seg) == 0:
            continue
        start_eq = float(seg["equity_liquidation_estimate_rub"].iloc[0])
        end_eq = float(seg["equity_liquidation_estimate_rub"].iloc[-1])
        out[split] = {
            "n_days": int(len(seg)),
            "start_equity_rub": start_eq,
            "end_equity_rub": end_eq,
            "roi_pct": _safe_pct(start_eq, end_eq),
        }
    return out


def _date_key(d: Any) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _rank_target(alpha_today: pd.DataFrame, tradeable_items: set[str], top_pct: float | None) -> set[str]:
    if not tradeable_items:
        return set()
    if top_pct is None:
        return set(tradeable_items)
    a = alpha_today[alpha_today["market_hash_name"].isin(tradeable_items)].copy()
    a["alpha_pred"] = pd.to_numeric(a["alpha_pred"], errors="coerce")
    a = a.dropna(subset=["alpha_pred"])
    if len(a) == 0:
        return set()
    k = max(1, int(math.ceil(len(a) * float(top_pct))))
    return set(a.sort_values("alpha_pred", ascending=False).head(k)["market_hash_name"])


def _available_units(units_sold: float, participation_rate: float) -> int:
    if not np.isfinite(units_sold) or units_sold <= 0:
        return 0
    raw = int(math.floor(float(units_sold) * float(participation_rate)))
    if ALLOW_ONE_UNIT_IF_VOLUME_POSITIVE:
        return max(1, raw)
    return raw


def _price_after_market_impact(side: str, mid: float, base_cost: float, units: int, units_sold: float) -> tuple[float, float]:
    if units <= 0 or not np.isfinite(mid) or mid <= 0:
        return np.nan, 0.0
    participation = min(1.0, float(units) / max(float(units_sold), 1.0)) if units_sold and units_sold > 0 else 1.0
    impact = IMPACT_COEFF * participation
    if side == "buy":
        px = mid * (1.0 + BUY_PREMIUM_RATE + impact)
    else:
        px = mid * (1.0 - BASE_SELL_DISCOUNT_RATE - base_cost - impact)
    return float(max(px, 0.0)), float(participation)


def simulate_true_market_strategy(
    *,
    strategy: str,
    selection_mode: str,
    top_pct: float | None,
    rebalance_days: int,
    sell_cost_rate: float,
    sim_dates: pd.DatetimeIndex,
    universe_by_date: dict[pd.Timestamp, set[str]],
    alpha_by_date: dict[pd.Timestamp, pd.DataFrame],
    price_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    price_mtm: pd.DataFrame,
    initial_capital: float,
    splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    orders: list[Order] = []
    last_rebalance: pd.Timestamp | None = None

    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []

    n_rebalance_events = 0
    n_rebalance_events_with_orders = 0
    expired_units = 0
    unfilled_due_to_volume = 0
    unfilled_due_to_cash = 0
    buy_turnover = 0.0
    sell_turnover_gross_mid = 0.0
    explicit_sell_cost_paid = 0.0
    impact_paid_est = 0.0

    all_price_cols = set(price_wide.columns)

    for d in sim_dates:
        d = _date_key(d)

        # ------------------------------
        # Execute open orders, sells first.
        # ------------------------------
        fresh_orders: list[Order] = []
        for side in ("sell", "buy"):
            for order in [o for o in orders if o.side == side]:
                age = int((d - order.created_date).days)
                if age < 1:  # next-day execution, not same-day
                    fresh_orders.append(order)
                    continue
                if age > MAX_ORDER_LAG_DAYS:
                    expired_units += int(order.units_remaining)
                    order_rows.append({
                        "date": d, "created_date": order.created_date, "side": order.side,
                        "item": order.item, "units_unfilled": int(order.units_remaining),
                        "status": "expired", "reason": order.reason, "strategy": strategy,
                    })
                    continue

                if order.item not in all_price_cols or d not in price_wide.index:
                    fresh_orders.append(order)
                    continue

                mid = price_wide.at[d, order.item]
                vol = volume_wide.at[d, order.item] if order.item in volume_wide.columns else 0.0
                if pd.isna(mid) or mid <= 0 or pd.isna(vol) or vol <= 0:
                    unfilled_due_to_volume += int(order.units_remaining)
                    fresh_orders.append(order)
                    continue

                max_units = _available_units(float(vol), PARTICIPATION_RATE)
                fill_units = min(int(order.units_remaining), max_units)
                if side == "sell":
                    held_units = positions.get(order.item, 0)
                    fill_units = min(fill_units, held_units)
                    if fill_units <= 0:
                        continue
                else:
                    # cash-limited; compute a provisional impact price at max fill.
                    px_tmp, _ = _price_after_market_impact("buy", float(mid), sell_cost_rate, fill_units, float(vol))
                    affordable = int(cash // px_tmp) if np.isfinite(px_tmp) and px_tmp > 0 else 0
                    if affordable <= 0:
                        unfilled_due_to_cash += int(order.units_remaining)
                        fresh_orders.append(order)
                        continue
                    fill_units = min(fill_units, affordable)

                if fill_units <= 0:
                    fresh_orders.append(order)
                    continue

                fill_px, participation = _price_after_market_impact(side, float(mid), sell_cost_rate, fill_units, float(vol))
                if not np.isfinite(fill_px) or fill_px <= 0:
                    fresh_orders.append(order)
                    continue

                mid_value = fill_units * float(mid)
                fill_value = fill_units * fill_px
                if side == "sell":
                    cash += fill_value
                    positions[order.item] = positions.get(order.item, 0) - fill_units
                    if positions.get(order.item, 0) <= 0:
                        positions.pop(order.item, None)
                    sell_turnover_gross_mid += mid_value
                    explicit_sell_cost_paid += mid_value * sell_cost_rate
                    impact_paid_est += max(0.0, mid_value * (BASE_SELL_DISCOUNT_RATE + IMPACT_COEFF * participation))
                else:
                    cash -= fill_value
                    positions[order.item] = positions.get(order.item, 0) + fill_units
                    buy_turnover += fill_value
                    impact_paid_est += max(0.0, fill_value - mid_value)

                trade_rows.append({
                    "date": d,
                    "created_date": order.created_date,
                    "side": side,
                    "item": order.item,
                    "units": int(fill_units),
                    "mid_price_rub": float(mid),
                    "fill_price_rub": float(fill_px),
                    "gross_mid_value_rub": float(mid_value),
                    "fill_value_rub": float(fill_value),
                    "participation_share_of_daily_volume": float(participation),
                    "daily_units_sold": float(vol),
                    "cash_after_rub": float(cash),
                    "reason": order.reason,
                    "strategy": strategy,
                    "sell_cost_rate": float(sell_cost_rate),
                })

                order.units_remaining -= fill_units
                if order.units_remaining > 0:
                    fresh_orders.append(order)
                else:
                    order_rows.append({
                        "date": d, "created_date": order.created_date, "side": order.side,
                        "item": order.item, "units_unfilled": 0,
                        "status": "filled", "reason": order.reason, "strategy": strategy,
                    })
        # Preserve orders that were not processed in this side loop exactly once.
        # The side-loop appends active orders; remove accidental duplicates.
        dedup: dict[tuple, Order] = {}
        for o in fresh_orders:
            key = (o.created_date, o.side, o.item, o.reason, o.strategy)
            if key in dedup:
                dedup[key].units_remaining = max(dedup[key].units_remaining, o.units_remaining)
            else:
                dedup[key] = o
        orders = [o for o in dedup.values() if o.units_remaining > 0]

        # ------------------------------
        # Rebalance signal generation.
        # ------------------------------
        do_rebalance = (
            last_rebalance is None
            or (rebalance_days > 0 and int((d - last_rebalance).days) >= int(rebalance_days))
        )
        if do_rebalance and d in universe_by_date:
            tradeable = universe_by_date.get(d, set())
            if selection_mode == "all":
                target = set(tradeable)
            else:
                target = _rank_target(alpha_by_date.get(pd.Timestamp(d).normalize()), tradeable, top_pct)

            if target:
                n_rebalance_events += 1
                # Target unit counts use signal-day MTM prices and current equity.
                held_items = [it for it in positions if it in price_mtm.columns]
                mtm_value = 0.0
                if held_items and d in price_mtm.index:
                    row = price_mtm.loc[d, held_items]
                    mtm_value = float(sum(positions[it] * float(row[it]) for it in held_items if pd.notna(row[it]) and row[it] > 0))
                equity = cash + mtm_value
                per_item_budget = equity / max(len(target), 1)

                desired_units: dict[str, int] = {}
                if d in price_mtm.index:
                    price_row = price_mtm.loc[d, list(target & set(price_mtm.columns))]
                    for it in target:
                        if it not in price_row.index:
                            continue
                        px = price_row[it]
                        if pd.notna(px) and px > 0:
                            desired_units[it] = int(per_item_budget // float(px))

                new_orders: list[Order] = []
                # Exit items no longer targeted.
                for it, units in list(positions.items()):
                    if it not in target and units > 0:
                        new_orders.append(Order(d, "sell", it, int(units), "rebalance_exit", strategy))
                # Adjust target items toward desired units. This is a real active
                # rotation, not a soft catalog hold.
                for it, target_units in desired_units.items():
                    current_units = int(positions.get(it, 0))
                    delta = int(target_units - current_units)
                    if delta > 0:
                        new_orders.append(Order(d, "buy", it, delta, "rebalance_enter_or_topup", strategy))
                    elif delta < 0:
                        new_orders.append(Order(d, "sell", it, -delta, "rebalance_trim", strategy))

                if new_orders:
                    orders.extend(new_orders)
                    n_rebalance_events_with_orders += 1
                    for o in new_orders:
                        order_rows.append({
                            "date": d, "created_date": d, "side": o.side, "item": o.item,
                            "units_unfilled": int(o.units_remaining), "status": "created",
                            "reason": o.reason, "strategy": strategy,
                        })
                last_rebalance = d

        # ------------------------------
        # Daily equity and liquidation estimate.
        # ------------------------------
        mtm_value = 0.0
        liquidation_value = 0.0
        if positions and d in price_mtm.index:
            held_items = [it for it in positions if it in price_mtm.columns]
            if held_items:
                row = price_mtm.loc[d, held_items]
                for it in held_items:
                    px = row[it]
                    if pd.notna(px) and px > 0:
                        gross = positions[it] * float(px)
                        mtm_value += gross
                        liquidation_value += gross * max(0.0, 1.0 - sell_cost_rate - BASE_SELL_DISCOUNT_RATE)
        equity_marked = cash + mtm_value
        equity_liq = cash + liquidation_value
        equity_rows.append({
            "date": d,
            "cash_rub": float(cash),
            "market_value_rub": float(mtm_value),
            "equity_marked_rub": float(equity_marked),
            "equity_liquidation_estimate_rub": float(equity_liq),
            "n_positions": int(sum(1 for u in positions.values() if u > 0)),
            "n_open_orders": int(len(orders)),
            "open_order_units": int(sum(o.units_remaining for o in orders)),
        })

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    orders_df = pd.DataFrame(order_rows)

    if len(equity_df):
        final_marked = float(equity_df["equity_marked_rub"].iloc[-1])
        final_liq = float(equity_df["equity_liquidation_estimate_rub"].iloc[-1])
    else:
        final_marked = final_liq = initial_capital

    summary = {
        "strategy": strategy,
        "strategy_type": "passive" if selection_mode == "all" and rebalance_days == 0 else ("passive_rebalanced" if selection_mode == "all" else "alpha_rotation"),
        "selection_mode": selection_mode,
        "top_pct": top_pct,
        "rebalance_days": int(rebalance_days),
        "sell_cost_rate": float(sell_cost_rate),
        "initial_capital_rub": float(initial_capital),
        "final_equity_marked_rub": final_marked,
        "final_equity_liquidation_estimate_rub": final_liq,
        "roi_marked_pct": _safe_pct(initial_capital, final_marked),
        "roi_liquidation_estimate_pct": _safe_pct(initial_capital, final_liq),
        "sharpe_liquidation_estimate": _sharpe(equity_df["equity_liquidation_estimate_rub"]) if len(equity_df) else np.nan,
        "max_drawdown_liquidation_estimate_pct": _max_drawdown_pct(equity_df["equity_liquidation_estimate_rub"]) if len(equity_df) else np.nan,
        "n_days": int(len(equity_df)),
        "n_trades": int(len(trades_df)),
        "n_buys": int((trades_df["side"] == "buy").sum()) if len(trades_df) else 0,
        "n_sells": int((trades_df["side"] == "sell").sum()) if len(trades_df) else 0,
        "avg_n_positions": float(equity_df["n_positions"].mean()) if len(equity_df) else 0.0,
        "max_n_positions": int(equity_df["n_positions"].max()) if len(equity_df) else 0,
        "rebalance_events": int(n_rebalance_events),
        "rebalance_events_with_orders": int(n_rebalance_events_with_orders),
        "buy_turnover_rub": float(buy_turnover),
        "sell_turnover_gross_mid_rub": float(sell_turnover_gross_mid),
        "total_turnover_rub": float(buy_turnover + sell_turnover_gross_mid),
        "turnover_to_initial_capital": float((buy_turnover + sell_turnover_gross_mid) / initial_capital),
        "explicit_sell_cost_paid_rub": float(explicit_sell_cost_paid),
        "impact_paid_estimate_rub": float(impact_paid_est),
        "expired_units": int(expired_units),
        "unfilled_due_to_volume_units_events": int(unfilled_due_to_volume),
        "unfilled_due_to_cash_units_events": int(unfilled_due_to_cash),
        "end_open_order_units": int(sum(o.units_remaining for o in orders)),
        "by_split": _split_summary(equity_df, splits),
    }
    return equity_df, trades_df, orders_df, summary


def _make_strategy_specs() -> list[dict[str, Any]]:
    specs = [{
        "strategy": "buy_and_hold_all",
        "selection_mode": "all",
        "top_pct": None,
        "rebalance_days": 0,
    }]
    for rb in REBALANCE_DAYS:
        specs.append({
            "strategy": f"all_universe_rebalance_{rb}d",
            "selection_mode": "all",
            "top_pct": None,
            "rebalance_days": rb,
        })
    for rb in REBALANCE_DAYS:
        for tp in TOP_PCTS:
            specs.append({
                "strategy": f"alpha_top_{int(tp*100)}pct_rebalance_{rb}d",
                "selection_mode": "top_pct",
                "top_pct": tp,
                "rebalance_days": rb,
            })
    return specs


def _interpolate_break_even(x: list[float], y: list[float]) -> float | None:
    # x = cost, y = ROI or excess ROI. Return first crossing from >=0 to <0.
    pts = sorted([(float(a), float(b)) for a, b in zip(x, y) if np.isfinite(b)])
    if not pts:
        return None
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 == 0:
            return x0
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            if y1 == y0:
                return x0
            return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
    if pts[-1][1] >= 0:
        return None  # profitable through max tested cost
    return None


def _build_research_tables(grid: pd.DataFrame, report_dir: Path) -> dict[str, str]:
    grid_csv = report_dir / "20_true_market_strategy_grid.csv"
    grid.to_csv(grid_csv, index=False)

    rows = []
    passive = grid[grid["strategy"] == "buy_and_hold_all"][["sell_cost_rate", "roi_liquidation_estimate_pct"]].rename(columns={"roi_liquidation_estimate_pct": "passive_roi"})
    for strat, g in grid.groupby("strategy"):
        if strat == "buy_and_hold_all":
            continue
        merged = g.merge(passive, on="sell_cost_rate", how="left")
        costs = merged["sell_cost_rate"].tolist()
        roi = merged["roi_liquidation_estimate_pct"].tolist()
        excess = (merged["roi_liquidation_estimate_pct"] - merged["passive_roi"]).tolist()
        rows.append({
            "strategy": strat,
            "strategy_type": str(g["strategy_type"].iloc[0]),
            "top_pct": g["top_pct"].iloc[0],
            "rebalance_days": int(g["rebalance_days"].iloc[0]),
            "roi_0pct_cost": float(merged.loc[merged["sell_cost_rate"] == 0.0, "roi_liquidation_estimate_pct"].iloc[0]) if (merged["sell_cost_rate"] == 0.0).any() else np.nan,
            "roi_max_tested_cost": float(merged.loc[merged["sell_cost_rate"] == merged["sell_cost_rate"].max(), "roi_liquidation_estimate_pct"].iloc[0]),
            "zero_profit_break_even_cost": _interpolate_break_even(costs, roi),
            "passive_relative_break_even_cost": _interpolate_break_even(costs, excess),
            "avg_turnover_to_initial": float(g["turnover_to_initial_capital"].mean()),
            "avg_expired_units": float(g["expired_units"].mean()),
            "avg_unfilled_due_to_volume_units_events": float(g["unfilled_due_to_volume_units_events"].mean()),
        })
    be = pd.DataFrame(rows).sort_values(["roi_0pct_cost", "roi_max_tested_cost"], ascending=False)
    be_csv = report_dir / "20_true_market_break_even_table.csv"
    be.to_csv(be_csv, index=False)

    # Compact paper-style markdown tables.
    md_path = report_dir / "20_true_market_research_tables.md"
    best_by_cost = (grid.sort_values("roi_liquidation_estimate_pct", ascending=False)
                      .groupby("sell_cost_rate", as_index=False)
                      .head(1)
                      .sort_values("sell_cost_rate"))
    cols1 = ["sell_cost_rate", "strategy", "roi_liquidation_estimate_pct", "sharpe_liquidation_estimate", "max_drawdown_liquidation_estimate_pct", "turnover_to_initial_capital", "n_trades"]
    cols2 = ["strategy", "roi_0pct_cost", "roi_max_tested_cost", "zero_profit_break_even_cost", "passive_relative_break_even_cost", "avg_turnover_to_initial"]
    md = []
    md.append("# True-market friction research tables\n")
    md.append("## Table 1. Best strategy by cost\n")
    md.append(best_by_cost[cols1].to_markdown(index=False))
    md.append("\n\n## Table 2. Break-even cost estimates\n")
    md.append(be[cols2].head(20).to_markdown(index=False))
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {
        "strategy_grid_csv": str(grid_csv),
        "break_even_table_csv": str(be_csv),
        "research_tables_md": str(md_path),
    }


def _make_figures(grid: pd.DataFrame, report_dir: Path) -> list[str]:
    if not HAS_MPL:
        return []
    figs: list[str] = []
    alpha = grid[grid["strategy_type"] == "alpha_rotation"].copy()
    passive = grid[grid["strategy"] == "buy_and_hold_all"]

    # Figure 1: ROI vs cost for best few zero-cost alpha strategies + passive.
    top_strats = (alpha[alpha["sell_cost_rate"] == 0.0]
                  .sort_values("roi_liquidation_estimate_pct", ascending=False)
                  .head(6)["strategy"].tolist())
    plt.figure(figsize=(10, 6))
    for strat in ["buy_and_hold_all"] + top_strats:
        g = grid[grid["strategy"] == strat].sort_values("sell_cost_rate")
        if len(g):
            plt.plot(g["sell_cost_rate"] * 100, g["roi_liquidation_estimate_pct"], marker="o", label=strat)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Sell cost / forced-sale haircut (%)")
    plt.ylabel("ROI, liquidation-estimate basis (%)")
    plt.title("True-market friction sweep: ROI vs cost")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = report_dir / "20_fig_true_market_roi_vs_cost.png"
    plt.savefig(p, dpi=160)
    plt.close()
    figs.append(str(p))

    # Figure 2: Excess ROI vs passive.
    passive_map = passive.set_index("sell_cost_rate")["roi_liquidation_estimate_pct"].to_dict()
    plt.figure(figsize=(10, 6))
    for strat in top_strats:
        g = grid[grid["strategy"] == strat].sort_values("sell_cost_rate").copy()
        if len(g):
            g["excess"] = g.apply(lambda r: r["roi_liquidation_estimate_pct"] - passive_map.get(r["sell_cost_rate"], np.nan), axis=1)
            plt.plot(g["sell_cost_rate"] * 100, g["excess"], marker="o", label=strat)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Sell cost / forced-sale haircut (%)")
    plt.ylabel("Excess ROI vs buy-and-hold (percentage points)")
    plt.title("Does alpha rotation beat passive after true-market friction?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p = report_dir / "20_fig_true_market_excess_roi_vs_passive.png"
    plt.savefig(p, dpi=160)
    plt.close()
    figs.append(str(p))

    # Figure 3: ROI vs turnover.
    plt.figure(figsize=(8, 6))
    subset = grid[grid["sell_cost_rate"].isin([0.0, 0.05, 0.10, 0.20])]
    plt.scatter(subset["turnover_to_initial_capital"], subset["roi_liquidation_estimate_pct"])
    plt.xlabel("Turnover / initial capital")
    plt.ylabel("ROI, liquidation-estimate basis (%)")
    plt.title("Turnover explains friction fragility")
    plt.tight_layout()
    p = report_dir / "20_fig_true_market_roi_vs_turnover.png"
    plt.savefig(p, dpi=160)
    plt.close()
    figs.append(str(p))

    # Figure 4: Fill failure vs ROI.
    plt.figure(figsize=(8, 6))
    fail = subset["expired_units"] + subset["unfilled_due_to_volume_units_events"]
    plt.scatter(fail, subset["roi_liquidation_estimate_pct"])
    plt.xlabel("Expired + volume-unfilled unit-events")
    plt.ylabel("ROI, liquidation-estimate basis (%)")
    plt.title("Execution capacity diagnostic")
    plt.tight_layout()
    p = report_dir / "20_fig_true_market_capacity_vs_roi.png"
    plt.savefig(p, dpi=160)
    plt.close()
    figs.append(str(p))

    return figs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--store-details", action="store_true", help="Save equity/trade/order parquet for every grid run. Default saves only selected runs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    initial_capital = float(cfg.get("research", {}).get("initial_capital_rub", INITIAL_CAPITAL_DEFAULT))

    split_min = min(pd.Timestamp(splits["train_start"]), pd.Timestamp(splits["val_start"]), pd.Timestamp(splits["test_start"])).normalize()
    split_max = pd.Timestamp(splits["test_end"]).normalize()

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    models_dir = resolve_path(cfg, "models_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    backtests_dir = resolve_path(cfg, "backtests_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    print("Loading feature store...")
    feat_cols = ["date", "market_hash_name", "price_rub", "units_sold"]
    feat = pd.read_parquet(feature_path, columns=feat_cols, filters=[("date", ">=", split_min), ("date", "<=", split_max)])
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat = feat.drop_duplicates(["date", "market_hash_name"], keep="last")
    feat["price_rub"] = pd.to_numeric(feat["price_rub"], errors="coerce")
    feat["units_sold"] = pd.to_numeric(feat["units_sold"], errors="coerce").fillna(0.0)

    print("Loading universe...")
    uni = pd.read_parquet(universe_path, columns=["date", "market_hash_name", "tradeable_today"], filters=[("date", ">=", split_min), ("date", "<=", split_max)])
    uni["date"] = pd.to_datetime(uni["date"]).dt.normalize()
    uni = uni.drop_duplicates(["date", "market_hash_name"], keep="last")
    uni = uni[uni["tradeable_today"].fillna(False).astype(bool)]
    universe_by_date = {d: set(g["market_hash_name"]) for d, g in uni.groupby("date", sort=True)}

    print("Loading alpha predictions...")
    alpha_path = models_dir / "alpha_predictions.parquet"
    alpha = pd.read_parquet(alpha_path, columns=["date", "market_hash_name", "alpha_pred"], filters=[("date", ">=", split_min), ("date", "<=", split_max)])
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha = alpha.drop_duplicates(["date", "market_hash_name"], keep="last")
    alpha_by_date = {d: g[["market_hash_name", "alpha_pred"]].copy() for d, g in alpha.groupby("date", sort=True)}

    # price_wide: ffilled feature price; volume_wide: execution availability.
    print("Building wide price/volume matrices...")
    price_wide = feat.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    volume_wide = feat.pivot(index="date", columns="market_hash_name", values="units_sold").sort_index().fillna(0.0)
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & set(universe_by_date.keys())))
    sim_dates = sim_dates[(sim_dates >= split_min) & (sim_dates <= split_max)]
    price_wide = price_wide.reindex(sim_dates)
    volume_wide = volume_wide.reindex(sim_dates).fillna(0.0)
    price_mtm = price_wide.ffill()

    specs = _make_strategy_specs()
    summaries: list[dict[str, Any]] = []
    strategies_by_cost: dict[str, dict[str, Any]] = {}

    print(f"Running {len(COSTS)} costs x {len(specs)} strategies = {len(COSTS) * len(specs)} simulations...")
    for cost in COSTS:
        cost_key = f"cost_{int(round(cost * 10000)):04d}bp"
        strategies_by_cost[cost_key] = {}
        for spec in specs:
            eq, tr, od, summ = simulate_true_market_strategy(
                strategy=spec["strategy"],
                selection_mode=spec["selection_mode"],
                top_pct=spec["top_pct"],
                rebalance_days=spec["rebalance_days"],
                sell_cost_rate=cost,
                sim_dates=sim_dates,
                universe_by_date=universe_by_date,
                alpha_by_date=alpha_by_date,
                price_wide=price_wide,
                volume_wide=volume_wide,
                price_mtm=price_mtm,
                initial_capital=initial_capital,
                splits=splits,
            )
            summaries.append(summ)
            strategies_by_cost[cost_key][spec["strategy"]] = summ
            # Store full artifacts for all runs only when requested; otherwise store
            # the most important baseline + best-known alpha rotation candidates.
            important = spec["strategy"] in {"buy_and_hold_all", "alpha_top_5pct_rebalance_30d", "alpha_top_10pct_rebalance_30d", "alpha_top_5pct_rebalance_180d", "alpha_top_10pct_rebalance_180d"}
            if args.store_details or important:
                suffix = f"{spec['strategy']}_cost_{int(round(cost*10000)):04d}bp"
                save_table(eq, backtests_dir / f"20_true_market_{suffix}_equity.parquet")
                save_table(tr, backtests_dir / f"20_true_market_{suffix}_trades.parquet")
                save_table(od, backtests_dir / f"20_true_market_{suffix}_orders.parquet")

    grid = pd.DataFrame(summaries)
    table_paths = _build_research_tables(grid, reports_dir)
    figures = _make_figures(grid, reports_dir)

    # Best strategy diagnostics.
    zero = grid[grid["sell_cost_rate"] == 0.0].sort_values("roi_liquidation_estimate_pct", ascending=False)
    max_cost = max(COSTS)
    best_at_max = grid[grid["sell_cost_rate"] == max_cost].sort_values("roi_liquidation_estimate_pct", ascending=False).head(1)
    best_alpha_zero = zero[zero["strategy_type"] == "alpha_rotation"].head(1)

    report = {
        "config": {
            "initial_capital_rub": initial_capital,
            "cost_model": "true_market_volume_capped_execution",
            "costs": COSTS,
            "rebalance_days": REBALANCE_DAYS,
            "top_pcts": TOP_PCTS,
            "buy_premium_rate": BUY_PREMIUM_RATE,
            "base_sell_discount_rate": BASE_SELL_DISCOUNT_RATE,
            "participation_rate": PARTICIPATION_RATE,
            "impact_coeff": IMPACT_COEFF,
            "max_order_lag_days": MAX_ORDER_LAG_DAYS,
            "allow_one_unit_if_volume_positive": ALLOW_ONE_UNIT_IF_VOLUME_POSITIVE,
            "sim_start": str(sim_dates.min().date()) if len(sim_dates) else None,
            "sim_end": str(sim_dates.max().date()) if len(sim_dates) else None,
            "n_sim_dates": int(len(sim_dates)),
            "main_roi_field": "roi_liquidation_estimate_pct",
        },
        "best_zero_cost_overall_strategy": zero.iloc[0].to_dict() if len(zero) else {},
        "best_zero_cost_alpha_strategy": best_alpha_zero.iloc[0].to_dict() if len(best_alpha_zero) else {},
        "best_max_cost_strategy": best_at_max.iloc[0].to_dict() if len(best_at_max) else {},
        "figures": figures,
        "tables": table_paths,
        "strategies_by_cost": strategies_by_cost,
    }
    write_json(reports_dir / "20_true_market_summary.json", report)

    print("Saved:")
    print(f"  {reports_dir / '20_true_market_summary.json'}")
    for k, v in table_paths.items():
        print(f"  {k}: {v}")
    for f in figures:
        print(f"  figure: {f}")


if __name__ == "__main__":
    main()
