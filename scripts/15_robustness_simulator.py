#!/usr/bin/env python
"""
Buy-and-hold robustness simulator.

Tests whether the +131% result from a single 2023-10-01 cohort is a robust
property of buying the CS2 tradeable universe, or vintage-specific to that
particular start date.

Method:
  - Sweep candidate start dates every 30 days through the sample.
  - For each (start_date, horizon_mode) pair, simulate full portfolio
    buy-and-hold with realistic capital constraints (50K RUB initial,
    integer units, equal cash budget per item, 10% final sell cost).
  - Aggregate the distribution of outcomes across vintages.

Three horizon modes:
  - 365d   : exactly 1 year forward from start_date
  - 730d   : exactly 2 years forward from start_date (fewer feasible
             starts since end date must fall within data range)
  - to_end : variable horizon, hold to data_max (2026-02-28). The
             2023-10-01 vintage here reproduces the baseline +131%.

What "robust" looks like in the output:
  - share_positive >= 0.75 at 365d : most vintages end in the green net
  - p10 >= 0%                       : even bottom-decile vintages don't lose
  - p25 >= 0%                       : the worst quartile is at least flat
If any of these fail, buy-and-hold is genuinely vintage-dependent and
forward expected return is materially worse than the headline.

Run:
  python scripts/15_robustness_simulator.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  data/processed/reports/15_robustness_summary.json
  data/processed/reports/15_robustness_vintages.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.utils import save_table, write_json


START_INTERVAL_DAYS = 30
HORIZON_MODES: dict[str, int | None] = {
    "365d": 365,
    "730d": 730,
    "to_end": None,
}


def simulate_vintage(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    universe_by_date: dict[pd.Timestamp, set[str]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    sell_cost_rate: float,
) -> dict | None:
    """Buy-and-hold from start_date to end_date. Returns vintage summary
    or None if buy infeasible (e.g., no universe data on start_date)."""
    target = universe_by_date.get(start_date)
    if not target:
        return None

    cols = [it for it in target if it in price_wide.columns]
    if not cols:
        return None

    start_row = price_wide.loc[start_date, cols]
    valid_mask = start_row.notna() & (start_row > 0)
    start_prices = start_row[valid_mask].astype(float)
    if len(start_prices) == 0:
        return None

    cash = float(initial_capital)
    positions: dict[str, int] = {}
    per_item_budget = cash / len(start_prices)
    for it, px in start_prices.items():
        units = int(per_item_budget // float(px))
        if units < 1:
            continue
        cost = units * float(px)
        if cost > cash:
            continue
        cash -= cost
        positions[it] = units

    if not positions:
        return None

    # Daily equity curve over the hold window (vectorized).
    item_list = list(positions.keys())
    units_arr = np.array([positions[it] for it in item_list], dtype=float)
    hold_mask = (price_wide_mtm.index >= start_date) & (price_wide_mtm.index <= end_date)
    hold_dates = price_wide_mtm.index[hold_mask]
    price_window = price_wide_mtm.loc[hold_mask, item_list].fillna(0.0).astype(float)
    mv_per_day = price_window.values @ units_arr  # (n_days,)
    equity_curve = cash + mv_per_day

    if len(equity_curve) == 0:
        return None

    final_mv = float(mv_per_day[-1])
    final_equity_gross = cash + final_mv
    final_equity_net = cash + final_mv * (1.0 - sell_cost_rate)

    running_max = np.maximum.accumulate(equity_curve)
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdown = (equity_curve - running_max) / safe_max
    max_dd_pct = float(drawdown.min() * 100.0)

    daily_returns = np.diff(equity_curve) / np.where(equity_curve[:-1] > 0, equity_curve[:-1], 1.0)
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) > 1 and float(np.std(daily_returns)) > 0:
        sharpe = float(np.sqrt(365) * np.mean(daily_returns) / np.std(daily_returns))
    else:
        sharpe = None

    hold_days = int((end_date - start_date).days)
    roi_gross = (final_equity_gross / initial_capital - 1.0)
    roi_net = (final_equity_net / initial_capital - 1.0)
    hold_years = hold_days / 365.25
    if hold_years > 0 and roi_net > -1.0:
        annualized_roi_net = (1.0 + roi_net) ** (1.0 / hold_years) - 1.0
    else:
        annualized_roi_net = None

    return {
        "start_date": start_date,
        "end_date": end_date,
        "hold_days": hold_days,
        "n_items_bought": int(len(positions)),
        "initial_capital_rub": float(initial_capital),
        "cash_drag_share": float(cash / initial_capital),
        "final_equity_gross_rub": float(final_equity_gross),
        "final_equity_net_rub": float(final_equity_net),
        "roi_gross_pct": float(roi_gross * 100.0),
        "roi_net_pct": float(roi_net * 100.0),
        "annualized_roi_net_pct": float(annualized_roi_net * 100.0) if annualized_roi_net is not None else None,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_gross": sharpe,
    }


def summarize_distribution(vintages: list[dict]) -> dict:
    if not vintages:
        return {"n_vintages": 0}
    df = pd.DataFrame(vintages)
    rois_net = df["roi_net_pct"]
    rois_gross = df["roi_gross_pct"]
    return {
        "n_vintages": int(len(df)),
        "hold_days_mean": float(df["hold_days"].mean()),
        "hold_days_min": int(df["hold_days"].min()),
        "hold_days_max": int(df["hold_days"].max()),
        "roi_net_pct": {
            "mean": float(rois_net.mean()),
            "median": float(rois_net.median()),
            "std": float(rois_net.std()) if len(rois_net) > 1 else None,
            "min": float(rois_net.min()),
            "p10": float(rois_net.quantile(0.10)),
            "p25": float(rois_net.quantile(0.25)),
            "p75": float(rois_net.quantile(0.75)),
            "p90": float(rois_net.quantile(0.90)),
            "max": float(rois_net.max()),
            "share_positive": float((rois_net > 0).mean()),
        },
        "roi_gross_pct": {
            "mean": float(rois_gross.mean()),
            "median": float(rois_gross.median()),
            "share_positive": float((rois_gross > 0).mean()),
        },
        "annualized_roi_net_pct": {
            "mean": float(df["annualized_roi_net_pct"].dropna().mean()) if df["annualized_roi_net_pct"].notna().any() else None,
            "median": float(df["annualized_roi_net_pct"].dropna().median()) if df["annualized_roi_net_pct"].notna().any() else None,
        },
        "max_drawdown_pct": {
            "mean": float(df["max_drawdown_pct"].mean()),
            "median": float(df["max_drawdown_pct"].median()),
            "worst": float(df["max_drawdown_pct"].min()),
        },
        "cash_drag_share_mean": float(df["cash_drag_share"].mean()),
        "n_items_bought_mean": float(df["n_items_bought"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sell_cost = float(cfg["research"]["sell_cost_rate"])
    initial_capital = float(cfg["research"]["initial_capital_rub"])

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading universe...")
    universe_df = pd.read_parquet(
        universe_path, columns=["date", "market_hash_name", "tradeable_today"]
    )
    universe_df["date"] = pd.to_datetime(universe_df["date"]).dt.normalize()
    universe_df["tradeable_today"] = universe_df["tradeable_today"].fillna(False).astype(bool)
    universe_df = universe_df[universe_df["tradeable_today"]]
    universe_by_date = {
        d: set(g["market_hash_name"].tolist())
        for d, g in universe_df.groupby("date", sort=False)
    }

    print("Loading price panel...")
    price_panel = pd.read_parquet(
        feature_path, columns=["date", "market_hash_name", "price_rub"]
    )
    price_panel["date"] = pd.to_datetime(price_panel["date"]).dt.normalize()
    price_panel["price_rub"] = pd.to_numeric(price_panel["price_rub"], errors="coerce").astype("float32")
    price_panel = price_panel.dropna(subset=["price_rub"])
    price_panel = price_panel.drop_duplicates(subset=["date", "market_hash_name"], keep="last")
    price_wide = price_panel.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    price_wide_mtm = price_wide.ffill()
    data_min = price_wide.index.min()
    data_max = price_wide.index.max()
    print(f"  {len(price_wide.columns)} items, {data_min.date()} to {data_max.date()}")

    # Generate candidate start dates every 30 days, snapped to actual data dates.
    candidate_starts: list[pd.Timestamp] = []
    cursor = data_min
    while cursor <= data_max:
        forward_mask = price_wide.index >= cursor
        if forward_mask.any():
            snapped = price_wide.index[forward_mask][0]
            if (snapped <= data_max
                    and snapped in universe_by_date
                    and (not candidate_starts or snapped != candidate_starts[-1])):
                candidate_starts.append(snapped)
        cursor = cursor + pd.Timedelta(days=START_INTERVAL_DAYS)
    print(f"Candidate start dates: {len(candidate_starts)}")

    results: dict = {
        "config": {
            "initial_capital_rub": initial_capital,
            "sell_cost_rate": sell_cost,
            "start_interval_days": START_INTERVAL_DAYS,
            "horizon_modes": list(HORIZON_MODES.keys()),
        },
        "data_range": {
            "start": str(data_min.date()),
            "end": str(data_max.date()),
        },
        "by_horizon_mode": {},
    }
    all_vintages: list[dict] = []

    for mode_name, horizon_days in HORIZON_MODES.items():
        print(f"\n=== {mode_name} ===")
        vintages: list[dict] = []
        for start in candidate_starts:
            if horizon_days is None:
                end = data_max
            else:
                target_end = start + pd.Timedelta(days=horizon_days)
                if target_end > data_max:
                    continue
                # Snap to nearest data date <= target_end
                back_mask = price_wide.index <= target_end
                if not back_mask.any():
                    continue
                end = price_wide.index[back_mask][-1]
            v = simulate_vintage(
                price_wide=price_wide,
                price_wide_mtm=price_wide_mtm,
                universe_by_date=universe_by_date,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
                sell_cost_rate=sell_cost,
            )
            if v is None:
                continue
            v = dict(v)
            v["mode"] = mode_name
            vintages.append(v)
            all_vintages.append(v)

        summary = summarize_distribution(vintages)
        results["by_horizon_mode"][mode_name] = summary

        if summary.get("n_vintages", 0) > 0:
            rn = summary["roi_net_pct"]
            ann = summary["annualized_roi_net_pct"]
            dd = summary["max_drawdown_pct"]
            print(f"  vintages: {summary['n_vintages']}, "
                  f"hold_days {summary['hold_days_min']}-{summary['hold_days_max']}")
            print(f"  ROI net: mean {rn['mean']:+.1f}%, median {rn['median']:+.1f}%")
            print(f"           p10 {rn['p10']:+.1f}%, p25 {rn['p25']:+.1f}%, "
                  f"p75 {rn['p75']:+.1f}%, p90 {rn['p90']:+.1f}%")
            print(f"           worst {rn['min']:+.1f}%, best {rn['max']:+.1f}%")
            print(f"  share_positive: {rn['share_positive']:.1%}")
            if ann["mean"] is not None:
                print(f"  annualized ROI net: mean {ann['mean']:+.1f}%, "
                      f"median {ann['median']:+.1f}%")
            print(f"  max drawdown: mean {dd['mean']:.1f}%, worst {dd['worst']:.1f}%")
            print(f"  cash drag: {summary['cash_drag_share_mean']:.1%} of capital sat idle on avg")
        else:
            print("  no feasible vintages")

    # Comparison table
    print("\n" + "=" * 120)
    print(f"{'Mode':<8} | {'N':>3} | {'Mean ROI net':>13} | {'Median':>8} | "
          f"{'p10':>8} | {'p25':>8} | {'p75':>8} | {'p90':>8} | {'Worst':>8} | "
          f"{'Share +':>8} | {'Ann mean':>9}")
    print("-" * 120)
    for mode_name in HORIZON_MODES:
        s = results["by_horizon_mode"].get(mode_name, {})
        if s.get("n_vintages", 0) == 0:
            print(f"{mode_name:<8} | {'-':>3} | (no vintages)")
            continue
        rn = s["roi_net_pct"]
        ann = s["annualized_roi_net_pct"]
        ann_str = f"{ann['mean']:+.1f}%" if ann["mean"] is not None else "-"
        print(
            f"{mode_name:<8} | {s['n_vintages']:>3} | "
            f"{rn['mean']:>+12.1f}% | {rn['median']:>+7.1f}% | "
            f"{rn['p10']:>+7.1f}% | {rn['p25']:>+7.1f}% | "
            f"{rn['p75']:>+7.1f}% | {rn['p90']:>+7.1f}% | "
            f"{rn['min']:>+7.1f}% | {rn['share_positive']:>7.1%} | {ann_str:>9}"
        )

    if all_vintages:
        df_all = pd.DataFrame(all_vintages)
        save_table(df_all, reports_dir / "15_robustness_vintages.parquet")
        print(f"\nSaved per-vintage detail to {reports_dir / '15_robustness_vintages.parquet'}")

    write_json(reports_dir / "15_robustness_summary.json", results)
    print(f"Saved summary to {reports_dir / '15_robustness_summary.json'}")


if __name__ == "__main__":
    main()
