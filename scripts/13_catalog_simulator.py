#!/usr/bin/env python
"""
Catalog strategy simulator: long-hold equal-weighted basket of the tradeable
CS2 universe with periodic rebalancing.

Tests four rebalance frequencies, with and without AI-driven exclusion of
the bottom 20% by alpha_pred at each rebalance:

  - buy_and_hold:  initial buy on first day, never rebalance
  - annual:        rebalance every 365 days
  - semi_annual:   rebalance every 180 days
  - quarterly:     rebalance every  90 days

Rebalance semantics (soft rebalance):
  - Items leaving the target set: SELL entirely (10% sell cost).
  - Items entering the target: BUY using available cash, split equally
    across new entries.
  - Items still in target: HOLD as-is. No reweighting to equal-weight.
    Winners are allowed to run; we don't pay 10% to trim them.

Exclusion variants:
  - At each rebalance, drop the bottom 20% by alpha_pred from the target.
  - Items without an alpha prediction are treated as neutral (not excluded).
  - Requires models/alpha_predictions.parquet. Skipped if missing.

Run:
  python scripts/13_catalog_simulator.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  data/processed/reports/13_catalog_sim_summary.json
  backtests/catalog_<config>_equity.parquet
  backtests/catalog_<config>_trades.parquet
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


def _select_target_set(
    universe_items: set[str],
    alpha_lookup: dict | None,
    date: pd.Timestamp,
    exclude_bottom_pct: float | None,
) -> set[str]:
    if not universe_items:
        return set()
    if exclude_bottom_pct is None or alpha_lookup is None:
        return set(universe_items)
    # Score items; items without alpha_pred are treated as neutral (0.0).
    scored = [(it, alpha_lookup.get((date, it), 0.0)) for it in universe_items]
    scored.sort(key=lambda x: x[1])  # ascending
    cutoff = int(np.ceil(len(scored) * exclude_bottom_pct))
    return {it for it, _ in scored[cutoff:]}


def simulate_catalog(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    universe_by_date: dict[pd.Timestamp, set[str]],
    alpha_lookup: dict | None,
    sim_dates: pd.DatetimeIndex,
    rebalance_days: int,
    initial_capital: float,
    sell_cost_rate: float,
    exclude_bottom_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate a catalog strategy over sim_dates.

    price_wide:     date x item, prices on actual quote days (ffill'd within
                    each item's active range by features.py).
    price_wide_mtm: same, but additionally ffill'd across columns so any
                    date inside price_wide.index has a last-known price for
                    every item that has ever quoted. Used for mark-to-market
                    on inactive holdings.
    rebalance_days: 0 means buy-and-hold (single initial buy, no further
                    trading until simulation end).
    """
    price_cols = set(price_wide.columns)

    cash = float(initial_capital)
    positions: dict[str, dict] = {}  # item -> {units, entry_price, entry_date}
    equity_rows: list[dict] = []
    trades_rows: list[dict] = []
    last_rebalance: pd.Timestamp | None = None

    for d in sim_dates:
        # Mark-to-market with cross-column ffill so dormant holdings keep
        # a sensible (last-known) carrying value.
        mv = 0.0
        if positions:
            held_items = [it for it in positions if it in price_cols]
            if held_items:
                row = price_wide_mtm.loc[d, held_items]
                for it in held_items:
                    px = row[it]
                    if pd.notna(px) and px > 0:
                        mv += positions[it]["units"] * float(px)
        equity = cash + mv
        equity_rows.append({
            "date": d,
            "cash_rub": cash,
            "market_value_rub": mv,
            "equity_rub": equity,
            "n_positions": len(positions),
        })

        do_rebalance = (
            last_rebalance is None
            or (rebalance_days > 0 and (d - last_rebalance).days >= rebalance_days)
        )
        if not do_rebalance:
            continue

        target = _select_target_set(
            universe_by_date.get(d, set()),
            alpha_lookup,
            d,
            exclude_bottom_pct,
        )
        if not target and last_rebalance is None:
            # No universe data on first day; defer initial buy to next day.
            continue

        # Prices available today (real quote, not just ffilled).
        today_prices: dict[str, float] = {}
        cols_to_check = list((target | set(positions.keys())) & price_cols)
        if cols_to_check:
            row = price_wide.loc[d, cols_to_check]
            for it in cols_to_check:
                px = row[it]
                if pd.notna(px) and px > 0:
                    today_prices[it] = float(px)

        # SELL items leaving the target set.
        for it in list(positions.keys()):
            if it in target:
                continue
            px = today_prices.get(it)
            if px is None and it in price_cols:
                # Fall back to last-known price (item went dormant).
                last_px = price_wide_mtm.at[d, it]
                if pd.notna(last_px) and last_px > 0:
                    px = float(last_px)
            if px is None or px <= 0:
                # No price at all; write off the position to 0.
                trades_rows.append({
                    "date": d, "side": "sell", "item": it,
                    "units": int(positions[it]["units"]),
                    "price": 0.0, "proceeds_rub": 0.0, "cash_after": cash,
                    "reason": "no_price_writeoff",
                })
                del positions[it]
                continue
            proceeds = positions[it]["units"] * px * (1.0 - sell_cost_rate)
            cash += proceeds
            trades_rows.append({
                "date": d, "side": "sell", "item": it,
                "units": int(positions[it]["units"]),
                "price": px, "proceeds_rub": float(proceeds), "cash_after": cash,
                "reason": "left_target",
            })
            del positions[it]

        # BUY items entering the target set. Hold existing in-target positions
        # as-is (no reweighting).
        to_buy = [it for it in target if it not in positions and it in today_prices]
        if to_buy and cash > 0:
            per_item_budget = cash / len(to_buy)
            for it in to_buy:
                px = today_prices[it]
                units = int(per_item_budget // px)
                if units < 1:
                    continue
                cost = units * px
                if cost > cash:
                    continue
                cash -= cost
                positions[it] = {
                    "units": units,
                    "entry_price": px,
                    "entry_date": d,
                }
                trades_rows.append({
                    "date": d, "side": "buy", "item": it,
                    "units": int(units), "price": px, "cost_rub": float(cost),
                    "cash_after": cash, "reason": "entered_target",
                })

        last_rebalance = d

    # Final mark
    last_day = sim_dates[-1] if len(sim_dates) else None
    final_mv_gross = 0.0
    if last_day is not None and positions:
        held_items = [it for it in positions if it in price_cols]
        if held_items:
            row = price_wide_mtm.loc[last_day, held_items]
            for it in held_items:
                px = row[it]
                if pd.notna(px) and px > 0:
                    final_mv_gross += positions[it]["units"] * float(px)
    final_equity_gross = cash + final_mv_gross
    final_equity_net = cash + final_mv_gross * (1.0 - sell_cost_rate)

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades_rows)

    if len(equity_df) > 1:
        eq = equity_df["equity_rub"].astype(float)
        daily_ret = eq.pct_change().dropna()
        sharpe = (
            float(np.sqrt(365) * daily_ret.mean() / daily_ret.std())
            if daily_ret.std() > 0 else None
        )
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max.replace(0, np.nan)
        max_dd_pct = float(drawdown.min() * 100.0) if drawdown.notna().any() else 0.0
    else:
        sharpe = None
        max_dd_pct = 0.0

    summary = {
        "rebalance_days": int(rebalance_days),
        "exclude_bottom_pct": exclude_bottom_pct,
        "initial_capital_rub": float(initial_capital),
        "final_equity_gross_rub": float(final_equity_gross),
        "final_equity_net_rub": float(final_equity_net),
        "roi_gross_pct": float((final_equity_gross / initial_capital - 1.0) * 100.0),
        "roi_net_pct": float((final_equity_net / initial_capital - 1.0) * 100.0),
        "sharpe_gross": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "n_trades": int(len(trades_df)),
        "n_buys": int((trades_df["side"] == "buy").sum()) if len(trades_df) else 0,
        "n_sells": int((trades_df["side"] == "sell").sum()) if len(trades_df) else 0,
        "n_days": int(len(equity_df)),
        "avg_n_positions": float(equity_df["n_positions"].mean()) if len(equity_df) else 0.0,
        "max_n_positions": int(equity_df["n_positions"].max()) if len(equity_df) else 0,
    }
    return equity_df, trades_df, summary


def per_split_roi(equity_df: pd.DataFrame, splits: dict) -> dict:
    out: dict = {}
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
            "roi_pct": float((end_eq / start_eq - 1.0) * 100.0) if start_eq > 0 else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    sell_cost = float(cfg["research"]["sell_cost_rate"])
    initial_capital = float(cfg["research"]["initial_capital_rub"])

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    models_dir = resolve_path(cfg, "models_dir")
    alpha_path = models_dir / "alpha_predictions.parquet"
    reports_dir = resolve_path(cfg, "reports_dir")
    backtests_dir = resolve_path(cfg, "backtests_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

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
    price_wide = price_panel.pivot(
        index="date", columns="market_hash_name", values="price_rub"
    ).sort_index()
    price_wide_mtm = price_wide.ffill()
    print(f"  Price grid: {price_wide.index.min().date()} to {price_wide.index.max().date()}, "
          f"{len(price_wide.columns)} items")

    alpha_lookup: dict | None = None
    if alpha_path.exists():
        print("Loading alpha predictions...")
        alpha_df = pd.read_parquet(
            alpha_path, columns=["date", "market_hash_name", "alpha_pred"]
        )
        alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
        alpha_lookup = {
            (r.date, r.market_hash_name): float(r.alpha_pred)
            for r in alpha_df.itertuples(index=False)
        }
        print(f"  Loaded {len(alpha_lookup):,} alpha predictions")
    else:
        print("  No alpha_predictions.parquet found; exclusion variants will be skipped")

    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & set(universe_by_date.keys())))
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    print(f"Simulating {len(sim_dates)} days from {sim_dates[0].date()} to {sim_dates[-1].date()}")

    configs: list[tuple[str, int, float | None]] = [
        ("buy_and_hold", 0, None),
        ("quarterly", 90, None),
        ("semi_annual", 180, None),
        ("annual", 365, None),
    ]
    if alpha_lookup is not None:
        configs.extend([
            ("buy_and_hold_excl20", 0, 0.20),
            ("quarterly_excl20", 90, 0.20),
            ("semi_annual_excl20", 180, 0.20),
            ("annual_excl20", 365, 0.20),
        ])

    all_summaries: dict = {}

    for name, rebalance_days, exclude_pct in configs:
        print(f"\n=== {name} ===")
        equity_df, trades_df, summary = simulate_catalog(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            universe_by_date=universe_by_date,
            alpha_lookup=alpha_lookup,
            sim_dates=sim_dates,
            rebalance_days=rebalance_days,
            initial_capital=initial_capital,
            sell_cost_rate=sell_cost,
            exclude_bottom_pct=exclude_pct,
        )
        summary["by_split"] = per_split_roi(equity_df, splits)
        all_summaries[name] = summary

        save_table(equity_df, backtests_dir / f"catalog_{name}_equity.parquet")
        save_table(trades_df, backtests_dir / f"catalog_{name}_trades.parquet")

        print(f"  final equity: gross={summary['final_equity_gross_rub']:,.0f} "
              f"net={summary['final_equity_net_rub']:,.0f}")
        print(f"  ROI: gross={summary['roi_gross_pct']:+.1f}% "
              f"net={summary['roi_net_pct']:+.1f}%")
        print(f"  trades: {summary['n_trades']} "
              f"({summary['n_buys']} buys, {summary['n_sells']} sells)")
        print(f"  avg positions: {summary['avg_n_positions']:.1f}, "
              f"max drawdown: {summary['max_drawdown_pct']:.1f}%")

    # Comparison table
    print("\n" + "=" * 110)
    print(f"{'Config':<22} | {'Trades':>6} | {'ROI gross':>10} | {'ROI net':>10} | "
          f"{'Train ROI':>10} | {'Val ROI':>9} | {'Test ROI':>9} | {'Max DD':>7}")
    print("-" * 110)
    for name, s in all_summaries.items():
        bs = s["by_split"]
        tr = bs.get("train", {}).get("roi_pct")
        va = bs.get("val", {}).get("roi_pct")
        te = bs.get("test", {}).get("roi_pct")
        print(
            f"{name:<22} | {s['n_trades']:>6} | "
            f"{s['roi_gross_pct']:>+9.1f}% | {s['roi_net_pct']:>+9.1f}% | "
            f"{(f'{tr:+.1f}%' if tr is not None else '-'):>10} | "
            f"{(f'{va:+.1f}%' if va is not None else '-'):>9} | "
            f"{(f'{te:+.1f}%' if te is not None else '-'):>9} | "
            f"{s['max_drawdown_pct']:>+6.1f}%"
        )

    write_json(reports_dir / "13_catalog_sim_summary.json", all_summaries)
    print(f"\nSaved summary to {reports_dir / '13_catalog_sim_summary.json'}")


if __name__ == "__main__":
    main()
