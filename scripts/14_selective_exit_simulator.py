#!/usr/bin/env python
"""
Selective-exit simulator: buy-and-hold the day-zero cohort, then use the
existing alpha model to selectively exit positions whose forward outlook
turns strongly negative.

Tests a sweep of exit thresholds:
  - buy_and_hold:  no exits ever (sanity reproduction of script 13's baseline)
  - exit_lt_-0.3, -0.5, -0.7, -0.9: exit any held position whose alpha_pred
    falls below the threshold, subject to a 7-day minimum hold.

Semantics:
  - Initial buy on first sim day: equal-weighted across the day's
    tradeable universe (same as script 13's buy_and_hold).
  - Daily, for each held position age >= 7 days: check alpha_pred for
    (today, item). If alpha_pred < exit_threshold, sell at today's price
    less 10% sell cost.
  - No buybacks. Cash from sells accumulates idle.
  - Items that exit the universe but have no alpha signal: hold (no
    auto-sell on universe departure).

Decision-quality diagnostic:
  For each sell, compare cash received to what the position would have
  been worth at sim end (held + sold at 10% cost). This isolates the
  model's exit-signal value from the strategy's overall performance.

Why this might work where rotation didn't:
  Selling pays 10%. For a sell to pay for itself, the model needs to
  identify a forward loss > 10%. The bottom-head of the alpha model
  has test AUC 0.61 at picking 30d-forward losers. If items with
  strongly negative alpha_pred have expected forward losses well past
  10%, selective exits beat holding.

Run:
  python scripts/14_selective_exit_simulator.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  data/processed/reports/14_selective_exit_summary.json
  backtests/selective_exit_<config>_equity.parquet
  backtests/selective_exit_<config>_trades.parquet
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


def simulate_selective_exit(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    universe_by_date: dict[pd.Timestamp, set[str]],
    alpha_lookup: dict,
    sim_dates: pd.DatetimeIndex,
    exit_threshold: float | None,
    min_holding_days: int,
    initial_capital: float,
    sell_cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """exit_threshold=None means buy-and-hold (never exit)."""
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_rows: list[dict] = []
    trades_rows: list[dict] = []
    initial_purchase_done = False

    for d in sim_dates:
        # Mark to market on last-known prices.
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

        # Initial buy: full universe equal-weighted on first day with data.
        if not initial_purchase_done:
            target = universe_by_date.get(d, set())
            if not target:
                continue
            cols = [it for it in target if it in price_cols]
            if not cols:
                continue
            row = price_wide.loc[d, cols]
            today_prices = {it: float(row[it]) for it in cols
                            if pd.notna(row[it]) and row[it] > 0}
            if not today_prices:
                continue
            per_item_budget = cash / len(today_prices)
            for it, px in today_prices.items():
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
                    "units": int(units), "price": px,
                    "cost_rub": float(cost), "cash_after": cash,
                    "reason": "initial_buy", "alpha_pred_at_exit": np.nan,
                    "days_held": 0,
                })
            initial_purchase_done = True
            continue

        # Pure buy-and-hold: skip all exit logic.
        if exit_threshold is None:
            continue

        # Check each holding for exit signal.
        for it in list(positions.keys()):
            pos = positions[it]
            days_held = (d - pos["entry_date"]).days
            if days_held < min_holding_days:
                continue
            alpha = alpha_lookup.get((d, it))
            if alpha is None:
                # No signal -> can't decide, hold.
                continue
            if alpha >= exit_threshold:
                continue
            # Exit this position. Prefer today's quoted price; fall back to
            # last-known if the item is dormant today.
            if it not in price_cols:
                continue
            px_today = price_wide.at[d, it]
            if pd.notna(px_today) and px_today > 0:
                px = float(px_today)
            else:
                px_mtm = price_wide_mtm.at[d, it]
                if pd.isna(px_mtm) or px_mtm <= 0:
                    continue
                px = float(px_mtm)
            proceeds = pos["units"] * px * (1.0 - sell_cost_rate)
            cash += proceeds
            trades_rows.append({
                "date": d, "side": "sell", "item": it,
                "units": int(pos["units"]), "price": px,
                "proceeds_rub": float(proceeds), "cash_after": cash,
                "reason": "alpha_exit",
                "alpha_pred_at_exit": float(alpha),
                "days_held": int(days_held),
            })
            del positions[it]

    # Final mark
    last_day = sim_dates[-1] if len(sim_dates) else None
    final_mv_gross = 0.0
    if last_day is not None and positions:
        held = [it for it in positions if it in price_cols]
        if held:
            row = price_wide_mtm.loc[last_day, held]
            for it in held:
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
        "exit_threshold": exit_threshold,
        "min_holding_days": int(min_holding_days),
        "initial_capital_rub": float(initial_capital),
        "final_equity_gross_rub": float(final_equity_gross),
        "final_equity_net_rub": float(final_equity_net),
        "final_cash_rub": float(cash),
        "final_cash_share": float(cash / final_equity_gross) if final_equity_gross > 0 else 1.0,
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


def evaluate_exit_decisions(
    trades_df: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    sim_dates: pd.DatetimeIndex,
    sell_cost_rate: float,
) -> tuple[pd.DataFrame, dict]:
    """For each sell, compare cash received to what holding to end would
    have produced. Returns annotated trades and aggregate diagnostics."""
    if not len(trades_df):
        return trades_df, {"n_sells": 0}

    trades_df = trades_df.copy()
    for col in [
        "final_price", "counterfactual_value_gross", "counterfactual_value_net",
        "decision_pnl_vs_hold_gross", "decision_pnl_vs_hold_net",
    ]:
        trades_df[col] = np.nan

    sell_mask = trades_df["side"] == "sell"
    if sell_mask.sum() == 0:
        return trades_df, {"n_sells": 0}

    last_day = sim_dates[-1]
    final_prices = price_wide_mtm.loc[last_day]

    def fp(item: str) -> float:
        if item in final_prices.index:
            px = final_prices[item]
            if pd.notna(px) and px > 0:
                return float(px)
        return np.nan

    sell_idx = trades_df.index[sell_mask]
    trades_df.loc[sell_idx, "final_price"] = trades_df.loc[sell_idx, "item"].map(fp)
    units = trades_df.loc[sell_idx, "units"].astype(float)
    final = trades_df.loc[sell_idx, "final_price"].astype(float)
    cv_gross = units * final
    cv_net = cv_gross * (1.0 - sell_cost_rate)
    proceeds = trades_df.loc[sell_idx, "proceeds_rub"].astype(float)
    trades_df.loc[sell_idx, "counterfactual_value_gross"] = cv_gross
    trades_df.loc[sell_idx, "counterfactual_value_net"] = cv_net
    trades_df.loc[sell_idx, "decision_pnl_vs_hold_gross"] = proceeds - cv_gross
    trades_df.loc[sell_idx, "decision_pnl_vs_hold_net"] = proceeds - cv_net

    evaluable = trades_df.loc[sell_idx].dropna(subset=["final_price"])
    if len(evaluable) == 0:
        return trades_df, {"n_sells": int(sell_mask.sum()), "n_evaluable_sells": 0}

    diagnostics = {
        "n_sells": int(sell_mask.sum()),
        "n_evaluable_sells": int(len(evaluable)),
        "total_decision_pnl_vs_hold_gross_rub": float(evaluable["decision_pnl_vs_hold_gross"].sum()),
        "total_decision_pnl_vs_hold_net_rub": float(evaluable["decision_pnl_vs_hold_net"].sum()),
        "mean_decision_pnl_vs_hold_gross_rub": float(evaluable["decision_pnl_vs_hold_gross"].mean()),
        "mean_decision_pnl_vs_hold_net_rub": float(evaluable["decision_pnl_vs_hold_net"].mean()),
        "median_decision_pnl_vs_hold_net_rub": float(evaluable["decision_pnl_vs_hold_net"].median()),
        "share_good_exits_gross": float((evaluable["decision_pnl_vs_hold_gross"] > 0).mean()),
        "share_good_exits_net": float((evaluable["decision_pnl_vs_hold_net"] > 0).mean()),
    }
    return trades_df, diagnostics


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

    if not alpha_path.exists():
        raise FileNotFoundError(
            f"Required alpha_predictions.parquet not found at {alpha_path}. Run step 07 first."
        )

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
    print(f"  {len(price_wide.columns)} items, {len(price_wide)} dates")

    print("Loading alpha predictions...")
    alpha_df = pd.read_parquet(alpha_path, columns=["date", "market_hash_name", "alpha_pred"])
    alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
    alpha_lookup = {
        (r.date, r.market_hash_name): float(r.alpha_pred)
        for r in alpha_df.itertuples(index=False)
    }
    print(f"  Loaded {len(alpha_lookup):,} alpha predictions")

    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & set(universe_by_date.keys())))
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    print(f"Simulating {len(sim_dates)} days from {sim_dates[0].date()} to {sim_dates[-1].date()}")

    min_holding_days = 7

    configs: list[tuple[str, float | None]] = [
        ("buy_and_hold", None),
        ("exit_lt_neg_0_3", -0.3),
        ("exit_lt_neg_0_5", -0.5),
        ("exit_lt_neg_0_7", -0.7),
        ("exit_lt_neg_0_9", -0.9),
    ]

    all_summaries: dict = {}

    for name, threshold in configs:
        print(f"\n=== {name} (threshold={threshold}) ===")
        equity_df, trades_df, summary = simulate_selective_exit(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            universe_by_date=universe_by_date,
            alpha_lookup=alpha_lookup,
            sim_dates=sim_dates,
            exit_threshold=threshold,
            min_holding_days=min_holding_days,
            initial_capital=initial_capital,
            sell_cost_rate=sell_cost,
        )
        trades_df, decision_diag = evaluate_exit_decisions(
            trades_df, price_wide_mtm, sim_dates, sell_cost
        )
        summary["by_split"] = per_split_roi(equity_df, splits)
        summary["decision_quality"] = decision_diag
        all_summaries[name] = summary

        save_table(equity_df, backtests_dir / f"selective_exit_{name}_equity.parquet")
        save_table(trades_df, backtests_dir / f"selective_exit_{name}_trades.parquet")

        print(f"  final: gross={summary['final_equity_gross_rub']:,.0f} "
              f"net={summary['final_equity_net_rub']:,.0f}")
        print(f"  ROI: gross={summary['roi_gross_pct']:+.1f}% "
              f"net={summary['roi_net_pct']:+.1f}%")
        print(f"  sells: {summary['n_sells']}, final cash share: {summary['final_cash_share']:.1%}, "
              f"max DD: {summary['max_drawdown_pct']:+.1f}%")
        if decision_diag.get("n_evaluable_sells", 0) > 0:
            print(f"  decision PnL vs hold (net): "
                  f"total {decision_diag['total_decision_pnl_vs_hold_net_rub']:+,.0f} RUB, "
                  f"mean {decision_diag['mean_decision_pnl_vs_hold_net_rub']:+,.0f}/sell, "
                  f"share good {decision_diag['share_good_exits_net']:.1%}")

    # Comparison table
    print("\n" + "=" * 120)
    print(f"{'Config':<20} | {'Sells':>5} | {'ROI net':>8} | {'Train':>8} | {'Val':>7} | {'Test':>7} | "
          f"{'Sharpe':>6} | {'Total exit PnL net':>18} | {'Good':>5}")
    print("-" * 120)
    for name, s in all_summaries.items():
        bs = s["by_split"]
        tr = bs.get("train", {}).get("roi_pct")
        va = bs.get("val", {}).get("roi_pct")
        te = bs.get("test", {}).get("roi_pct")
        dq = s.get("decision_quality", {})
        total_pnl = dq.get("total_decision_pnl_vs_hold_net_rub")
        share_good = dq.get("share_good_exits_net")
        sharpe = s.get("sharpe_gross")
        print(
            f"{name:<20} | {s['n_sells']:>5} | "
            f"{s['roi_net_pct']:>+7.1f}% | "
            f"{(f'{tr:+.1f}%' if tr is not None else '-'):>8} | "
            f"{(f'{va:+.1f}%' if va is not None else '-'):>7} | "
            f"{(f'{te:+.1f}%' if te is not None else '-'):>7} | "
            f"{(f'{sharpe:.2f}' if sharpe is not None else '-'):>6} | "
            f"{(f'{total_pnl:>+12,.0f} RUB' if total_pnl is not None else '-'):>18} | "
            f"{(f'{share_good:.0%}' if share_good is not None else '-'):>5}"
        )

    write_json(reports_dir / "14_selective_exit_summary.json", all_summaries)
    print(f"\nSaved summary to {reports_dir / '14_selective_exit_summary.json'}")


if __name__ == "__main__":
    main()
