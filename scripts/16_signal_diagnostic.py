#!/usr/bin/env python
"""
Research script: AI signal diagnostic and friction sensitivity analysis.

NOT a strategy script. The goal is to characterize whether the existing
alpha model has predictive value in the CS2 market under realistic and
hypothetical cost structures.

Part 1 — Alpha calibration:
  For each split (train/val/test), bin items into within-day alpha_pred
  deciles and compute the realized future_excess_gross_return_30d
  distribution per decile. A model with predictive signal shows a
  monotonic relationship between decile rank and mean realized return.
  A flat or inverted relationship indicates the model cannot rank
  items in a way that corresponds to future outcomes.

  Also reports:
    - Spearman correlation between alpha_pred and realized target
    - Top minus bottom decile spread (the actionable magnitude)

Part 2 — Friction sensitivity:
  Run the selective-exit strategy at a grid of sell-cost rates
  (0%, 2%, 5%, 7%, 10%) crossed with exit thresholds
  (-0.5, -0.7, -0.9). Compare each (cost, threshold) result to
  buy-and-hold at the same cost. The break-even cost — if any —
  is where the best exit variant first matches buy-and-hold.

Three possible findings:
  (1) Signal exists, friction kills it: zero-cost selective-exit
      beats buy-and-hold by a margin that shrinks as cost rises,
      crossing zero at some intermediate friction level.
  (2) Signal exists but not actionable: calibration shows ranking
      ability, but even zero-cost trading doesn't beat passive
      because exits paid for themselves only marginally.
  (3) No signal: calibration is flat, zero-cost trading still
      underperforms passive. The training framework does not
      capture predictable patterns at the tested horizon.

Run:
  python scripts/16_signal_diagnostic.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  data/processed/reports/16_signal_diagnostic_summary.json
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
from cs2_trading_rebuild.utils import write_json


N_DECILES = 10
TARGET_COL = "future_excess_gross_return_30d"
FRICTION_LEVELS = [0.0, 0.02, 0.05, 0.07, 0.10]
EXIT_THRESHOLDS = [-0.5, -0.7, -0.9]
MIN_HOLDING_DAYS = 7


# ---------------------------------------------------------------------------
# Part 1 — Alpha calibration
# ---------------------------------------------------------------------------

def calibration_per_split(alpha_pred_df: pd.DataFrame, splits: dict) -> dict:
    """For each split, bin within-day alpha_pred into deciles and report
    realized return statistics per decile, plus overall ranking metrics."""
    df = alpha_pred_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["alpha_pred"] = pd.to_numeric(df["alpha_pred"], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=["alpha_pred", TARGET_COL])

    out: dict = {}
    for split_name, (start_key, end_key) in [
        ("train", ("train_start", "train_end")),
        ("val", ("val_start", "val_end")),
        ("test", ("test_start", "test_end")),
    ]:
        s = pd.Timestamp(splits[start_key]).normalize()
        e = pd.Timestamp(splits[end_key]).normalize()
        seg = df[(df["date"] >= s) & (df["date"] <= e)].copy()
        if len(seg) < 100:
            out[split_name] = {"n_rows": int(len(seg)), "note": "insufficient_rows"}
            continue

        # Within-day deciles by alpha_pred (matches how the model is used in practice).
        def _decile(group: pd.Series) -> pd.Series:
            ranks = group.rank(method="first")
            try:
                return pd.qcut(ranks, N_DECILES, labels=False, duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=group.index)

        seg["decile"] = seg.groupby("date", group_keys=False)["alpha_pred"].transform(_decile)
        seg = seg.dropna(subset=["decile"])
        seg["decile"] = seg["decile"].astype(int)

        agg = seg.groupby("decile").agg(
            n_items=("alpha_pred", "size"),
            alpha_mean=("alpha_pred", "mean"),
            alpha_median=("alpha_pred", "median"),
            realized_mean=(TARGET_COL, "mean"),
            realized_median=(TARGET_COL, "median"),
            realized_std=(TARGET_COL, "std"),
            realized_share_positive=(TARGET_COL, lambda x: float((x > 0).mean())),
        ).reset_index()

        spearman = float(seg[["alpha_pred", TARGET_COL]].corr(method="spearman").iloc[0, 1])
        top_dec = agg.iloc[-1]
        bot_dec = agg.iloc[0]
        top_minus_bottom_mean = float(top_dec["realized_mean"] - bot_dec["realized_mean"])
        top_minus_bottom_median = float(top_dec["realized_median"] - bot_dec["realized_median"])

        # Monotonicity check: number of adjacent decile pairs where mean realized return increases.
        means = agg["realized_mean"].to_numpy()
        n_monotonic_up = int(np.sum(np.diff(means) > 0))
        n_pairs = len(means) - 1

        out[split_name] = {
            "n_rows": int(len(seg)),
            "spearman_alpha_vs_realized": spearman,
            "top_minus_bottom_decile_mean": top_minus_bottom_mean,
            "top_minus_bottom_decile_median": top_minus_bottom_median,
            "monotonic_up_pairs": n_monotonic_up,
            "n_decile_pairs": n_pairs,
            "deciles": agg.to_dict(orient="records"),
        }
    return out


# ---------------------------------------------------------------------------
# Part 2 — Friction sensitivity
# Reuses the selective-exit simulator logic from script 14, parameterized
# by sell_cost_rate.
# ---------------------------------------------------------------------------

def simulate_selective_exit(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    universe_by_date: dict,
    alpha_lookup: dict,
    sim_dates: pd.DatetimeIndex,
    exit_threshold: float | None,
    min_holding_days: int,
    initial_capital: float,
    sell_cost_rate: float,
) -> dict:
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    initial_purchase_done = False
    n_sells = 0

    final_day_for_curve = None
    eq_values: list[float] = []

    for d in sim_dates:
        # Mark-to-market
        mv = 0.0
        if positions:
            held = [it for it in positions if it in price_cols]
            if held:
                row = price_wide_mtm.loc[d, held]
                for it in held:
                    px = row[it]
                    if pd.notna(px) and px > 0:
                        mv += positions[it]["units"] * float(px)
        eq_values.append(cash + mv)
        final_day_for_curve = d

        # Initial buy
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
                positions[it] = {"units": units, "entry_date": d}
            initial_purchase_done = True
            continue

        if exit_threshold is None:
            continue

        for it in list(positions.keys()):
            pos = positions[it]
            if (d - pos["entry_date"]).days < min_holding_days:
                continue
            alpha = alpha_lookup.get((d, it))
            if alpha is None or alpha >= exit_threshold:
                continue
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
            del positions[it]
            n_sells += 1

    # Final mark
    final_mv_gross = 0.0
    if final_day_for_curve is not None and positions:
        held = [it for it in positions if it in price_cols]
        if held:
            row = price_wide_mtm.loc[final_day_for_curve, held]
            for it in held:
                px = row[it]
                if pd.notna(px) and px > 0:
                    final_mv_gross += positions[it]["units"] * float(px)
    final_equity_gross = cash + final_mv_gross
    final_equity_net = cash + final_mv_gross * (1.0 - sell_cost_rate)

    eq_arr = np.array(eq_values, dtype=float)
    if len(eq_arr) > 1:
        daily_ret = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
        daily_ret = daily_ret[np.isfinite(daily_ret)]
        if len(daily_ret) > 1 and float(np.std(daily_ret)) > 0:
            sharpe = float(np.sqrt(365) * np.mean(daily_ret) / np.std(daily_ret))
        else:
            sharpe = None
        running_max = np.maximum.accumulate(eq_arr)
        safe_max = np.where(running_max > 0, running_max, 1.0)
        max_dd_pct = float(((eq_arr - running_max) / safe_max).min() * 100.0)
    else:
        sharpe = None
        max_dd_pct = 0.0

    return {
        "sell_cost_rate": float(sell_cost_rate),
        "exit_threshold": exit_threshold,
        "final_equity_net_rub": float(final_equity_net),
        "final_equity_gross_rub": float(final_equity_gross),
        "roi_net_pct": float((final_equity_net / initial_capital - 1.0) * 100.0),
        "roi_gross_pct": float((final_equity_gross / initial_capital - 1.0) * 100.0),
        "n_sells": int(n_sells),
        "sharpe_gross": sharpe,
        "max_drawdown_pct": max_dd_pct,
    }


def run_friction_grid(
    price_wide, price_wide_mtm, universe_by_date, alpha_lookup, sim_dates,
    initial_capital,
) -> dict:
    grid: dict = {}
    for cost in FRICTION_LEVELS:
        cost_key = f"cost_{int(round(cost * 100))}pct"
        print(f"\n  friction = {cost * 100:.0f}%")
        cost_results: dict = {}
        # Buy-and-hold baseline at this cost
        bh = simulate_selective_exit(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            universe_by_date=universe_by_date,
            alpha_lookup=alpha_lookup,
            sim_dates=sim_dates,
            exit_threshold=None,
            min_holding_days=MIN_HOLDING_DAYS,
            initial_capital=initial_capital,
            sell_cost_rate=cost,
        )
        cost_results["buy_and_hold"] = bh
        print(f"    buy_and_hold       net ROI {bh['roi_net_pct']:+.1f}%")
        for thr in EXIT_THRESHOLDS:
            r = simulate_selective_exit(
                price_wide=price_wide,
                price_wide_mtm=price_wide_mtm,
                universe_by_date=universe_by_date,
                alpha_lookup=alpha_lookup,
                sim_dates=sim_dates,
                exit_threshold=thr,
                min_holding_days=MIN_HOLDING_DAYS,
                initial_capital=initial_capital,
                sell_cost_rate=cost,
            )
            delta = r["roi_net_pct"] - bh["roi_net_pct"]
            cost_results[f"exit_lt_{thr}"] = r
            cost_results[f"exit_lt_{thr}"]["delta_vs_buy_and_hold"] = float(delta)
            mark = "  WIN" if delta > 0 else ""
            print(f"    exit < {thr}        net ROI {r['roi_net_pct']:+.1f}%  "
                  f"(delta vs BH {delta:+.1f}%, sells {r['n_sells']}){mark}")
        grid[cost_key] = cost_results
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    initial_capital = float(cfg["research"]["initial_capital_rub"])

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    models_dir = resolve_path(cfg, "models_dir")
    alpha_path = models_dir / "alpha_predictions.parquet"
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not alpha_path.exists():
        raise FileNotFoundError(f"alpha_predictions.parquet not found at {alpha_path}.")

    # Part 1: Calibration
    print("=" * 70)
    print("PART 1 — Alpha calibration")
    print("=" * 70)
    print(f"Loading alpha predictions from {alpha_path}")
    alpha_pred_df = pd.read_parquet(
        alpha_path,
        columns=["date", "market_hash_name", "alpha_pred", TARGET_COL],
    )
    print(f"  {len(alpha_pred_df):,} prediction rows")
    calibration = calibration_per_split(alpha_pred_df, splits)

    for split_name, res in calibration.items():
        if "deciles" not in res:
            print(f"\n--- {split_name} --- {res}")
            continue
        print(f"\n--- {split_name} (n={res['n_rows']:,}) ---")
        print(f"  Spearman alpha_pred vs realized: {res['spearman_alpha_vs_realized']:+.3f}")
        print(f"  Top - bottom decile mean: {res['top_minus_bottom_decile_mean']:+.4f}")
        print(f"  Top - bottom decile median: {res['top_minus_bottom_decile_median']:+.4f}")
        print(f"  Monotonic-up adjacent pairs: {res['monotonic_up_pairs']}/{res['n_decile_pairs']}")
        print(f"  Decile  N      AlphaMean   RealMean   RealMedian  SharePos")
        for d in res["deciles"]:
            print(f"    {int(d['decile']):>3}   {int(d['n_items']):>6}  "
                  f"{d['alpha_mean']:>+8.3f}   {d['realized_mean']:>+7.4f}   "
                  f"{d['realized_median']:>+7.4f}    {d['realized_share_positive']:>5.1%}")

    # Part 2: Friction sensitivity
    print("\n" + "=" * 70)
    print("PART 2 — Friction sensitivity")
    print("=" * 70)
    print("Loading universe and prices...")
    universe_df = pd.read_parquet(
        universe_path, columns=["date", "market_hash_name", "tradeable_today"]
    )
    universe_df["date"] = pd.to_datetime(universe_df["date"]).dt.normalize()
    universe_df["tradeable_today"] = universe_df["tradeable_today"].fillna(False).astype(bool)
    universe_df = universe_df[universe_df["tradeable_today"]]
    universe_by_date = {d: set(g["market_hash_name"].tolist())
                        for d, g in universe_df.groupby("date", sort=False)}

    price_panel = pd.read_parquet(feature_path, columns=["date", "market_hash_name", "price_rub"])
    price_panel["date"] = pd.to_datetime(price_panel["date"]).dt.normalize()
    price_panel["price_rub"] = pd.to_numeric(price_panel["price_rub"], errors="coerce").astype("float32")
    price_panel = price_panel.dropna(subset=["price_rub"]).drop_duplicates(
        subset=["date", "market_hash_name"], keep="last"
    )
    price_wide = price_panel.pivot(
        index="date", columns="market_hash_name", values="price_rub"
    ).sort_index()
    price_wide_mtm = price_wide.ffill()

    alpha_lookup = {(r.date, r.market_hash_name): float(r.alpha_pred)
                    for r in alpha_pred_df[["date", "market_hash_name", "alpha_pred"]]
                    .itertuples(index=False)}
    print(f"  {len(alpha_lookup):,} alpha predictions for the lookup")

    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & set(universe_by_date.keys())))
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    print(f"  Simulating {len(sim_dates)} days")

    friction_grid = run_friction_grid(
        price_wide=price_wide,
        price_wide_mtm=price_wide_mtm,
        universe_by_date=universe_by_date,
        alpha_lookup=alpha_lookup,
        sim_dates=sim_dates,
        initial_capital=initial_capital,
    )

    # Cross-tab summary
    print("\n" + "=" * 70)
    print("Friction x Strategy: ROI net %")
    print("=" * 70)
    cost_keys = list(friction_grid.keys())
    headers = ["Strategy"] + [k.replace("cost_", "") for k in cost_keys]
    print(f"{headers[0]:<18} | " + " | ".join(f"{h:>7}" for h in headers[1:]))
    print("-" * (18 + 3 + 10 * len(cost_keys)))
    for strat_name in ["buy_and_hold"] + [f"exit_lt_{t}" for t in EXIT_THRESHOLDS]:
        row = [strat_name]
        for ck in cost_keys:
            r = friction_grid[ck].get(strat_name, {})
            row.append(f"{r.get('roi_net_pct', float('nan')):+6.1f}%")
        print(f"{row[0]:<18} | " + " | ".join(f"{v:>7}" for v in row[1:]))

    print("\nDelta vs buy-and-hold (positive = AI beats passive at that friction):")
    print(f"{'Strategy':<18} | " + " | ".join(f"{k.replace('cost_', ''):>7}" for k in cost_keys))
    print("-" * (18 + 3 + 10 * len(cost_keys)))
    for thr in EXIT_THRESHOLDS:
        row = [f"exit_lt_{thr}"]
        for ck in cost_keys:
            r = friction_grid[ck].get(f"exit_lt_{thr}", {})
            d = r.get("delta_vs_buy_and_hold", float("nan"))
            row.append(f"{d:+6.1f}%")
        print(f"{row[0]:<18} | " + " | ".join(f"{v:>7}" for v in row[1:]))

    output = {
        "config": {
            "n_deciles": N_DECILES,
            "target_col": TARGET_COL,
            "friction_levels": FRICTION_LEVELS,
            "exit_thresholds": EXIT_THRESHOLDS,
            "min_holding_days": MIN_HOLDING_DAYS,
            "initial_capital_rub": initial_capital,
        },
        "calibration": calibration,
        "friction_grid": friction_grid,
    }
    write_json(reports_dir / "16_signal_diagnostic_summary.json", output)
    print(f"\nSaved summary to {reports_dir / '16_signal_diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
