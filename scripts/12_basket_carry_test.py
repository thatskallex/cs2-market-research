#!/usr/bin/env python
"""
Empirical check: does an equal-weighted basket of the daily tradeable
CS2 universe have positive carry over train/val/test periods?

This is a model-free benchmark. The question we want to answer:
  - If we had bought all items tradeable on day T and held them to day T+H,
    what would our return be (gross and net of the configured sell cost)?
  - How does that vary by horizon H?
  - How does the distribution across start dates look?

If the basket has positive net carry at long horizons, the "exclusion-based
catalog" framing has a foundation. If it doesn't, that framing collapses
and we need a different design.

Outputs:
  data/processed/reports/12_basket_carry_report.json
  data/processed/reports/12_basket_carry_holds.parquet

Run:
  python scripts/12_basket_carry_test.py --config configs/cs2_trading_rebuilt.yaml
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


HOLD_HORIZONS_DAYS = [30, 90, 180, 365]


def _summarize(values: pd.Series) -> dict:
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "share_positive": float((s > 0).mean()),
    }


def _basket_carry_for_horizon(
    universe_split: pd.DataFrame,
    price_wide: pd.DataFrame,
    horizon_days: int,
    sell_cost_rate: float,
    data_max_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """For each usable start date in universe_split, compute the equal-weighted
    basket return over a forward window of `horizon_days`."""
    split_dates = pd.DatetimeIndex(sorted(universe_split["date"].unique()))
    usable_starts = [d for d in split_dates if d + pd.Timedelta(days=horizon_days) <= data_max_date]

    if len(usable_starts) == 0:
        return pd.DataFrame(), {"n_start_dates": 0, "horizon_days": horizon_days}

    universe_by_date = {d: g for d, g in universe_split.groupby("date", sort=False)}
    price_cols = set(price_wide.columns)

    rows = []
    item_stuck_share = []  # fraction of items per start_date whose data ended early

    for start in usable_starts:
        end = start + pd.Timedelta(days=horizon_days)

        uni_today = universe_by_date.get(start)
        if uni_today is None:
            continue
        tradeable_items = [it for it in uni_today.loc[uni_today["tradeable_today"], "market_hash_name"] if it in price_cols]
        if not tradeable_items:
            continue

        # Need start prices that exist and are positive.
        try:
            start_prices = price_wide.loc[start, tradeable_items]
        except KeyError:
            continue
        valid = start_prices.notna() & (start_prices > 0)
        start_prices = start_prices[valid]
        if len(start_prices) == 0:
            continue
        valid_items = start_prices.index.tolist()

        # End prices: last available value within [start, end] per item.
        window = price_wide.loc[start:end, valid_items]
        end_prices = window.ffill().iloc[-1]

        ok = end_prices.notna() & (end_prices > 0)
        if ok.sum() == 0:
            continue
        sp = start_prices[ok].astype("float64")
        ep = end_prices[ok].astype("float64")

        returns_gross = (ep / sp - 1.0).to_numpy()
        returns_net = (1.0 + returns_gross) * (1.0 - sell_cost_rate) - 1.0

        # Track how many items "stopped trading" inside the window.
        last_valid = window[ok.index[ok]].apply(lambda c: c.last_valid_index())
        days_short = (end - last_valid).dt.days.fillna(0)
        share_stuck = float((days_short > 7).mean())
        item_stuck_share.append(share_stuck)

        rows.append({
            "start_date": start,
            "horizon_days": horizon_days,
            "n_items": int(len(returns_gross)),
            "basket_gross_return": float(returns_gross.mean()),
            "basket_net_return": float(returns_net.mean()),
            "basket_median_gross": float(np.median(returns_gross)),
            "basket_median_net": float(np.median(returns_net)),
            "share_items_positive_gross": float((returns_gross > 0).mean()),
            "share_items_positive_net": float((returns_net > 0).mean()),
            "share_items_stuck": share_stuck,
        })

    holds_df = pd.DataFrame(rows)
    if len(holds_df) == 0:
        return holds_df, {"n_start_dates": 0, "horizon_days": horizon_days}

    summary = {
        "horizon_days": horizon_days,
        "n_start_dates": int(len(holds_df)),
        "basket_gross_return_across_starts": _summarize(holds_df["basket_gross_return"]),
        "basket_net_return_across_starts": _summarize(holds_df["basket_net_return"]),
        "median_item_gross_across_starts": _summarize(holds_df["basket_median_gross"]),
        "mean_share_items_stuck": float(np.mean(item_stuck_share)) if item_stuck_share else 0.0,
    }
    return holds_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    sell_cost = float(cfg["research"]["sell_cost_rate"])

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading universe from {universe_path}")
    universe_df = pd.read_parquet(universe_path, columns=["date", "market_hash_name", "tradeable_today"])
    universe_df["date"] = pd.to_datetime(universe_df["date"]).dt.normalize()
    universe_df["tradeable_today"] = universe_df["tradeable_today"].fillna(False).astype(bool)

    print(f"Loading price panel from {feature_path}")
    price_panel = pd.read_parquet(feature_path, columns=["date", "market_hash_name", "price_rub"])
    price_panel["date"] = pd.to_datetime(price_panel["date"]).dt.normalize()
    price_panel["price_rub"] = pd.to_numeric(price_panel["price_rub"], errors="coerce").astype("float32")
    price_panel = price_panel.dropna(subset=["price_rub"])
    price_panel = price_panel.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    print(f"Pivoting prices to wide ({price_panel['market_hash_name'].nunique()} items "
          f"x {price_panel['date'].nunique()} dates)")
    price_wide = price_panel.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    data_max_date = price_wide.index.max()
    print(f"Price grid spans {price_wide.index.min().date()} to {data_max_date.date()}")

    report = {
        "sell_cost_rate": sell_cost,
        "splits": {k: str(v) for k, v in splits.items()},
        "data_max_date": str(data_max_date.date()),
        "hold_horizons_days": HOLD_HORIZONS_DAYS,
        "by_split": {},
    }
    all_holds: list[pd.DataFrame] = []

    for split_name, (start_key, end_key) in [
        ("train", ("train_start", "train_end")),
        ("val", ("val_start", "val_end")),
        ("test", ("test_start", "test_end")),
    ]:
        split_start = pd.Timestamp(splits[start_key]).normalize()
        split_end = pd.Timestamp(splits[end_key]).normalize()
        split_universe = universe_df.loc[
            (universe_df["date"] >= split_start) & (universe_df["date"] <= split_end)
        ]
        n_dates = split_universe["date"].nunique()
        print(f"\n=== {split_name} ({split_start.date()} -> {split_end.date()}, {n_dates} dates) ===")

        per_horizon: dict = {}
        for h in HOLD_HORIZONS_DAYS:
            holds_df, summary = _basket_carry_for_horizon(
                universe_split=split_universe,
                price_wide=price_wide,
                horizon_days=h,
                sell_cost_rate=sell_cost,
                data_max_date=data_max_date,
            )
            per_horizon[f"hold_{h}d"] = summary
            if len(holds_df):
                holds_df["split"] = split_name
                all_holds.append(holds_df)
                gross = summary["basket_gross_return_across_starts"]
                net = summary["basket_net_return_across_starts"]
                print(f"  H={h:>3}d  n_starts={summary['n_start_dates']:>4}  "
                      f"gross mean={gross['mean']:+.4f} median={gross['median']:+.4f}  "
                      f"net mean={net['mean']:+.4f}  "
                      f"share_starts_positive_net={net['share_positive']:.2f}  "
                      f"stuck={summary['mean_share_items_stuck']:.2f}")
            else:
                print(f"  H={h:>3}d  no usable start dates (need {h}d of forward data)")

        report["by_split"][split_name] = per_horizon

    if all_holds:
        out_holds = pd.concat(all_holds, ignore_index=True)
        out_path = reports_dir / "12_basket_carry_holds.parquet"
        save_table(out_holds, out_path)
        print(f"\nSaved hold-level rows to {out_path}")

    report_path = reports_dir / "12_basket_carry_report.json"
    write_json(report_path, report)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
