#!/usr/bin/env python
"""
Long-horizon absolute-loss alpha retrain + selective-exit test.

Trains a binary classifier predicting "will this item lose money in
absolute terms over the next 180 days." Uses the result as a
selective-exit signal layered on buy-and-hold.

The 180d target is computed INLINE from price_rub at T and T+180.
The current feature_store (config alpha_horizons_days=[30,45]) does
not contain future_gross_return_180d. Computing inline is
mathematically identical to what features.py would have produced
because features.py already reindexes each item to a continuous
daily grid with forward-filled prices.

Design choices:
  - Horizon: 180d
  - Target: binary, 1 if future_gross_return_180d < 0
  - Train embargo: 180d
  - Val embargo: 0 (val targets resolve into test period, OK here)
  - Features: same as existing alpha model

Run:
  python scripts/17_long_horizon_loss_retrain.py --config configs/cs2_trading_rebuilt.yaml

Outputs:
  models/alpha_long_horizon_predictions.parquet
  data/processed/reports/17_long_horizon_retrain_summary.json
  backtests/long_horizon_exit_<threshold>_equity.parquet
  backtests/long_horizon_exit_<threshold>_trades.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.utils import save_table, write_json
from cs2_trading_rebuild.models import (
    CATEGORICAL_FEATURES,
    FEATURE_NUMERIC_COLS,
    MODEL_NUMERIC_FEATURES,
)


LOSS_HORIZON_DAYS = 180
TARGET_COL = "target_loss_180d"
EXIT_PROB_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
MIN_HOLDING_DAYS = 7
N_DECILES = 10


def split_mask(df: pd.DataFrame, start, end, embargo_days: int = 0) -> pd.Series:
    d = pd.to_datetime(df["date"]).dt.normalize()
    end_ts = pd.Timestamp(end) - pd.Timedelta(days=embargo_days)
    return (d >= pd.Timestamp(start)) & (d <= end_ts)


def fit_category_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for c in CATEGORICAL_FEATURES:
        ser = train_df[c].astype("string").fillna("__MISSING__")
        vc = ser.value_counts(dropna=False)
        maps[c] = {str(v): i for i, v in enumerate(vc.index.tolist())}
    return maps


def build_dense_matrix(df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]) -> np.ndarray:
    n = len(df)
    X = np.empty((n, len(MODEL_NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)), dtype=np.float32)
    for j, c in enumerate(MODEL_NUMERIC_FEATURES):
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors="coerce").astype("float32")
            X[:, j] = ser.to_numpy(copy=False)
        else:
            X[:, j] = np.nan
    base = len(MODEL_NUMERIC_FEATURES)
    for k, c in enumerate(CATEGORICAL_FEATURES):
        mapping = cat_maps[c]
        ser = df[c].astype("string").fillna("__MISSING__")
        X[:, base + k] = ser.map(mapping).fillna(-1).astype("float32").to_numpy(copy=False)
    return X


def load_work_table(cfg: dict) -> pd.DataFrame:
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    splits = cfg["splits"]

    split_min = pd.Timestamp(splits["train_start"]).normalize()
    split_max = pd.Timestamp(splits["test_end"]).normalize()

    # Load model features only — exclude any future_* columns. We compute the
    # 180d target inline below, so we don't need the feature_store to contain
    # a future_gross_return_180d column.
    feature_cols_to_load = list(dict.fromkeys(
        ["date", "market_hash_name"]
        + [c for c in MODEL_NUMERIC_FEATURES if c != "tradability_score"]
        + CATEGORICAL_FEATURES
    ))

    print(f"Loading feature store ({len(feature_cols_to_load)} columns)...")
    feat = pd.read_parquet(
        feature_path,
        columns=feature_cols_to_load,
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    feat = feat.loc[:, ~feat.columns.duplicated()].copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

    # (date, item) -> price lookup BEFORE filtering to tradeable.
    # Needed because items may have left the universe at T+180.
    print("Building price lookup for inline 180d target computation...")
    price_indexed = (
        feat[["date", "market_hash_name", "price_rub"]]
        .dropna(subset=["price_rub"])
        .drop_duplicates(subset=["date", "market_hash_name"], keep="last")
        .set_index(["date", "market_hash_name"])["price_rub"]
        .astype("float32")
    )

    print("Loading universe...")
    uni = pd.read_parquet(
        universe_path,
        columns=["date", "market_hash_name", "tradeable_today", "tradability_score"],
        filters=[("date", ">=", split_min), ("date", "<=", split_max)],
    )
    uni["date"] = pd.to_datetime(uni["date"]).dt.normalize()
    uni = uni.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    work = feat.merge(uni, on=["date", "market_hash_name"], how="inner")
    work = work[work["tradeable_today"].fillna(False).astype(bool)].copy()
    del feat
    gc.collect()

    print(f"Computing {LOSS_HORIZON_DAYS}d future returns for {len(work):,} work rows...")
    target_dates = (work["date"] + pd.Timedelta(days=LOSS_HORIZON_DAYS)).values
    lookup_idx = pd.MultiIndex.from_arrays(
        [target_dates, work["market_hash_name"].values],
        names=["date", "market_hash_name"],
    )
    future_prices = price_indexed.reindex(lookup_idx).to_numpy()

    work["price_future_180d"] = future_prices
    work["future_gross_return_180d"] = (
        pd.to_numeric(work["price_future_180d"], errors="coerce")
        / pd.to_numeric(work["price_rub"], errors="coerce")
    ) - 1.0

    work[TARGET_COL] = (work["future_gross_return_180d"] < 0).astype("Int8")
    work.loc[work["future_gross_return_180d"].isna(), TARGET_COL] = pd.NA

    work["tradability_score"] = pd.to_numeric(work["tradability_score"], errors="coerce").fillna(0.0)

    for c in CATEGORICAL_FEATURES:
        if c in work.columns:
            work[c] = work[c].astype("string").fillna("__MISSING__")

    work = work.sort_values(["date", "market_hash_name"]).reset_index(drop=True)

    n_valid_target = int(work[TARGET_COL].notna().sum())
    n_loss = int((work[TARGET_COL] == 1).sum())
    n_no_loss = int((work[TARGET_COL] == 0).sum())
    print(f"  work table: {len(work):,} tradeable item-day rows")
    print(f"  rows with valid {LOSS_HORIZON_DAYS}d target: {n_valid_target:,}")
    print(f"    target=1 (loss): {n_loss:,} ({n_loss / max(n_valid_target, 1):.1%} base rate)")
    print(f"    target=0 (no loss): {n_no_loss:,}")
    return work


def train_loss_classifier(work: pd.DataFrame, cfg: dict) -> tuple:
    splits = cfg["splits"]

    train_mask = split_mask(work, splits["train_start"], splits["train_end"],
                            embargo_days=LOSS_HORIZON_DAYS) & work[TARGET_COL].notna()
    val_mask = split_mask(work, splits["val_start"], splits["val_end"],
                          embargo_days=0) & work[TARGET_COL].notna()

    train_df = work.loc[train_mask].copy()
    val_df = work.loc[val_mask].copy()
    print(f"\nTrain rows (after {LOSS_HORIZON_DAYS}d embargo & target valid): {len(train_df):,}")
    print(f"Val rows (target resolves into test period): {len(val_df):,}")

    if len(train_df) == 0:
        raise ValueError("Train set empty — check splits and horizon.")

    cat_maps = fit_category_maps(train_df)
    X_train = build_dense_matrix(train_df, cat_maps)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()

    mcfg = cfg["modeling"]["alpha_model"]
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=int(mcfg.get("max_depth", 4)),
        min_samples_leaf=int(mcfg.get("min_samples_leaf", 80)),
        max_iter=int(mcfg.get("n_estimators", 300)),
        learning_rate=float(mcfg.get("learning_rate", 0.03)),
        l2_regularization=float(mcfg.get("l2_regularization", 2.0)),
        max_bins=int(mcfg.get("max_bins", 255)),
        random_state=int(cfg["project"]["random_seed"]),
    )
    print("Fitting classifier...")
    clf.fit(X_train, y_train)

    print("Predicting on full panel...")
    X_all = build_dense_matrix(work, cat_maps)
    work["loss_prob"] = clf.predict_proba(X_all)[:, 1].astype("float32")

    train_prob = work.loc[train_df.index, "loss_prob"].to_numpy()
    val_prob = work.loc[val_df.index, "loss_prob"].to_numpy() if len(val_df) else np.array([])
    y_val = val_df[TARGET_COL].astype(int).to_numpy() if len(val_df) else np.array([])

    report = {
        "horizon_days": LOSS_HORIZON_DAYS,
        "target": TARGET_COL,
        "train_embargo_days": LOSS_HORIZON_DAYS,
        "val_embargo_days": 0,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()) if len(y_val) else None,
        "train_roc_auc": float(roc_auc_score(y_train, train_prob)) if len(np.unique(y_train)) > 1 else None,
        "val_roc_auc": float(roc_auc_score(y_val, val_prob)) if (len(y_val) and len(np.unique(y_val)) > 1) else None,
    }
    return clf, cat_maps, work, train_df.index, val_df.index, report


def calibration_per_split(work: pd.DataFrame, train_idx, val_idx) -> dict:
    out: dict = {}
    for label, idx in [("train", train_idx), ("val", val_idx)]:
        seg = work.loc[idx].copy()
        seg = seg.dropna(subset=["loss_prob", TARGET_COL])
        if len(seg) < 100:
            out[label] = {"n_rows": int(len(seg))}
            continue

        try:
            seg["decile"] = pd.qcut(seg["loss_prob"].rank(method="first"),
                                    N_DECILES, labels=False, duplicates="drop")
        except ValueError:
            seg["decile"] = 0
        seg = seg.dropna(subset=["decile"])
        seg["decile"] = seg["decile"].astype(int)
        seg[TARGET_COL] = seg[TARGET_COL].astype(int)

        agg = seg.groupby("decile").agg(
            n=("loss_prob", "size"),
            loss_prob_mean=("loss_prob", "mean"),
            realized_loss_rate=(TARGET_COL, "mean"),
            realized_return_mean=("future_gross_return_180d", "mean"),
            realized_return_median=("future_gross_return_180d", "median"),
        ).reset_index()

        rates = agg["realized_loss_rate"].to_numpy()
        n_monotonic_up = int(np.sum(np.diff(rates) > 0))
        n_pairs = len(rates) - 1

        out[label] = {
            "n_rows": int(len(seg)),
            "monotonic_up_pairs": n_monotonic_up,
            "n_decile_pairs": n_pairs,
            "deciles": agg.to_dict(orient="records"),
        }
    return out


def simulate_selective_exit_lossprob(
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    universe_by_date: dict[pd.Timestamp, set[str]],
    loss_prob_lookup: dict,
    sim_dates: pd.DatetimeIndex,
    exit_prob_threshold: float | None,
    min_holding_days: int,
    initial_capital: float,
    sell_cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_rows: list[dict] = []
    trades_rows: list[dict] = []
    initial_purchase_done = False

    for d in sim_dates:
        mv = 0.0
        if positions:
            held = [it for it in positions if it in price_cols]
            if held:
                row = price_wide_mtm.loc[d, held]
                for it in held:
                    px = row[it]
                    if pd.notna(px) and px > 0:
                        mv += positions[it]["units"] * float(px)
        equity_rows.append({
            "date": d, "cash_rub": cash, "market_value_rub": mv,
            "equity_rub": cash + mv, "n_positions": len(positions),
        })

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
                trades_rows.append({
                    "date": d, "side": "buy", "item": it, "units": int(units),
                    "price": px, "cost_rub": float(cost), "cash_after": cash,
                    "reason": "initial_buy",
                    "loss_prob_at_exit": np.nan, "days_held": 0,
                })
            initial_purchase_done = True
            continue

        if exit_prob_threshold is None:
            continue

        for it in list(positions.keys()):
            pos = positions[it]
            days_held = (d - pos["entry_date"]).days
            if days_held < min_holding_days:
                continue
            prob = loss_prob_lookup.get((d, it))
            if prob is None or prob <= exit_prob_threshold:
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
            trades_rows.append({
                "date": d, "side": "sell", "item": it, "units": int(pos["units"]),
                "price": px, "proceeds_rub": float(proceeds), "cash_after": cash,
                "reason": "loss_prob_exit",
                "loss_prob_at_exit": float(prob), "days_held": int(days_held),
            })
            del positions[it]

    last_day = sim_dates[-1]
    final_mv_gross = 0.0
    if positions:
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
        "exit_prob_threshold": exit_prob_threshold,
        "min_holding_days": int(min_holding_days),
        "initial_capital_rub": float(initial_capital),
        "final_equity_gross_rub": float(final_equity_gross),
        "final_equity_net_rub": float(final_equity_net),
        "final_cash_rub": float(cash),
        "roi_gross_pct": float((final_equity_gross / initial_capital - 1.0) * 100.0),
        "roi_net_pct": float((final_equity_net / initial_capital - 1.0) * 100.0),
        "sharpe_gross": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "n_trades": int(len(trades_df)),
        "n_buys": int((trades_df["side"] == "buy").sum()) if len(trades_df) else 0,
        "n_sells": int((trades_df["side"] == "sell").sum()) if len(trades_df) else 0,
    }
    return equity_df, trades_df, summary


def per_split_roi(equity_df: pd.DataFrame, splits: dict) -> dict:
    out: dict = {}
    for sn, (sk, ek) in [
        ("train", ("train_start", "train_end")),
        ("val", ("val_start", "val_end")),
        ("test", ("test_start", "test_end")),
    ]:
        s = pd.Timestamp(splits[sk]).normalize()
        e = pd.Timestamp(splits[ek]).normalize()
        seg = equity_df[(equity_df["date"] >= s) & (equity_df["date"] <= e)]
        if len(seg) < 2:
            out[sn] = {"n_days": int(len(seg))}
            continue
        s0 = float(seg["equity_rub"].iloc[0])
        e0 = float(seg["equity_rub"].iloc[-1])
        out[sn] = {
            "n_days": int(len(seg)),
            "start_equity_rub": s0, "end_equity_rub": e0,
            "roi_pct": float((e0 / s0 - 1.0) * 100.0) if s0 > 0 else None,
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

    models_dir = resolve_path(cfg, "models_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    backtests_dir = resolve_path(cfg, "backtests_dir")
    for p in (models_dir, reports_dir, backtests_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP A — Build work table")
    print("=" * 70)
    work = load_work_table(cfg)

    print("\n" + "=" * 70)
    print("STEP B — Train binary absolute-loss classifier")
    print("=" * 70)
    clf, cat_maps, work, train_idx, val_idx, train_report = train_loss_classifier(work, cfg)
    print(f"  train ROC AUC: {train_report['train_roc_auc']}")
    print(f"  val   ROC AUC: {train_report['val_roc_auc']}")
    print(f"  train positive rate: {train_report['train_positive_rate']:.3f}")
    print(f"  val   positive rate: {train_report['val_positive_rate']}")

    pred_df = work[["date", "market_hash_name", "loss_prob"]].copy()
    save_table(pred_df, models_dir / "alpha_long_horizon_predictions.parquet")
    print(f"\nSaved predictions to {models_dir / 'alpha_long_horizon_predictions.parquet'}")

    print("\n" + "=" * 70)
    print("STEP C — Calibration of loss_prob vs realized")
    print("=" * 70)
    calibration = calibration_per_split(work, train_idx, val_idx)
    for split_label, res in calibration.items():
        if "deciles" not in res:
            print(f"\n--- {split_label} --- {res}")
            continue
        print(f"\n--- {split_label} (n={res['n_rows']:,}) ---")
        print(f"  Monotonic-up adjacent pairs: {res['monotonic_up_pairs']}/{res['n_decile_pairs']}")
        print(f"  Decile  N      LossProbMean  RealizedLossRate  RealRetMean   RealRetMedian")
        for d in res["deciles"]:
            print(f"    {int(d['decile']):>3}   {int(d['n']):>6}  "
                  f"{d['loss_prob_mean']:>+8.3f}      "
                  f"{d['realized_loss_rate']:>8.3f}          "
                  f"{d['realized_return_mean']:>+7.4f}      "
                  f"{d['realized_return_median']:>+7.4f}")

    print("\n" + "=" * 70)
    print("STEP D — Selective-exit with loss_prob signal")
    print("=" * 70)

    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")

    print("Loading prices...")
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

    print("Loading universe (tradeable map)...")
    uni = pd.read_parquet(universe_path, columns=["date", "market_hash_name", "tradeable_today"])
    uni["date"] = pd.to_datetime(uni["date"]).dt.normalize()
    uni = uni[uni["tradeable_today"].fillna(False).astype(bool)]
    universe_by_date = {d: set(g["market_hash_name"].tolist())
                        for d, g in uni.groupby("date", sort=False)}

    print("Building loss_prob lookup...")
    loss_prob_lookup = {
        (r.date, r.market_hash_name): float(r.loss_prob)
        for r in pred_df.itertuples(index=False)
    }

    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & set(universe_by_date.keys())))
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    print(f"Simulating {len(sim_dates)} days from {sim_dates[0].date()} to {sim_dates[-1].date()}")

    configs: list[tuple[str, float | None]] = [("buy_and_hold", None)]
    configs.extend([(f"exit_prob_gt_{int(t * 100)}", t) for t in EXIT_PROB_THRESHOLDS])

    sim_summaries: dict = {}
    for name, threshold in configs:
        print(f"\n--- {name} (threshold={threshold}) ---")
        equity_df, trades_df, summary = simulate_selective_exit_lossprob(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            universe_by_date=universe_by_date,
            loss_prob_lookup=loss_prob_lookup,
            sim_dates=sim_dates,
            exit_prob_threshold=threshold,
            min_holding_days=MIN_HOLDING_DAYS,
            initial_capital=initial_capital,
            sell_cost_rate=sell_cost,
        )
        summary["by_split"] = per_split_roi(equity_df, splits)
        sim_summaries[name] = summary
        save_table(equity_df, backtests_dir / f"long_horizon_exit_{name}_equity.parquet")
        save_table(trades_df, backtests_dir / f"long_horizon_exit_{name}_trades.parquet")
        print(f"  ROI net: {summary['roi_net_pct']:+.1f}%  sells: {summary['n_sells']}  "
              f"sharpe: {summary['sharpe_gross']}  max DD: {summary['max_drawdown_pct']:+.1f}%")

    print("\n" + "=" * 110)
    print(f"{'Config':<24} | {'Sells':>5} | {'ROI net':>8} | {'Train':>8} | {'Val':>7} | {'Test':>7} | {'Sharpe':>6} | {'vs B&H':>7}")
    print("-" * 110)
    bh_roi = sim_summaries["buy_and_hold"]["roi_net_pct"]
    for name, s in sim_summaries.items():
        bs = s["by_split"]
        tr = bs.get("train", {}).get("roi_pct")
        va = bs.get("val", {}).get("roi_pct")
        te = bs.get("test", {}).get("roi_pct")
        delta = s["roi_net_pct"] - bh_roi
        sharpe = s.get("sharpe_gross")
        print(
            f"{name:<24} | {s['n_sells']:>5} | "
            f"{s['roi_net_pct']:>+7.1f}% | "
            f"{(f'{tr:+.1f}%' if tr is not None else '-'):>8} | "
            f"{(f'{va:+.1f}%' if va is not None else '-'):>7} | "
            f"{(f'{te:+.1f}%' if te is not None else '-'):>7} | "
            f"{(f'{sharpe:.2f}' if sharpe is not None else '-'):>6} | "
            f"{delta:>+6.1f}%"
        )

    output = {
        "training": train_report,
        "calibration": calibration,
        "simulation": sim_summaries,
        "config_used": {
            "loss_horizon_days": LOSS_HORIZON_DAYS,
            "exit_prob_thresholds": EXIT_PROB_THRESHOLDS,
            "min_holding_days": MIN_HOLDING_DAYS,
            "sell_cost_rate": sell_cost,
            "initial_capital_rub": initial_capital,
        },
    }
    write_json(reports_dir / "17_long_horizon_retrain_summary.json", output)
    print(f"\nSaved summary to {reports_dir / '17_long_horizon_retrain_summary.json'}")


if __name__ == "__main__":
    main()
