#!/usr/bin/env python
"""
Regime-threshold sweep for the CS2 alpha-rotation strategy.

Purpose
-------
This script tests how the decision threshold of the market-regime model affects:
  1) classification diagnostics of the regime model;
  2) economic performance of an alpha-rotation strategy gated by regime_prob.

The model itself is NOT retrained. Only the decision rule changes:

    if regime_prob >= tau: allow the scheduled alpha-rotation rebalance
    else: skip that rebalance and keep the current portfolio unchanged

The threshold is selected on the validation period and then reported on the
held-out test period. This keeps the test period as an out-of-sample check.

Default strategy tested
-----------------------
  top_pct = 0.05
  rebalance_days = 90
  sell_cost_rate = 0.10

Outputs
-------
All files are written to:
  data/processed/reports/chapter4_regime_threshold_sweep/

Tables:
  T_regime_threshold_classification.csv/.md
  T_regime_threshold_strategy_metrics.csv/.md
  T_regime_threshold_selected.csv/.md

Figures:
  Рис_regime_threshold_validation_tradeoff.png
  Рис_regime_threshold_classification_tradeoff.png
  Рис_regime_threshold_test_equity_curves.png

Run
---
  python scripts/24_regime_threshold_sweep.py --config configs/cs2_trading_rebuilt.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = balanced_accuracy_score = precision_score = recall_score = f1_score = roc_auc_score = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path


DEFAULT_THRESHOLDS = [round(x, 2) for x in np.arange(0.00, 1.0001, 0.05)]


def _load_friction_module() -> Any:
    """Load scripts/19_alpha_rotation_friction_sweep.py as a module."""
    path = SCRIPT_DIR / "19_alpha_rotation_friction_sweep.py"
    if not path.exists():
        # fallback for interactive/sandbox work
        path = Path(__file__).resolve().with_name("19_alpha_rotation_friction_sweep.py")
    if not path.exists():
        raise FileNotFoundError("Cannot find scripts/19_alpha_rotation_friction_sweep.py")
    spec = importlib.util.spec_from_file_location("alpha_rotation_friction_sweep", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alpha_rotation_friction_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _split_bounds(splits: dict[str, Any], split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "val":
        return pd.Timestamp(splits["val_start"]).normalize(), pd.Timestamp(splits["val_end"]).normalize()
    if split == "test":
        return pd.Timestamp(splits["test_start"]).normalize(), pd.Timestamp(splits["test_end"]).normalize()
    if split == "train":
        return pd.Timestamp(splits["train_start"]).normalize(), pd.Timestamp(splits["train_end"]).normalize()
    raise ValueError(split)


def _filter_dates(sim_dates: pd.DatetimeIndex, splits: dict[str, Any], split: str) -> pd.DatetimeIndex:
    start, end = _split_bounds(splits, split)
    return pd.DatetimeIndex([d for d in sim_dates if start <= pd.Timestamp(d) <= end])


def _safe_div(a: float, b: float) -> float | None:
    if b == 0 or pd.isna(b):
        return None
    return float(a / b)


def _roi_pct(initial: float, final: float | None) -> float | None:
    if final is None or initial <= 0:
        return None
    return float((float(final) / initial - 1.0) * 100.0)


def _max_drawdown_pct(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) == 0:
        return None
    running_max = eq.cummax().replace(0, np.nan)
    dd = (eq - running_max) / running_max
    if not dd.notna().any():
        return None
    return float(dd.min() * 100.0)


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


def _sortino(equity: pd.Series, annualization_days: int = 365) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    r = eq.pct_change().dropna()
    if len(r) < 2:
        return None
    downside = r[r < 0]
    if len(downside) == 0:
        return None
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if not np.isfinite(dd) or dd <= 0:
        return None
    return float(math.sqrt(annualization_days) * float(r.mean()) / dd)


def _annualized_turnover(trades: pd.DataFrame, initial_capital: float, n_days: int) -> float | None:
    if initial_capital <= 0 or n_days <= 0 or len(trades) == 0:
        return 0.0
    total = float(pd.to_numeric(trades.get("gross_value_rub", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    return float((total / initial_capital) * (365.0 / n_days))


def _strategy_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if len(equity) == 0:
        return {
            "n_days": 0,
            "final_equity_rub": initial_capital,
            "roi_pct": 0.0,
            "sharpe": None,
            "sortino": None,
            "mdd_pct": None,
            "annual_turnover": 0.0,
            "n_trades": 0,
            "avg_n_positions": 0.0,
        }
    final_eq = float(equity["liquidated_equity_rub"].iloc[-1])
    n_days = int(len(equity))
    return {
        "n_days": n_days,
        "final_equity_rub": final_eq,
        "roi_pct": _roi_pct(initial_capital, final_eq),
        "sharpe": _sharpe(equity["liquidated_equity_rub"]),
        "sortino": _sortino(equity["liquidated_equity_rub"]),
        "mdd_pct": _max_drawdown_pct(equity["liquidated_equity_rub"]),
        "annual_turnover": _annualized_turnover(trades, initial_capital, n_days),
        "n_trades": int(len(trades)),
        "avg_n_positions": float(equity["n_positions"].mean()) if "n_positions" in equity.columns else None,
    }


def _classification_metrics(regime: pd.DataFrame, split_dates: pd.DatetimeIndex, threshold: float | None) -> dict[str, Any]:
    part = regime[regime["date"].isin(split_dates)].dropna(subset=["regime_prob", "regime_target"]).copy()
    out: dict[str, Any] = {
        "n_days_classification": int(len(part)),
        "positive_rate_true": float(part["regime_target"].mean()) if len(part) else None,
        "auc": None,
    }
    if len(part) == 0:
        return out
    y = part["regime_target"].astype(int)
    p = part["regime_prob"].astype(float)
    if roc_auc_score is not None and y.nunique() == 2:
        out["auc"] = float(roc_auc_score(y, p))
    if threshold is None:
        return out
    pred = (p >= float(threshold)).astype(int)
    out.update({
        "threshold": float(threshold),
        "active_day_share": float(pred.mean()),
        "accuracy": float(accuracy_score(y, pred)) if accuracy_score is not None else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if balanced_accuracy_score is not None else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if precision_score is not None else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if recall_score is not None else None,
        "f1": float(f1_score(y, pred, zero_division=0)) if f1_score is not None else None,
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    })
    return out


def _make_gated_daily_scores(
    daily_scores: dict[pd.Timestamp, pd.DataFrame],
    regime_by_date: dict[pd.Timestamp, float],
    threshold: float | None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Gate scheduled rebalances by regime probability.

    threshold=None means no regime gate: always allow scheduled alpha rotation.
    If the threshold is not passed, the simulator sees an empty candidate list
    for that signal date and therefore skips the rebalance while keeping existing
    positions unchanged.
    """
    if threshold is None:
        return daily_scores
    empty = pd.DataFrame({"market_hash_name": pd.Series(dtype=str), "alpha_pred": pd.Series(dtype=float)})
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    tau = float(threshold)
    for d, g in daily_scores.items():
        prob = regime_by_date.get(pd.Timestamp(d), np.nan)
        if pd.notna(prob) and float(prob) >= tau:
            out[pd.Timestamp(d)] = g
        else:
            out[pd.Timestamp(d)] = empty
    return out


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def _write_table(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    _format_table(df).to_markdown(path_base.with_suffix(".md"), index=False)


def _pick_threshold(val_strategy: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Pick a threshold on validation without looking at test.

    Important: the trading objective is very coarse for a 90-day rebalance strategy
    because validation may contain only 2-3 rebalance decisions. Therefore the
    safer default is a classification/ranking criterion such as balanced_accuracy,
    while economic metrics are still available as an explicit option.
    """
    cand = val_strategy[val_strategy["threshold"].notna()].copy()
    # Exclude degenerate "never trade" threshold from automatic selection.
    cand = cand[(cand["threshold"] < 1.0) & (cand["n_trades"] > 0)]
    if cand.empty:
        cand = val_strategy[val_strategy["threshold"].notna()].copy()
        cand = cand[cand["threshold"] < 1.0]
    if cand.empty:
        return {}

    if metric not in cand.columns:
        metric = "balanced_accuracy" if "balanced_accuracy" in cand.columns else "sortino"

    cand["metric_score"] = pd.to_numeric(cand[metric], errors="coerce")
    cand["roi_score"] = pd.to_numeric(cand.get("roi_pct"), errors="coerce")
    cand["mdd_abs"] = pd.to_numeric(cand.get("mdd_pct"), errors="coerce").abs()
    cand["active_score"] = pd.to_numeric(cand.get("active_day_share"), errors="coerce")
    cand["turnover_score"] = pd.to_numeric(cand.get("annual_turnover"), errors="coerce")

    if cand["metric_score"].notna().sum() == 0:
        cand["metric_score"] = cand["roi_score"]

    economic_metrics = {"sortino", "sharpe", "roi_pct"}
    if metric in economic_metrics:
        # Economic selection is allowed, but only among trading candidates.
        # Do not force a positive-ROI preference when all validation strategies lose;
        # otherwise the rule becomes unstable and tends to choose the least-bad sparse trade.
        cand = cand.sort_values(
            ["metric_score", "roi_score", "mdd_abs", "turnover_score"],
            ascending=[False, False, True, True],
            na_position="last",
        )
    else:
        # Classification threshold selection: pure max balanced_accuracy can choose
        # a very low threshold when the score is not probability-calibrated. To keep
        # the selected gate interpretable and less aggressive, use a conservative
        # near-optimal rule: among thresholds within 1 percentage point of the best
        # validation classification score, choose the higher threshold; then break
        # ties by precision, recall, active-day-share and ROI.
        best_score = float(cand["metric_score"].max())
        near_tol = 0.01
        near = cand[cand["metric_score"] >= best_score - near_tol].copy()
        if not near.empty:
            cand = near
        tie_cols = []
        if "precision" in cand.columns:
            tie_cols.append("precision")
        if "recall" in cand.columns:
            tie_cols.append("recall")
        sort_cols = ["threshold", "metric_score"] + tie_cols + ["active_score", "roi_score"]
        asc = [False] * len(sort_cols)
        cand = cand.sort_values(sort_cols, ascending=asc, na_position="last")
    return cand.iloc[0].to_dict()


def _plot_validation_tradeoff(out_dir: Path, val_rows: pd.DataFrame, chosen_tau: float | None) -> None:
    if plt is None or val_rows.empty:
        return
    d = val_rows[val_rows["threshold"].notna()].sort_values("threshold")
    fig, ax1 = plt.subplots(figsize=(11.5, 6.2))
    ax1.plot(d["threshold"], d["roi_pct"], marker="o", linewidth=2.0, label="ROI, %")
    ax1.plot(d["threshold"], d["sharpe"], marker="o", linewidth=1.8, label="Sharpe")
    ax1.plot(d["threshold"], d["sortino"], marker="o", linewidth=1.8, label="Sortino")
    ax1.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    if chosen_tau is not None:
        ax1.axvline(chosen_tau, color="black", linestyle="--", linewidth=1.2, label=f"выбранный порог τ={chosen_tau:.2f}")
    ax1.set_xlabel("Порог режимной вероятности τ")
    ax1.set_ylabel("Экономическая метрика")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(d["threshold"], d["active_day_share"] * 100, color="0.35", linestyle=":", linewidth=2.0, label="активные дни, %")
    ax2.set_ylabel("Доля активных дней, %")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    plt.title("Калибровка порога режимной модели на validation")
    plt.tight_layout()
    plt.savefig(out_dir / "Рис_regime_threshold_validation_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close()


def _plot_classification_tradeoff(out_dir: Path, class_df: pd.DataFrame, chosen_tau: float | None) -> None:
    if plt is None or class_df.empty:
        return
    d = class_df[(class_df["split"] == "val") & class_df["threshold"].notna()].sort_values("threshold")
    if d.empty:
        return
    plt.figure(figsize=(11.5, 6.2))
    for col, label in [("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "F1")]:
        if col in d.columns:
            plt.plot(d["threshold"], d[col], marker="o", linewidth=1.9, label=label)
    if chosen_tau is not None:
        plt.axvline(chosen_tau, color="black", linestyle="--", linewidth=1.2, label=f"выбранный порог τ={chosen_tau:.2f}")
    plt.xlabel("Порог режимной вероятности τ")
    plt.ylabel("Классификационная метрика")
    plt.title("Классификационные метрики режимной модели при разных порогах")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "Рис_regime_threshold_classification_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close()


def _plot_test_equity_curves(out_dir: Path, equity_curves: dict[str, pd.DataFrame]) -> None:
    if plt is None or not equity_curves:
        return

    # Main clean figure for the dissertation: compare the baseline full pipeline
    # with the selected additional regime threshold. The arbitrary 0.50 threshold
    # is kept only in a separate audit figure because it is a diagnostic control,
    # not the main benchmark.
    main_order = [k for k in equity_curves.keys() if ("выбранный" in k or "базовый" in k)]
    audit_order = list(equity_curves.keys())

    def _draw(keys: list[str], filename: str, title: str) -> None:
        plt.figure(figsize=(11.5, 6.3))
        for label in keys:
            eq = equity_curves.get(label, pd.DataFrame())
            if len(eq) == 0:
                continue
            y = eq["liquidated_equity_rub"] / float(eq["liquidated_equity_rub"].iloc[0])
            is_selected = "выбранный" in label
            is_baseline = "базовый" in label
            lw = 3.0 if is_selected or is_baseline else 1.7
            ls = "--" if "0.50" in label else "-"
            alpha = 1.0 if is_selected or is_baseline else 0.75
            plt.plot(eq["date"], y, linewidth=lw, linestyle=ls, alpha=alpha, label=label)
        plt.axhline(1.0, color="black", linewidth=0.9, alpha=0.7)
        plt.xlabel("Дата")
        plt.ylabel("Нормированная стоимость портфеля")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close()

    if main_order:
        _draw(
            main_order,
            "Рис_regime_threshold_test_equity_curves.png",
            "Test: базовый full pipeline и мягкий режимный фильтр",
        )
    _draw(
        audit_order,
        "Рис_regime_threshold_test_equity_curves_audit.png",
        "Audit: equity curves при разных порогах режимной модели",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-pct", type=float, default=0.05)
    parser.add_argument("--rebalance-days", type=int, default=90)
    parser.add_argument("--sell-cost", type=float, default=0.10)
    parser.add_argument(
        "--selection-metric",
        choices=["balanced_accuracy", "f1", "precision", "recall", "sortino", "sharpe", "roi_pct"],
        default="balanced_accuracy",
        help=(
            "Metric used to choose the threshold on validation. "
            "For classification metrics the script applies a conservative near-optimal "
            "rule: among thresholds within 1 pp of the best validation score, it chooses "
            "the higher threshold to avoid an overly permissive gate."
        ),
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = cfg["splits"]
    initial_capital = float(cfg["research"]["initial_capital_rub"])
    execute_next_day = bool(cfg["research"].get("execute_next_day", True))

    reports_dir = resolve_path(cfg, "reports_dir") / "chapter4_regime_threshold_sweep"
    reports_dir.mkdir(parents=True, exist_ok=True)

    friction = _load_friction_module()
    universe, alpha, price_wide, price_wide_mtm = friction._load_inputs(cfg)
    sim_dates = friction._make_sim_dates(price_wide, universe, splits)
    daily_scores = friction._build_daily_tradeable_scores(universe, alpha, sim_dates)

    regime_path = resolve_path(cfg, "models_dir") / "regime_predictions.parquet"
    if not regime_path.exists():
        raise FileNotFoundError(f"regime_predictions.parquet not found: {regime_path}")
    regime = pd.read_parquet(regime_path)
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    if "regime_prob" not in regime.columns or "regime_target" not in regime.columns:
        raise KeyError("regime_predictions.parquet must contain regime_prob and regime_target")
    regime_by_date = dict(zip(regime["date"], pd.to_numeric(regime["regime_prob"], errors="coerce")))

    rows_class: list[dict[str, Any]] = []
    rows_strategy: list[dict[str, Any]] = []
    equity_store: dict[tuple[str, str], pd.DataFrame] = {}

    thresholds = sorted(set(float(x) for x in args.thresholds if 0.0 <= float(x) <= 1.0))
    modes: list[tuple[str, float | None]] = [("без фильтра", None)] + [(f"τ={t:.2f}", t) for t in thresholds]

    for split in ["val", "test"]:
        split_dates = _filter_dates(sim_dates, splits, split)
        if len(split_dates) == 0:
            continue
        for mode_name, tau in modes:
            class_row = {"split": split, "mode": mode_name}
            class_row.update(_classification_metrics(regime, split_dates, tau))
            rows_class.append(class_row)

            gated = _make_gated_daily_scores(daily_scores, regime_by_date, tau)
            eq, tr, extra = friction.simulate_rebalance_with_costs(
                price_wide=price_wide,
                price_wide_mtm=price_wide_mtm,
                daily_scores=gated,
                sim_dates=split_dates,
                initial_capital=initial_capital,
                rebalance_days=int(args.rebalance_days),
                selection_mode="top_pct",
                sell_cost_rate=float(args.sell_cost),
                top_pct=float(args.top_pct),
                execute_next_day=execute_next_day,
            )
            met = _strategy_metrics(eq, tr, initial_capital)
            strategy_row = {
                "split": split,
                "mode": mode_name,
                "threshold": tau,
                "top_pct": float(args.top_pct),
                "rebalance_days": int(args.rebalance_days),
                "sell_cost_rate": float(args.sell_cost),
            }
            strategy_row.update({k: v for k, v in class_row.items() if k in ["active_day_share", "auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]})
            strategy_row.update(met)
            strategy_row.update(extra)
            rows_strategy.append(strategy_row)
            equity_store[(split, mode_name)] = eq

    class_df = pd.DataFrame(rows_class)
    strat_df = pd.DataFrame(rows_strategy)

    val_strat = strat_df[strat_df["split"] == "val"].copy()
    chosen = _pick_threshold(val_strat, args.selection_metric)
    chosen_tau = None if not chosen else float(chosen["threshold"])

    selected_rows = []
    if chosen:
        for split in ["val", "test"]:
            row = strat_df[(strat_df["split"] == split) & (strat_df["threshold"] == chosen_tau)]
            if len(row):
                selected_rows.append(row.iloc[0].to_dict())
        # Add benchmarks: no-gate and default 0.5 threshold on test.
        for split in ["val", "test"]:
            for mode_name in ["без фильтра", "τ=0.50"]:
                row = strat_df[(strat_df["split"] == split) & (strat_df["mode"] == mode_name)]
                if len(row):
                    d = row.iloc[0].to_dict()
                    d["benchmark_note"] = mode_name
                    selected_rows.append(d)
    selected_df = pd.DataFrame(selected_rows)

    # Write tables.
    _write_table(class_df, reports_dir / "T_regime_threshold_classification")
    _write_table(strat_df, reports_dir / "T_regime_threshold_strategy_metrics")
    _write_table(selected_df, reports_dir / "T_regime_threshold_selected")

    # Figures.
    val_rows = strat_df[strat_df["split"] == "val"].copy()
    _plot_validation_tradeoff(reports_dir, val_rows, chosen_tau)
    _plot_classification_tradeoff(reports_dir, class_df, chosen_tau)

    equity_curves: dict[str, pd.DataFrame] = {}
    if chosen_tau is not None:
        chosen_label = f"выбранный мягкий фильтр τ={chosen_tau:.2f}"
        equity_curves[chosen_label] = equity_store.get(("test", f"τ={chosen_tau:.2f}"), pd.DataFrame())
    # This is the same conceptual benchmark as Full pipeline in the ablation plot:
    # alpha_pred is used without an additional hard threshold on regime_prob.
    equity_curves["базовый full pipeline (без внешнего порога)"] = equity_store.get(("test", "без фильтра"), pd.DataFrame())
    # Kept only for the audit figure: it shows why the default 0.50 gate is too restrictive.
    equity_curves["жёсткий контроль τ=0.50"] = equity_store.get(("test", "τ=0.50"), pd.DataFrame())
    _plot_test_equity_curves(reports_dir, equity_curves)

    summary = {
        "output_dir": str(reports_dir),
        "strategy": {
            "top_pct": float(args.top_pct),
            "rebalance_days": int(args.rebalance_days),
            "sell_cost_rate": float(args.sell_cost),
            "execute_next_day": bool(execute_next_day),
        },
        "selection_metric": args.selection_metric,
        "chosen_threshold": chosen_tau,
    }
    pd.Series(summary).to_json(reports_dir / "regime_threshold_sweep_summary.json", force_ascii=False, indent=2)

    print("\nSaved regime threshold sweep to:", reports_dir)
    if chosen_tau is not None:
        print(f"Chosen threshold on validation: τ={chosen_tau:.2f} by {args.selection_metric}")
    print("Tables: T_regime_threshold_classification, T_regime_threshold_strategy_metrics, T_regime_threshold_selected")
    print("Figures: Рис_regime_threshold_validation_tradeoff, Рис_regime_threshold_classification_tradeoff, Рис_regime_threshold_test_equity_curves")


if __name__ == "__main__":
    main()
