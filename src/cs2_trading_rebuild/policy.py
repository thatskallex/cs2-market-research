from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .backtest_portfolio import BACKTEST_PANEL_COLS, run_portfolio_backtest

POLICY_PANEL_COLS = [
    "date",
    "market_hash_name",
    "alpha_pred",
    "tradeable_today",
    "tradability_score",
    "regime_prob",
    "price_rub",
    "units_sold",
    "turnover_rub",
    "price_median_30d",
    "units_median_30d",
    "turnover_median_30d",
    "quality",
    "stattrak",
    "souvenir",
    "type",
    "rarity",
    "metatype",
]

OPTIONAL_SCORE_COLS = [
    "alpha_rank_score_raw",
    "alpha_prob_top",
    "alpha_prob_bottom",
    "alpha_prob_top_raw",
    "alpha_prob_bottom_raw",
]


@dataclass
class PolicyResult:
    best_policy: dict[str, Any]
    search_table: pd.DataFrame
    validation_summary: dict[str, Any]


def _robust_scale_by_day(score: pd.Series, dates: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.normalize(),
        "score": pd.to_numeric(score, errors="coerce"),
    })

    def _norm_group(g: pd.DataFrame) -> pd.Series:
        s = g["score"].astype(float)
        med = float(np.nanmedian(s)) if len(s) else 0.0
        mad = float(np.nanmedian(np.abs(s - med))) if len(s) else 0.0
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            std = float(np.nanstd(s))
            scale = std if np.isfinite(std) and std > 1e-6 else 1.0
        z = (s - med) / scale
        return np.clip(z, -5.0, 5.0)

    z = tmp.groupby("date", group_keys=False).apply(_norm_group)
    z.index = tmp.index
    return z.astype("float32")


def _rank_pct_centered(score: pd.Series, dates: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.normalize(),
        "score": pd.to_numeric(score, errors="coerce"),
    })
    pct = tmp.groupby("date", group_keys=False)["score"].rank(method="average", pct=True)
    centered = (pct - 0.5) * 2.0
    return centered.astype("float32")


def _extract_base_score(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit rank score if models.py emits it.
    if "alpha_rank_score_raw" in df.columns:
        return pd.to_numeric(df["alpha_rank_score_raw"], errors="coerce")

    # Prefer explicit two-head probabilities if available.
    top_cols = [c for c in ["alpha_prob_top", "alpha_prob_top_raw"] if c in df.columns]
    bottom_cols = [c for c in ["alpha_prob_bottom", "alpha_prob_bottom_raw"] if c in df.columns]
    if top_cols and bottom_cols:
        return (
            pd.to_numeric(df[top_cols[0]], errors="coerce")
            - pd.to_numeric(df[bottom_cols[0]], errors="coerce")
        )

    # Single-head fallback.
    return pd.to_numeric(df["alpha_pred"], errors="coerce")


def _extract_badness_score(df: pd.DataFrame, base_score: pd.Series) -> pd.Series:
    # Higher badness = more likely loser / avoid.
    if "alpha_prob_bottom" in df.columns:
        return pd.to_numeric(df["alpha_prob_bottom"], errors="coerce")
    if "alpha_prob_bottom_raw" in df.columns:
        return pd.to_numeric(df["alpha_prob_bottom_raw"], errors="coerce")
    if "alpha_prob_top" in df.columns:
        return 1.0 - pd.to_numeric(df["alpha_prob_top"], errors="coerce")
    if "alpha_prob_top_raw" in df.columns:
        return 1.0 - pd.to_numeric(df["alpha_prob_top_raw"], errors="coerce")
    return -pd.to_numeric(base_score, errors="coerce")


def _filter_bad_names_by_day(
    tradeable_today: pd.Series,
    badness_score: pd.Series,
    dates: pd.Series,
    drop_frac: float,
) -> pd.Series:
    """
    Remove the worst daily fraction of names from the tradable set.
    This matches the current alpha model's observed strength: it is better at
    spotting likely losers than elite winners.
    """
    drop_frac = float(np.clip(drop_frac, 0.0, 0.49))
    if drop_frac <= 0:
        return tradeable_today.fillna(False).astype(bool)

    tmp = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.normalize(),
        "tradeable_today": tradeable_today.fillna(False).astype(bool),
        "badness": pd.to_numeric(badness_score, errors="coerce"),
    })

    def _keep_group(g: pd.DataFrame) -> pd.Series:
        mask = g["tradeable_today"].fillna(False).astype(bool)
        if mask.sum() <= 1:
            return mask
        # percentile rank among current tradeable names only
        ranked = g.loc[mask, "badness"].rank(method="average", pct=True, ascending=True)
        # highest badness should be dropped -> ascending rank high percentiles are worse
        keep_mask = pd.Series(False, index=g.index)
        keep_mask.loc[mask] = ranked < (1.0 - drop_frac)
        return keep_mask

    keep = tmp.groupby("date", group_keys=False).apply(_keep_group)
    keep.index = tmp.index
    return keep.fillna(False).astype(bool)


def build_policy_panel_from_alpha_predictions(alpha_pred_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in POLICY_PANEL_COLS if c not in alpha_pred_df.columns]
    if missing:
        raise KeyError(f"Missing required policy panel columns in alpha predictions: {missing}")

    use_cols = POLICY_PANEL_COLS + [c for c in OPTIONAL_SCORE_COLS if c in alpha_pred_df.columns]
    out = alpha_pred_df.loc[:, use_cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["tradeable_today"] = out["tradeable_today"].fillna(False).astype(bool)

    for c in [
        "alpha_pred",
        "tradability_score",
        "regime_prob",
        "price_rub",
        "units_sold",
        "turnover_rub",
        "price_median_30d",
        "units_median_30d",
        "turnover_median_30d",
        "alpha_rank_score_raw",
        "alpha_prob_top",
        "alpha_prob_bottom",
        "alpha_prob_top_raw",
        "alpha_prob_bottom_raw",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.drop_duplicates(subset=["date", "market_hash_name"], keep="last")
    out = out.sort_values(["date", "market_hash_name"]).reset_index(drop=True)

    base_score = _extract_base_score(out)
    badness_score = _extract_badness_score(out, base_score)

    # Preserve raw scores for debugging.
    out["alpha_pred_raw"] = pd.to_numeric(base_score, errors="coerce").astype("float32")
    out["alpha_badness_raw"] = pd.to_numeric(badness_score, errors="coerce").astype("float32")

    # New fundamental policy logic:
    # 1) remove likely losers first
    # 2) rank survivors using the remaining score
    loser_drop_frac = 0.15
    out["tradeable_today"] = _filter_bad_names_by_day(
        out["tradeable_today"],
        out["alpha_badness_raw"],
        out["date"],
        drop_frac=loser_drop_frac,
    )

    # Rank-style signals for downstream policy/backtest.
    out["alpha_rank_pct_centered"] = _rank_pct_centered(out["alpha_pred_raw"], out["date"])
    out["alpha_pred"] = _robust_scale_by_day(out["alpha_pred_raw"], out["date"])
    return out


def _policy_selection_score(summary: dict[str, Any], initial_capital: float) -> float:
    final_equity = float(summary["final_equity_rub"])
    trade_count = int(summary["trade_count"])
    sharpe = summary.get("sharpe", np.nan)
    avg_open = float(summary.get("avg_open_positions", 0.0))

    score = final_equity
    if pd.notna(sharpe):
        score += 75.0 * float(sharpe)
    score -= 1.5 * trade_count
    score -= 5.0 * avg_open

    if trade_count == 0:
        # Small penalty only. A flat no-trade result is better than a confirmed loser.
        score -= 25.0
    elif trade_count < 4:
        score -= 50.0
    return float(score)


def _admissible_mask(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    sharpe_ok = df["sharpe"].fillna(-np.inf) >= 0.0
    return (
        (df["final_equity_rub"] > initial_capital)
        & (df["trade_count"] >= 4)
        & sharpe_ok
    )


def _no_trade_policy(alpha_horizon_days: int, max_daily_turnover_fraction: float) -> dict[str, Any]:
    # This policy should result in zero buys inside the current backtester.
    return {
        "regime_prob_threshold": 1.1,
        "entry_alpha_threshold": 999.0,
        "exit_alpha_threshold": -999.0,
        "top_k": 0,
        "max_weight_per_item": 0.0,
        "max_holding_days": int(max(14, alpha_horizon_days)),
        "max_daily_turnover_fraction": float(max_daily_turnover_fraction),
    }


def calibrate_policy(panel_df: pd.DataFrame, cfg: dict[str, Any]) -> PolicyResult:
    research = cfg["research"]
    splits = cfg["splits"]
    initial_capital = float(research["initial_capital_rub"])
    alpha_horizon_days = int(research.get("primary_alpha_horizon_days", 30))

    d = pd.to_datetime(panel_df["date"]).dt.normalize()
    val_mask = (d >= pd.Timestamp(splits["val_start"])) & (d <= pd.Timestamp(splits["val_end"]))
    val_df = panel_df.loc[val_mask, BACKTEST_PANEL_COLS].copy().reset_index(drop=True)

    if len(val_df) == 0:
        raise ValueError("Validation policy panel is empty. Check split dates and upstream outputs.")

    rows: list[dict[str, Any]] = []

    # Fundamental fix: refuse ultra-short max holding windows for a 30d alpha target.
    min_allowed_hold = max(14, alpha_horizon_days // 2)

    for regime_prob, entry_alpha, exit_alpha, top_k, max_weight, max_hold in product(
        research["regime_prob_grid"],
        research["entry_alpha_grid"],
        research["exit_alpha_grid"],
        research["top_k_grid"],
        research["max_weight_grid"],
        research["max_holding_days_grid"],
    ):
        if int(max_hold) < min_allowed_hold:
            continue

        policy = {
            "regime_prob_threshold": float(regime_prob),
            "entry_alpha_threshold": float(entry_alpha),
            "exit_alpha_threshold": float(exit_alpha),
            "top_k": int(top_k),
            "max_weight_per_item": float(max_weight),
            "max_holding_days": int(max_hold),
            "max_daily_turnover_fraction": float(research["max_daily_turnover_fraction"]),
        }

        bt = run_portfolio_backtest(
            panel_df=val_df,
            policy=policy,
            initial_capital_rub=initial_capital,
            sell_cost_rate=float(research["sell_cost_rate"]),
            execute_next_day=bool(research["execute_next_day"]),
            store_details=False,
        )

        row = {**policy, **bt.summary}
        row["selection_score"] = _policy_selection_score(bt.summary, initial_capital)
        rows.append(row)

    search_table = pd.DataFrame(rows)
    if len(search_table) == 0:
        # Degenerate grid after hold filtering -> choose no-trade.
        best_policy = _no_trade_policy(alpha_horizon_days, research["max_daily_turnover_fraction"])
        validation_summary = {
            "initial_capital_rub": initial_capital,
            "final_equity_rub": initial_capital,
            "roi_pct": 0.0,
            "sharpe": None,
            "trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "avg_open_positions": 0.0,
            "max_open_positions": 0,
            "selection_score": initial_capital - 25.0,
            "admissible": False,
            "fallback_mode": "no_trade_empty_grid",
        }
        return PolicyResult(best_policy=best_policy, search_table=search_table, validation_summary=validation_summary)

    search_table["admissible"] = _admissible_mask(search_table, initial_capital)

    if bool(search_table["admissible"].any()):
        chosen_df = search_table.loc[search_table["admissible"]].sort_values(
            ["final_equity_rub", "sharpe", "trade_count", "avg_open_positions"],
            ascending=[False, False, True, True],
        )
        best_row = chosen_df.iloc[0].to_dict()
        best_policy = {
            "regime_prob_threshold": float(best_row["regime_prob_threshold"]),
            "entry_alpha_threshold": float(best_row["entry_alpha_threshold"]),
            "exit_alpha_threshold": float(best_row["exit_alpha_threshold"]),
            "top_k": int(best_row["top_k"]),
            "max_weight_per_item": float(best_row["max_weight_per_item"]),
            "max_holding_days": int(best_row["max_holding_days"]),
            "max_daily_turnover_fraction": float(best_row["max_daily_turnover_fraction"]),
        }
        validation_summary = {
            "initial_capital_rub": float(best_row["initial_capital_rub"]),
            "final_equity_rub": float(best_row["final_equity_rub"]),
            "roi_pct": float(best_row["roi_pct"]),
            "sharpe": None if pd.isna(best_row["sharpe"]) else float(best_row["sharpe"]),
            "trade_count": int(best_row["trade_count"]),
            "buy_count": int(best_row["buy_count"]),
            "sell_count": int(best_row["sell_count"]),
            "avg_open_positions": float(best_row["avg_open_positions"]),
            "max_open_positions": int(best_row["max_open_positions"]),
            "selection_score": float(best_row["selection_score"]),
            "admissible": True,
            "fallback_mode": None,
        }
        return PolicyResult(best_policy=best_policy, search_table=search_table, validation_summary=validation_summary)

    # No admissible strategy exists -> choose capital preservation explicitly.
    best_policy = _no_trade_policy(alpha_horizon_days, research["max_daily_turnover_fraction"])
    validation_summary = {
        "initial_capital_rub": initial_capital,
        "final_equity_rub": initial_capital,
        "roi_pct": 0.0,
        "sharpe": None,
        "trade_count": 0,
        "buy_count": 0,
        "sell_count": 0,
        "avg_open_positions": 0.0,
        "max_open_positions": 0,
        "selection_score": initial_capital - 25.0,
        "admissible": False,
        "fallback_mode": "no_trade_no_admissible_policy",
    }
    return PolicyResult(best_policy=best_policy, search_table=search_table, validation_summary=validation_summary)
