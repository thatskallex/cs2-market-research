#!/usr/bin/env python
"""
Portfolio concentration diagnostics for the CS2 alpha-rotation strategies.

Purpose
-------
Adds concentration-risk diagnostics for Chapter 4:
  - Herfindahl--Hirschman Index (HHI) of item weights;
  - effective number of positions, 1 / HHI;
  - capital share of the largest item and the top-3 items;
  - time-series and distribution figures for concentration risk;
  - top contributors to capital concentration.

Run from repo root:
    python scripts/21_portfolio_concentration_analysis.py --config configs/cs2_trading_rebuilt.yaml

Typical focused run for the best high-cost candidate:
    python scripts/21_portfolio_concentration_analysis.py \
        --config configs/cs2_trading_rebuilt.yaml \
        --strategies alpha_top_10pct_rebalance_180d \
        --sell-cost-rate 0.10

The script simulates the same next-day alpha rotation logic used in the proof
scripts, but stores daily item-level weights so concentration can be measured.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
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

DEFAULT_STRATEGIES = [
    "alpha_top_10pct_rebalance_90d",
    "alpha_top_10pct_rebalance_180d",
    "alpha_top_20pct_rebalance_180d",
    "alpha_top_10pct_rebalance_365d",
]


def _safe_name(name: str) -> str:
    name = name.lower().replace("%", "pct")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _parse_strategy(strategy: str) -> dict[str, Any]:
    if strategy == "buy_and_hold_all":
        return {"strategy": strategy, "selection_mode": "all", "top_pct": None, "rebalance_days": 10**9}
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d$", strategy)
    if not m:
        raise ValueError(
            f"Unknown strategy name: {strategy}. Expected buy_and_hold_all or alpha_top_<N>pct_rebalance_<D>d"
        )
    top, days = m.groups()
    return {
        "strategy": strategy,
        "selection_mode": "top_pct",
        "top_pct": float(top) / 100.0,
        "rebalance_days": int(days),
    }


def _strategy_label_ru(strategy: str) -> str:
    if strategy == "buy_and_hold_all":
        return "Пассивное удержание"
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d$", strategy)
    if m:
        top, days = m.groups()
        return f"Альфа top {top}%, ребалансировка {days} дн."
    return strategy


def _strategy_label_short_ru(strategy: str) -> str:
    """Compact label for legends and horizontal axes."""
    if strategy == "buy_and_hold_all":
        return "Buy-and-hold"
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d$", strategy)
    if m:
        top, days = m.groups()
        return f"Top {top}%, {days} дн."
    return strategy


def _split_ranges(cfg: dict[str, Any] | None) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if not cfg or "splits" not in cfg:
        return []
    s = cfg["splits"]
    return [
        ("Обучение", pd.Timestamp(s["train_start"]).normalize(), pd.Timestamp(s["train_end"]).normalize()),
        ("Валидация", pd.Timestamp(s["val_start"]).normalize(), pd.Timestamp(s["val_end"]).normalize()),
        ("Тест", pd.Timestamp(s["test_start"]).normalize(), pd.Timestamp(s["test_end"]).normalize()),
    ]


def _shade_splits(ax, cfg: dict[str, Any] | None, y_text: bool = False) -> None:
    colors = ["#F2F6FC", "#FFF8E6", "#F1FAF3"]
    for i, (name, start, end) in enumerate(_split_ranges(cfg)):
        ax.axvspan(start, end, color=colors[i % len(colors)], alpha=0.52, linewidth=0, zorder=0)
        if i > 0:
            ax.axvline(start, color="0.35", linestyle="--", linewidth=0.9, alpha=0.70, zorder=1)
        if y_text:
            ymin, ymax = ax.get_ylim()
            ax.text(start + (end - start) / 2, ymax, name, ha="center", va="top", fontsize=8.5, color="0.32")


def _strategy_color_map(strategies: list[str]) -> dict[str, str]:
    palette = ["#2F6B9A", "#D9822B", "#3D8B40", "#9A3D6A", "#6B5B95", "#7B7B7B"]
    return {s: palette[i % len(palette)] for i, s in enumerate(strategies)}


def _short_item_name(name: str, max_len: int = 46) -> str:
    s = str(name)
    s = s.replace(" StatTrak™", " ST")
    s = s.replace("The Professionals", "Prof.")
    s = s.replace("Field-Tested", "FT").replace("Minimal Wear", "MW").replace("Battle-Scarred", "BS")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _load_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    alpha_path = resolve_path(cfg, "models_dir") / "alpha_predictions.parquet"
    if not alpha_path.exists():
        raise FileNotFoundError(f"alpha_predictions.parquet not found at {alpha_path}")

    universe = pd.read_parquet(universe_path, columns=["date", "market_hash_name", "tradeable_today"])
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["tradeable_today"] = universe["tradeable_today"].fillna(False).astype(bool)
    universe = universe.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    prices = pd.read_parquet(feature_path, columns=["date", "market_hash_name", "price_rub"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["price_rub"] = pd.to_numeric(prices["price_rub"], errors="coerce")
    prices = prices.dropna(subset=["price_rub"])
    prices = prices.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    alpha = pd.read_parquet(alpha_path, columns=["date", "market_hash_name", "alpha_pred"])
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha["alpha_pred"] = pd.to_numeric(alpha["alpha_pred"], errors="coerce")
    alpha = alpha.dropna(subset=["alpha_pred"])
    alpha = alpha.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    price_wide = prices.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    price_wide_mtm = price_wide.ffill()
    return universe, alpha, price_wide, price_wide_mtm


def _make_sim_dates(price_wide: pd.DataFrame, universe: pd.DataFrame, splits: dict[str, Any]) -> pd.DatetimeIndex:
    universe_dates = set(universe.loc[universe["tradeable_today"], "date"].unique())
    sim_dates = pd.DatetimeIndex(sorted(set(price_wide.index) & universe_dates))
    sim_start = pd.Timestamp(splits["train_start"]).normalize()
    sim_end = pd.Timestamp(splits["test_end"]).normalize()
    sim_dates = sim_dates[(sim_dates >= sim_start) & (sim_dates <= sim_end)]
    if len(sim_dates) == 0:
        raise ValueError("No simulation dates after intersecting price, universe, and split ranges.")
    return sim_dates


def _build_daily_scores(universe: pd.DataFrame, alpha: pd.DataFrame, sim_dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.DataFrame]:
    df = universe.loc[universe["tradeable_today"], ["date", "market_hash_name"]].merge(
        alpha[["date", "market_hash_name", "alpha_pred"]],
        on=["date", "market_hash_name"],
        how="left",
    )
    df = df[df["date"].isin(sim_dates)].dropna(subset=["alpha_pred"])
    return {pd.Timestamp(d): g[["market_hash_name", "alpha_pred"]].copy() for d, g in df.groupby("date", sort=True)}


def _target_items(scores: pd.DataFrame, selection_mode: str, top_pct: float | None) -> list[str]:
    if selection_mode == "all":
        return scores["market_hash_name"].dropna().astype(str).tolist()
    if selection_mode == "top_pct":
        scored = scores.dropna(subset=["alpha_pred"]).sort_values(
            ["alpha_pred", "market_hash_name"], ascending=[False, True]
        )
        if len(scored) == 0:
            return []
        k = max(1, int(math.ceil(len(scored) * float(top_pct))))
        return scored.head(k)["market_hash_name"].astype(str).tolist()
    raise ValueError(f"Unknown selection mode: {selection_mode}")


def _next_exec_date(sim_dates: pd.DatetimeIndex, current_idx: int, execute_next_day: bool) -> pd.Timestamp | None:
    if not execute_next_day:
        return pd.Timestamp(sim_dates[current_idx])
    if current_idx + 1 >= len(sim_dates):
        return None
    return pd.Timestamp(sim_dates[current_idx + 1])


def _calc_concentration_for_day(
    d: pd.Timestamp,
    positions: dict[str, int],
    cash: float,
    price_wide_mtm: pd.DataFrame,
    strategy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values: list[tuple[str, float]] = []
    for item, units in positions.items():
        if item not in price_wide_mtm.columns:
            continue
        px = price_wide_mtm.at[d, item]
        if pd.notna(px) and px > 0 and units > 0:
            values.append((item, float(units) * float(px)))

    invested = float(sum(v for _, v in values))
    total_equity = invested + float(cash)
    detail_rows: list[dict[str, Any]] = []

    if invested <= 0:
        row = {
            "date": d,
            "strategy": strategy,
            "cash_rub": float(cash),
            "invested_value_rub": invested,
            "equity_rub": total_equity,
            "cash_share": 1.0 if total_equity > 0 else np.nan,
            "n_positions": 0,
            "hhi": np.nan,
            "hhi_points": np.nan,
            "effective_n_positions": np.nan,
            "top1_share": np.nan,
            "top3_share": np.nan,
            "top5_share": np.nan,
            "largest_item": None,
        }
        return row, detail_rows

    values = sorted(values, key=lambda x: x[1], reverse=True)
    shares = np.array([v / invested for _, v in values], dtype=float)
    hhi = float(np.sum(shares**2))
    effective_n = float(1.0 / hhi) if hhi > 0 else np.nan
    row = {
        "date": d,
        "strategy": strategy,
        "cash_rub": float(cash),
        "invested_value_rub": invested,
        "equity_rub": total_equity,
        "cash_share": float(cash / total_equity) if total_equity > 0 else np.nan,
        "n_positions": int(len(values)),
        "hhi": hhi,
        "hhi_points": float(hhi * 10000.0),
        "effective_n_positions": effective_n,
        "top1_share": float(np.sum(shares[:1])),
        "top3_share": float(np.sum(shares[:3])),
        "top5_share": float(np.sum(shares[:5])),
        "largest_item": values[0][0] if values else None,
    }
    for rank, ((item, value), share) in enumerate(zip(values, shares), start=1):
        detail_rows.append({
            "date": d,
            "strategy": strategy,
            "market_hash_name": item,
            "rank_by_weight": int(rank),
            "position_value_rub": float(value),
            "weight_invested": float(share),
        })
    return row, detail_rows


def simulate_strategy_concentration(
    strategy_cfg: dict[str, Any],
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    daily_scores: dict[pd.Timestamp, pd.DataFrame],
    sim_dates: pd.DatetimeIndex,
    initial_capital: float,
    sell_cost_rate: float,
    execute_next_day: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = strategy_cfg["strategy"]
    selection_mode = strategy_cfg["selection_mode"]
    top_pct = strategy_cfg.get("top_pct")
    rebalance_days = int(strategy_cfg["rebalance_days"])

    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    pending_targets: dict[pd.Timestamp, list[list[str]]] = {}
    last_signal_date: pd.Timestamp | None = None
    concentration_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    trades_rows: list[dict[str, Any]] = []

    def px_sell(d: pd.Timestamp, it: str) -> float | None:
        if it not in price_cols:
            return None
        px = price_wide_mtm.at[d, it]
        if pd.notna(px) and px > 0:
            return float(px)
        return None

    def px_buy(d: pd.Timestamp, it: str) -> float | None:
        if it not in price_cols:
            return None
        px = price_wide.at[d, it]
        if pd.notna(px) and px > 0:
            return float(px)
        return None

    def execute_target(d: pd.Timestamp, target: list[str]) -> None:
        nonlocal cash, positions
        target = sorted(set([it for it in target if it in price_cols and px_buy(d, it) is not None]))
        if not target:
            return
        current_mv = 0.0
        for it, units in positions.items():
            px = px_sell(d, it)
            if px is not None:
                current_mv += float(units) * px
        equity = cash + current_mv
        if equity <= 0:
            return
        budget = equity / len(target)
        desired: dict[str, int] = {}
        for it in target:
            px = px_buy(d, it)
            if px is None:
                continue
            desired[it] = int(budget // px)

        # Sell or trim first.
        for it in list(positions.keys()):
            old_units = int(positions[it])
            new_units = int(desired.get(it, 0))
            if new_units >= old_units:
                continue
            units_to_sell = old_units - new_units
            px = px_sell(d, it)
            if px is None:
                continue
            gross = units_to_sell * px
            net = gross * (1.0 - sell_cost_rate)
            cash += net
            if new_units > 0:
                positions[it] = new_units
            else:
                del positions[it]
            trades_rows.append({
                "date": d,
                "strategy": strategy,
                "side": "sell",
                "market_hash_name": it,
                "units": int(units_to_sell),
                "price_rub": px,
                "gross_value_rub": float(gross),
                "net_value_rub": float(net),
                "cost_paid_rub": float(gross - net),
            })

        # Buy or top up.
        for it, target_units in desired.items():
            old_units = int(positions.get(it, 0))
            if target_units <= old_units:
                continue
            px = px_buy(d, it)
            if px is None:
                continue
            units_to_buy = min(target_units - old_units, int(cash // px))
            if units_to_buy <= 0:
                continue
            cost = units_to_buy * px
            cash -= cost
            positions[it] = old_units + units_to_buy
            trades_rows.append({
                "date": d,
                "strategy": strategy,
                "side": "buy",
                "market_hash_name": it,
                "units": int(units_to_buy),
                "price_rub": px,
                "gross_value_rub": float(cost),
                "net_value_rub": float(cost),
                "cost_paid_rub": 0.0,
            })

    for i, d0 in enumerate(sim_dates):
        d = pd.Timestamp(d0)
        for target in pending_targets.pop(d, []):
            execute_target(d, target)

        row, details = _calc_concentration_for_day(d, positions, cash, price_wide_mtm, strategy)
        concentration_rows.append(row)
        holdings_rows.extend(details)

        due = last_signal_date is None or (rebalance_days > 0 and (d - last_signal_date).days >= rebalance_days)
        if not due:
            continue
        scores = daily_scores.get(d)
        if scores is None or len(scores) == 0:
            continue
        target = _target_items(scores, selection_mode, top_pct)
        if not target:
            continue
        exec_date = _next_exec_date(sim_dates, i, execute_next_day)
        if exec_date is None:
            continue
        pending_targets.setdefault(exec_date, []).append(target)
        last_signal_date = d

    return pd.DataFrame(concentration_rows), pd.DataFrame(holdings_rows), pd.DataFrame(trades_rows)


def _summary_table(concentration: pd.DataFrame, sell_cost_rate: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, g in concentration.groupby("strategy", sort=False):
        gg = g[g["n_positions"] > 0].copy()
        if len(gg) == 0:
            continue
        rows.append({
            "Стратегия": _strategy_label_ru(strategy),
            "Издержки продажи, %": sell_cost_rate * 100.0,
            "Дней с позициями": int(len(gg)),
            "Медианное число позиций": float(gg["n_positions"].median()),
            "Медианный HHI": float(gg["hhi_points"].median()),
            "Средний HHI": float(gg["hhi_points"].mean()),
            "90-й перцентиль HHI": float(gg["hhi_points"].quantile(0.90)),
            "Максимальный HHI": float(gg["hhi_points"].max()),
            "Медианное эффективное число позиций": float(gg["effective_n_positions"].median()),
            "Медианная доля top-1, %": float(gg["top1_share"].median() * 100.0),
            "Медианная доля top-3, %": float(gg["top3_share"].median() * 100.0),
            "90-й перцентиль доли top-3, %": float(gg["top3_share"].quantile(0.90) * 100.0),
            "Максимальная доля top-3, %": float(gg["top3_share"].max() * 100.0),
            "Дней с top-3 > 50%, %": float((gg["top3_share"] > 0.50).mean() * 100.0),
            "Дней с HHI > 2500, %": float((gg["hhi_points"] > 2500).mean() * 100.0),
        })
    return pd.DataFrame(rows)


def _top_exposures(holdings: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if len(holdings) == 0:
        return pd.DataFrame()
    rows = []
    for strategy, g in holdings.groupby("strategy", sort=False):
        dates = int(g["date"].nunique())
        agg = g.groupby("market_hash_name").agg(
            days_held=("date", "nunique"),
            avg_weight_on_held_days=("weight_invested", "mean"),
            max_weight=("weight_invested", "max"),
            capital_days_weight=("weight_invested", "sum"),
        ).reset_index()
        agg["avg_weight_all_days"] = agg["capital_days_weight"] / max(dates, 1)
        agg = agg.sort_values(["avg_weight_all_days", "max_weight"], ascending=False).head(top_n)
        for _, r in agg.iterrows():
            rows.append({
                "Стратегия": _strategy_label_ru(strategy),
                "Предмет": r["market_hash_name"],
                "Дней в портфеле": int(r["days_held"]),
                "Средний вес в дни владения, %": float(r["avg_weight_on_held_days"] * 100.0),
                "Средний вес по всем дням, %": float(r["avg_weight_all_days"] * 100.0),
                "Максимальный вес, %": float(r["max_weight"] * 100.0),
            })
    return pd.DataFrame(rows)


def _setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 11.2,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.18",
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt

def _savefig(plt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_time_series(conc: pd.DataFrame, reports_dir: Path, cfg: dict[str, Any] | None = None) -> Path:
    """Two-panel time-series figure with split shading and an external legend."""
    plt = _setup_matplotlib()
    strategies = conc["strategy"].dropna().unique().tolist()
    colors = _strategy_color_map(strategies)
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.2), sharex=True, gridspec_kw={"hspace": 0.12})

    for ax in axes:
        _shade_splits(ax, cfg, y_text=False)

    for strategy in strategies:
        g = conc[(conc["strategy"] == strategy) & (conc["n_positions"] > 0)].sort_values("date")
        if len(g) == 0:
            continue
        label = _strategy_label_short_ru(strategy)
        axes[0].plot(g["date"], g["hhi_points"], linewidth=2.1, color=colors[strategy], label=label, zorder=3)
        axes[1].plot(g["date"], g["top3_share"] * 100.0, linewidth=2.1, color=colors[strategy], label=label, zorder=3)

    axes[0].axhline(2500, color="0.20", linestyle=(0, (4, 3)), linewidth=1.05, alpha=0.85, zorder=2)
    axes[0].annotate("Порог высокой концентрации HHI = 2500", xy=(conc["date"].min(), 2500), xytext=(10, 6), textcoords="offset points", fontsize=9, color="0.25")
    axes[0].set_ylabel("HHI, пунктов")

    axes[1].axhline(50, color="0.20", linestyle=(0, (4, 3)), linewidth=1.05, alpha=0.85, zorder=2)
    axes[1].annotate("Доля top-3 = 50%", xy=(conc["date"].min(), 50), xytext=(10, 6), textcoords="offset points", fontsize=9, color="0.25")
    axes[1].set_ylabel("Доля капитала в top-3, %")
    axes[1].set_xlabel("Дата")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.22)
        ax.grid(True, axis="x", alpha=0.10)

    # Split labels at top of figure after limits are settled.
    for name, start, end in _split_ranges(cfg):
        mid = start + (end - start) / 2
        axes[0].text(mid, 1.01, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8.8, color="0.32")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(4, len(labels)), frameon=True, fancybox=False, edgecolor="0.75")
    fig.suptitle("Концентрация модельных портфелей во времени", y=1.02)
    out = reports_dir / "Рис_4_9_HHI_и_доля_top3_во_времени.png"
    _savefig(plt, out)
    return out


def _plot_distribution(conc: pd.DataFrame, reports_dir: Path) -> Path:
    """Horizontal boxplots: easier to read than rotated x-labels."""
    plt = _setup_matplotlib()
    valid = conc[conc["n_positions"] > 0].copy()
    med_order = valid.groupby("strategy")["hhi_points"].median().sort_values(ascending=False).index.tolist()
    labels = [_strategy_label_short_ru(s) for s in med_order]
    colors = _strategy_color_map(med_order)
    hhi_data = [valid[valid["strategy"] == s]["hhi_points"].dropna().values for s in med_order]
    top3_data = [valid[valid["strategy"] == s]["top3_share"].dropna().values * 100.0 for s in med_order]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(4.8, 0.48 * len(med_order) + 3.0)), sharey=True)
    bp0 = axes[0].boxplot(hhi_data, labels=labels, showfliers=False, patch_artist=True, vert=False, widths=0.62)
    bp1 = axes[1].boxplot(top3_data, labels=labels, showfliers=False, patch_artist=True, vert=False, widths=0.62)
    for bp, axis in [(bp0, axes[0]), (bp1, axes[1])]:
        for patch, s in zip(bp["boxes"], med_order):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.42)
            patch.set_edgecolor(colors[s])
        for med in bp["medians"]:
            med.set_color("#222222")
            med.set_linewidth(1.6)
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set_color("0.30")
            whisk.set_linewidth(1.0)
        axis.grid(True, axis="x", alpha=0.22)

    axes[0].axvline(2500, color="0.20", linestyle=(0, (4, 3)), linewidth=1.05)
    axes[0].set_title("HHI")
    axes[0].set_xlabel("Пункты")
    axes[0].annotate("2500", xy=(2500, len(med_order) + 0.55), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="0.25")

    axes[1].axvline(50, color="0.20", linestyle=(0, (4, 3)), linewidth=1.05)
    axes[1].set_title("Доля капитала в top-3")
    axes[1].set_xlabel("% инвестированной части")
    axes[1].annotate("50%", xy=(50, len(med_order) + 0.55), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="0.25")

    fig.suptitle("Распределение показателей концентрации по стратегиям", y=1.02)
    out = reports_dir / "Рис_4_10_распределение_показателей_концентрации.png"
    _savefig(plt, out)
    return out


def _plot_effective_n(conc: pd.DataFrame, reports_dir: Path, cfg: dict[str, Any] | None = None) -> Path:
    plt = _setup_matplotlib()
    strategies = conc["strategy"].dropna().unique().tolist()
    colors = _strategy_color_map(strategies)
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    _shade_splits(ax, cfg, y_text=False)
    for strategy in strategies:
        g = conc[(conc["strategy"] == strategy) & (conc["n_positions"] > 0)].sort_values("date")
        if len(g):
            ax.plot(g["date"], g["effective_n_positions"], linewidth=2.1, color=colors[strategy], label=_strategy_label_short_ru(strategy), zorder=3)
    ax.axhline(4, color="0.20", linestyle=(0, (4, 3)), linewidth=1.05, alpha=0.85)
    ax.annotate("HHI = 2500 соответствует 4 равновесным позициям", xy=(conc["date"].min(), 4), xytext=(10, 7), textcoords="offset points", fontsize=9, color="0.25")
    ax.set_title("Эффективное число позиций в модельном портфеле")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Эффективное число позиций, 1 / HHI")
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(4, len(strategies)), frameon=True, fancybox=False, edgecolor="0.75")
    out = reports_dir / "Рис_4_11_эффективное_число_позиций.png"
    _savefig(plt, out)
    return out


def _plot_concentration_profile(conc: pd.DataFrame, reports_dir: Path) -> Path:
    """Compact academic dashboard: median HHI, median top-3 and median effective N."""
    plt = _setup_matplotlib()
    valid = conc[conc["n_positions"] > 0].copy()
    prof = valid.groupby("strategy", as_index=False).agg(
        hhi_med=("hhi_points", "median"),
        top3_med=("top3_share", "median"),
        effn_med=("effective_n_positions", "median"),
    )
    prof = prof.sort_values("hhi_med", ascending=False)
    labels = [_strategy_label_short_ru(s) for s in prof["strategy"]]
    y = np.arange(len(prof))
    colors = _strategy_color_map(prof["strategy"].tolist())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, max(4.2, 0.45 * len(prof) + 2.5)), sharey=True)
    metrics = [
        ("hhi_med", "Медианный HHI", "пунктов", 2500),
        ("top3_med", "Медианная доля top-3", "%", 0.50),
        ("effn_med", "Медианное эффективное число позиций", "позиций", 4),
    ]
    for ax, (col, title, unit, ref) in zip(axes, metrics):
        vals = prof[col].astype(float).values.copy()
        if col == "top3_med":
            vals = vals * 100.0
            ref = 50
        ax.barh(y, vals, color=[colors[s] for s in prof["strategy"]], alpha=0.72)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.grid(True, axis="x", alpha=0.20)
        if ref is not None:
            ax.axvline(ref, color="0.20", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
        for yy, v in zip(y, vals):
            ax.text(v, yy, f" {v:.1f}", va="center", fontsize=8.5, color="0.25")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Сводный профиль концентрационного риска", y=1.02)
    out = reports_dir / "Рис_4_13_сводный_профиль_концентрационного_риска.png"
    _savefig(plt, out)
    return out


def _plot_weight_heatmap(holdings: pd.DataFrame, reports_dir: Path, strategy: str, top_n: int = 12) -> Path | None:
    if len(holdings) == 0:
        return None
    g = holdings[holdings["strategy"] == strategy].copy()
    if len(g) == 0:
        return None
    top_items = (
        g.groupby("market_hash_name")["weight_invested"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    gg = g[g["market_hash_name"].isin(top_items)]
    pivot = gg.pivot_table(index="market_hash_name", columns="date", values="weight_invested", aggfunc="sum").fillna(0.0)
    pivot = pivot.loc[top_items]
    if pivot.shape[1] > 220:
        step = int(math.ceil(pivot.shape[1] / 220))
        pivot = pivot.iloc[:, ::step]

    plt = _setup_matplotlib()
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="white")
    data = np.ma.masked_where(pivot.values <= 0, pivot.values * 100.0)
    fig, ax = plt.subplots(figsize=(13.2, max(5.4, 0.42 * len(pivot) + 2.2)))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.0, float(np.nanpercentile(pivot.values * 100.0, 98))))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_short_item_name(x) for x in pivot.index], fontsize=8.4)
    x_idx = np.linspace(0, pivot.shape[1] - 1, min(9, pivot.shape[1])).astype(int)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([pd.Timestamp(pivot.columns[i]).strftime("%m.%Y") for i in x_idx], rotation=0)
    ax.set_title(f"Карта весов крупнейших предметов: {_strategy_label_ru(strategy)}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Предмет")
    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Вес в инвестированной части портфеля, %")
    fig.text(0.012, 0.012, "Белые области означают отсутствие предмета в портфеле; цвет отражает вес только в инвестированной части.", fontsize=9, color="0.30")
    out = reports_dir / f"Рис_4_12_карта_весов_крупнейших_предметов_{_safe_name(strategy)}.png"
    _savefig(plt, out)
    return out

def _write_markdown(reports_dir: Path, summary: pd.DataFrame, exposures: pd.DataFrame, figures: list[Path]) -> None:
    lines = [
        "# Анализ концентрации модельного портфеля",
        "",
        "## Таблица 4.x. Показатели концентрации портфеля",
        "",
        summary.to_markdown(index=False) if len(summary) else "Нет данных.",
        "",
        "## Таблица 4.x. Предметы с наибольшим вкладом в концентрацию",
        "",
        exposures.to_markdown(index=False) if len(exposures) else "Нет данных.",
        "",
        "## Рисунки",
        "",
    ]
    for p in figures:
        lines.append(f"- `{p.name}`")
    (reports_dir / "21_анализ_концентрации_портфеля.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    ap.add_argument("--sell-cost-rate", type=float, default=0.10)
    ap.add_argument("--same-day-execution", action="store_true")
    ap.add_argument("--top-heatmap-strategy", default="alpha_top_10pct_rebalance_180d")
    args = ap.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    initial_capital = float(cfg.get("research", {}).get("initial_capital_rub", 50000.0))
    execute_next_day = not bool(args.same_day_execution)

    universe, alpha, price_wide, price_wide_mtm = _load_inputs(cfg)
    sim_dates = _make_sim_dates(price_wide, universe, cfg["splits"])
    daily_scores = _build_daily_scores(universe, alpha, sim_dates)

    all_conc: list[pd.DataFrame] = []
    all_holdings: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    for strategy in args.strategies:
        print(f"Simulating concentration diagnostics: {strategy}")
        strat_cfg = _parse_strategy(strategy)
        c, h, t = simulate_strategy_concentration(
            strategy_cfg=strat_cfg,
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            daily_scores=daily_scores,
            sim_dates=sim_dates,
            initial_capital=initial_capital,
            sell_cost_rate=float(args.sell_cost_rate),
            execute_next_day=execute_next_day,
        )
        all_conc.append(c)
        all_holdings.append(h)
        all_trades.append(t)

    concentration = pd.concat(all_conc, ignore_index=True) if all_conc else pd.DataFrame()
    holdings = pd.concat(all_holdings, ignore_index=True) if all_holdings else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    suffix = f"cost_{int(round(args.sell_cost_rate * 10000))}bp"
    concentration.to_csv(reports_dir / f"21_portfolio_concentration_daily_{suffix}.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(reports_dir / f"21_portfolio_item_weights_daily_{suffix}.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(reports_dir / f"21_portfolio_concentration_trades_{suffix}.csv", index=False, encoding="utf-8-sig")

    summary = _summary_table(concentration, float(args.sell_cost_rate))
    exposures = _top_exposures(holdings, top_n=15)
    summary.to_csv(reports_dir / f"T4_x_показатели_концентрации_портфеля_{suffix}.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(reports_dir / f"T4_x_крупнейшие_предметы_по_вкладу_в_концентрацию_{suffix}.csv", index=False, encoding="utf-8-sig")

    figures: list[Path] = []
    if len(concentration):
        figures.append(_plot_time_series(concentration, reports_dir, cfg=cfg))
        figures.append(_plot_distribution(concentration, reports_dir))
        figures.append(_plot_effective_n(concentration, reports_dir, cfg=cfg))
        figures.append(_plot_concentration_profile(concentration, reports_dir))
    if len(holdings):
        fig = _plot_weight_heatmap(holdings, reports_dir, strategy=args.top_heatmap_strategy)
        if fig is not None:
            figures.append(fig)

    _write_markdown(reports_dir, summary, exposures, figures)
    print("Saved concentration diagnostics to", reports_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
