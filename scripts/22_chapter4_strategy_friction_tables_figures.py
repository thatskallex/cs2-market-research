#!/usr/bin/env python
"""
Таблицы 4.6--4.8 и рисунки 4.4--4.7 для раздела о стратегиях,
издержках и устойчивости alpha rotation.

Скрипт читает результаты 19_alpha_rotation_friction_sweep.py и формирует:
  T4_6_метрики_стратегий_при_10pct_sell_cost.{csv,md}
  T4_7_ROI_стратегий_по_уровням_sell_cost.{csv,md}
  T4_8_break_even_cost_по_стратегиям.{csv,md}
  Рис_4_4_equity_curves_стратегий_при_10pct.png
  Рис_4_5_heatmap_excess_ROI_vs_passive_10pct.png
  Рис_4_6_cost_sensitivity_curves.png
  Рис_4_7_turnover_ROI_scatter_10pct.png

Запуск:
  python scripts/22_chapter4_strategy_friction_tables_figures.py \
      --config configs/cs2_trading_rebuilt.yaml

Если equity parquet-файлов нет, скрипт сам пересчитает equity-кривые для c=10%
через функции из scripts/19_alpha_rotation_friction_sweep.py.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.utils import save_table


ALPHA_TOP_PCTS = [0.05, 0.10, 0.20, 0.30]
REBALANCE_DAYS = [30, 90, 180, 365]
COST_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
TARGET_COST = 0.10
ANNUALIZATION_DAYS = 365


def _fmt_pct_value(x: Any, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"


def _fmt_cost(x: Any, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def _pct_label(cost: float) -> str:
    pct = cost * 100
    return f"{pct:g}%"


def _strategy_name(strategy: str) -> str:
    if strategy == "buy_and_hold_all":
        return "Пассивное удержание"
    import re
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d", str(strategy))
    if m:
        top, days = m.groups()
        return f"Alpha top {top}%, ребал. {days} дн."
    if strategy.startswith("all_universe_rebalance_"):
        days = strategy.split("_")[-1].replace("d", "")
        return f"Пассивная ребал. {days} дн."
    return str(strategy)


def _safe_name(strategy: str) -> str:
    # Must match 19_alpha_rotation_friction_sweep._safe_name for strategy names used here.
    import re
    name = str(strategy).lower().replace("%", "pct")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _cost_label(cost: float) -> str:
    # Must match 19_alpha_rotation_friction_sweep._cost_label.
    return f"cost_{int(round(float(cost) * 10000))}bp"


def _all_report_strategies() -> list[str]:
    out = ["buy_and_hold_all"]
    for days in REBALANCE_DAYS:
        for pct in ALPHA_TOP_PCTS:
            out.append(f"alpha_top_{int(round(pct * 100))}pct_rebalance_{days}d")
    return out


def _strategy_order_key(strategy: str) -> tuple[int, int, int]:
    if strategy == "buy_and_hold_all":
        return (0, 0, 0)
    import re
    m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d", str(strategy))
    if m:
        top, days = map(int, m.groups())
        return (1, days, top)
    return (9, 9999, 9999)


def _annualized_turnover(row: pd.Series) -> float | None:
    n_days = pd.to_numeric(row.get("n_days"), errors="coerce")
    turnover = pd.to_numeric(row.get("turnover_to_initial_capital"), errors="coerce")
    if pd.isna(n_days) or n_days <= 0 or pd.isna(turnover):
        return None
    return float(turnover * ANNUALIZATION_DAYS / n_days)


def _sortino(equity: pd.Series, annualization_days: int = ANNUALIZATION_DAYS) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    ret = eq.pct_change().dropna()
    if len(ret) < 2:
        return None
    downside = ret[ret < 0]
    if len(downside) < 2:
        return None
    downside_std = float(downside.std(ddof=1))
    if not np.isfinite(downside_std) or downside_std <= 0:
        return None
    return float(math.sqrt(annualization_days) * float(ret.mean()) / downside_std)


def _sharpe(equity: pd.Series, annualization_days: int = ANNUALIZATION_DAYS) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    ret = eq.pct_change().dropna()
    if len(ret) < 2:
        return None
    sd = float(ret.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(math.sqrt(annualization_days) * float(ret.mean()) / sd)


def _max_drawdown_pct(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) == 0:
        return None
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min() * 100.0)


def _roi_pct(start: float, end: float) -> float | None:
    if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
        return None
    return float((end / start - 1.0) * 100.0)


def _load_friction_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("friction_sweep_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_reports(reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_path_parquet = reports_dir / "19_strategy_cost_grid.parquet"
    grid_path_csv = reports_dir / "19_strategy_cost_grid.csv"
    be_path_parquet = reports_dir / "19_break_even_table.parquet"
    be_path_csv = reports_dir / "19_break_even_table.csv"

    if grid_path_parquet.exists():
        grid = pd.read_parquet(grid_path_parquet)
    elif grid_path_csv.exists():
        grid = pd.read_csv(grid_path_csv)
    else:
        raise FileNotFoundError(
            "Не найден 19_strategy_cost_grid. Сначала запусти scripts/19_alpha_rotation_friction_sweep.py"
        )

    if be_path_parquet.exists():
        break_even = pd.read_parquet(be_path_parquet)
    elif be_path_csv.exists():
        break_even = pd.read_csv(be_path_csv)
    else:
        raise FileNotFoundError(
            "Не найден 19_break_even_table. Сначала запусти scripts/19_alpha_rotation_friction_sweep.py"
        )

    grid["sell_cost_rate"] = pd.to_numeric(grid["sell_cost_rate"], errors="coerce")
    return grid, break_even


def _select_grid(grid: pd.DataFrame) -> pd.DataFrame:
    strategies = set(_all_report_strategies())
    out = grid[grid["strategy"].isin(strategies)].copy()
    out["order_key"] = out["strategy"].map(_strategy_order_key)
    out = out.sort_values(["order_key", "sell_cost_rate"]).drop(columns="order_key")
    return out


def _load_or_simulate_equities(
    cfg: dict[str, Any],
    strategies: list[str],
    cost: float,
    backtests_dir: Path,
    friction_script: Path,
    force_resim: bool = False,
) -> dict[str, pd.DataFrame]:
    equities: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    label = _cost_label(cost)
    for s in strategies:
        path = backtests_dir / f"friction_{label}_{_safe_name(s)}_equity.parquet"
        if path.exists() and not force_resim:
            eq = pd.read_parquet(path)
            eq["date"] = pd.to_datetime(eq["date"]).dt.normalize()
            equities[s] = eq
        else:
            missing.append(s)

    if not missing:
        return equities

    print("Equity parquet не найден для части стратегий; пересчитываю c=10% для нужных кривых...")
    mod = _load_friction_module(friction_script)
    universe, alpha, price_wide, price_wide_mtm = mod._load_inputs(cfg)
    sim_dates = mod._make_sim_dates(price_wide, universe, cfg["splits"])
    daily_scores = mod._build_daily_tradeable_scores(universe, alpha, sim_dates)
    initial_capital = float(cfg["research"]["initial_capital_rub"])
    execute_next_day = True

    for s in missing:
        if s == "buy_and_hold_all":
            selection_mode = "all"
            top_pct = None
            rebalance_days = 10**9
        else:
            import re
            m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d", s)
            if not m:
                continue
            top_pct = float(m.group(1)) / 100.0
            rebalance_days = int(m.group(2))
            selection_mode = "top_pct"
        eq, tr, _stats = mod.simulate_rebalance_with_costs(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            daily_scores=daily_scores,
            sim_dates=sim_dates,
            initial_capital=initial_capital,
            rebalance_days=rebalance_days,
            selection_mode=selection_mode,
            sell_cost_rate=cost,
            top_pct=top_pct,
            execute_next_day=execute_next_day,
        )
        eq["date"] = pd.to_datetime(eq["date"]).dt.normalize()
        equities[s] = eq
        try:
            save_table(eq, backtests_dir / f"friction_{label}_{_safe_name(s)}_equity.parquet")
            save_table(tr, backtests_dir / f"friction_{label}_{_safe_name(s)}_trades.parquet")
        except Exception:
            pass
    return equities




def _simulate_test_fresh_equities(
    cfg: dict[str, Any],
    strategies: list[str],
    cost: float,
    friction_script: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Simulate each strategy from cash at the beginning of the test interval.

    This is different from slicing a full-period backtest. Slicing inherits positions
    accumulated before test_start, while a fresh out-of-sample table should start with
    the same initial cash and no open positions at test_start, matching the pipeline
    decomposition table.
    """
    mod = _load_friction_module(friction_script)
    universe, alpha, price_wide, price_wide_mtm = mod._load_inputs(cfg)
    all_dates = mod._make_sim_dates(price_wide, universe, cfg["splits"])
    test_start = pd.Timestamp(cfg["splits"]["test_start"]).normalize()
    test_end = pd.Timestamp(cfg["splits"]["test_end"]).normalize()
    test_dates = pd.DatetimeIndex([d for d in all_dates if test_start <= pd.Timestamp(d) <= test_end])
    if len(test_dates) == 0:
        raise ValueError("No test simulation dates found for fresh test backtest.")
    daily_scores = mod._build_daily_tradeable_scores(universe, alpha, test_dates)
    initial_capital = float(cfg["research"]["initial_capital_rub"])
    equities: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}

    for s in strategies:
        if s == "buy_and_hold_all":
            selection_mode = "all"
            top_pct = None
            rebalance_days = 10**9
        else:
            import re
            m = re.match(r"alpha_top_(\d+)pct_rebalance_(\d+)d", s)
            if not m:
                continue
            top_pct = float(m.group(1)) / 100.0
            rebalance_days = int(m.group(2))
            selection_mode = "top_pct"
        eq, tr, _stats = mod.simulate_rebalance_with_costs(
            price_wide=price_wide,
            price_wide_mtm=price_wide_mtm,
            daily_scores=daily_scores,
            sim_dates=test_dates,
            initial_capital=initial_capital,
            rebalance_days=rebalance_days,
            selection_mode=selection_mode,
            sell_cost_rate=cost,
            top_pct=top_pct,
            execute_next_day=True,
        )
        if len(eq) and "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"]).dt.normalize()
        if len(tr) and "date" in tr.columns:
            tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
        equities[s] = eq
        trades[s] = tr
    return equities, trades
def _load_trades_for_strategies(strategies: list[str], cost: float, backtests_dir: Path) -> dict[str, pd.DataFrame]:
    trades: dict[str, pd.DataFrame] = {}
    label = _cost_label(cost)
    for s in strategies:
        path = backtests_dir / f"friction_{label}_{_safe_name(s)}_trades.parquet"
        if path.exists():
            tr = pd.read_parquet(path)
            if "date" in tr.columns:
                tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
            trades[s] = tr
        else:
            trades[s] = pd.DataFrame()
    return trades


def _build_table_46_test(
    equities: dict[str, pd.DataFrame],
    trades: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    reports_dir: Path,
) -> pd.DataFrame:
    """Build Table 4.6 on the test interval, so it is comparable with T4.9."""
    test_start = pd.Timestamp(cfg["splits"]["test_start"]).normalize()
    test_end = pd.Timestamp(cfg["splits"]["test_end"]).normalize()
    rows = []
    for s in _all_report_strategies():
        eq = equities.get(s, pd.DataFrame()).copy()
        if len(eq) and "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"]).dt.normalize()
            seg = eq[(eq["date"] >= test_start) & (eq["date"] <= test_end)].copy()
        else:
            seg = pd.DataFrame()
        if len(seg) >= 2 and "liquidated_equity_rub" in seg.columns:
            start_eq = float(seg["liquidated_equity_rub"].iloc[0])
            end_eq = float(seg["liquidated_equity_rub"].iloc[-1])
            roi = _roi_pct(start_eq, end_eq)
            sharpe = _sharpe(seg["liquidated_equity_rub"])
            sortino = _sortino(seg["liquidated_equity_rub"])
            mdd = _max_drawdown_pct(seg["liquidated_equity_rub"])
            avg_pos = float(pd.to_numeric(seg.get("n_positions", pd.Series(dtype=float)), errors="coerce").mean())
            n_days = int(len(seg))
        else:
            start_eq = np.nan
            roi = sharpe = sortino = mdd = avg_pos = np.nan
            n_days = 0
        tr = trades.get(s, pd.DataFrame()).copy()
        if len(tr) and "date" in tr.columns:
            tr_seg = tr[(tr["date"] >= test_start) & (tr["date"] <= test_end)].copy()
        else:
            tr_seg = pd.DataFrame()
        turnover = np.nan
        n_trades = int(len(tr_seg))
        if n_days > 0 and len(tr_seg) and "gross_value_rub" in tr_seg.columns and np.isfinite(start_eq) and start_eq > 0:
            gross_turnover = pd.to_numeric(tr_seg["gross_value_rub"], errors="coerce").fillna(0.0).sum()
            turnover = float((gross_turnover / start_eq) * ANNUALIZATION_DAYS / n_days)
        rows.append({
            "Стратегия": _strategy_name(s),
            "ROI, %": roi,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MDD, %": mdd,
            "Годовой оборот, × капитала": turnover,
            "Число сделок": n_trades,
            "Среднее число открытых позиций": avg_pos,
        })
    table = pd.DataFrame(rows)
    table.to_csv(reports_dir / "T4_6_метрики_стратегий_на_test_при_10pct_sell_cost.csv", index=False)
    _write_md_table(table, reports_dir / "T4_6_метрики_стратегий_на_test_при_10pct_sell_cost.md", digits=2)
    return table


def _build_table_46(grid: pd.DataFrame, equities: dict[str, pd.DataFrame], reports_dir: Path) -> pd.DataFrame:
    cost_rows = grid[np.isclose(grid["sell_cost_rate"], TARGET_COST)].copy()
    cost_rows = cost_rows[cost_rows["strategy"].isin(_all_report_strategies())].copy()
    cost_rows["order_key"] = cost_rows["strategy"].map(_strategy_order_key)
    cost_rows = cost_rows.sort_values("order_key")

    rows = []
    for _, r in cost_rows.iterrows():
        s = str(r["strategy"])
        eq = equities.get(s)
        sortino = _sortino(eq["liquidated_equity_rub"] if eq is not None and "liquidated_equity_rub" in eq.columns else pd.Series(dtype=float))
        rows.append({
            "Стратегия": _strategy_name(s),
            "ROI, %": r.get("roi_pct"),
            "Sharpe": r.get("sharpe"),
            "Sortino": sortino,
            "MDD, %": r.get("max_drawdown_pct"),
            "Годовой оборот, × капитала": _annualized_turnover(r),
            "Число сделок": r.get("n_trades"),
            "Среднее число открытых позиций": r.get("avg_n_positions"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(reports_dir / "T4_6_метрики_стратегий_при_10pct_sell_cost.csv", index=False)
    _write_md_table(table, reports_dir / "T4_6_метрики_стратегий_при_10pct_sell_cost.md", digits=2)
    return table


def _build_table_47(grid: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    g = grid[grid["strategy"].isin(_all_report_strategies())].copy()
    g["cost_col"] = g["sell_cost_rate"].map(_pct_label)
    pivot = g.pivot_table(index="strategy", columns="cost_col", values="roi_pct", aggfunc="first")
    cols = [_pct_label(c) for c in COST_GRID]
    pivot = pivot.reindex(columns=cols)
    pivot = pivot.reset_index()
    pivot["order_key"] = pivot["strategy"].map(_strategy_order_key)
    pivot = pivot.sort_values("order_key").drop(columns="order_key")
    pivot.insert(0, "Стратегия", pivot.pop("strategy").map(_strategy_name))
    pivot.to_csv(reports_dir / "T4_7_ROI_стратегий_по_уровням_sell_cost.csv", index=False)
    _write_md_table(pivot, reports_dir / "T4_7_ROI_стратегий_по_уровням_sell_cost.md", digits=2)
    return pivot


def _build_table_48(break_even: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    be = break_even[break_even["strategy"].isin(_all_report_strategies())].copy()
    be["order_key"] = be["strategy"].map(_strategy_order_key)
    be = be.sort_values("order_key")
    table = pd.DataFrame({
        "Стратегия": be["strategy"].map(_strategy_name),
        "Zero-profit break-even cost, %": pd.to_numeric(be.get("zero_profit_break_even_cost"), errors="coerce") * 100,
        "Passive-relative break-even cost, %": pd.to_numeric(be.get("passive_relative_break_even_cost"), errors="coerce") * 100,
    })
    table.loc[table["Стратегия"] == "Пассивное удержание", "Passive-relative break-even cost, %"] = np.nan
    table.to_csv(reports_dir / "T4_8_break_even_cost_по_стратегиям.csv", index=False)
    _write_md_table(table, reports_dir / "T4_8_break_even_cost_по_стратегиям.md", digits=2)
    return table


def _write_md_table(df: pd.DataFrame, path: Path, digits: int = 2) -> None:
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]) or pd.api.types.is_integer_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.{digits}f}")
        else:
            tmp[col] = tmp[col].fillna("—")
    path.write_text(tmp.to_markdown(index=False), encoding="utf-8")


def _select_key_strategies(grid: pd.DataFrame, n_alpha: int = 4) -> list[str]:
    at_cost = grid[np.isclose(grid["sell_cost_rate"], TARGET_COST)].copy()
    at_cost = at_cost[at_cost["strategy_type"] == "alpha_rotation"].copy()
    at_cost = at_cost.sort_values("roi_pct", ascending=False)
    selected = []
    # Always include robust candidate if present.
    preferred = ["alpha_top_10pct_rebalance_180d", "alpha_top_5pct_rebalance_90d"]
    for s in preferred:
        if s in at_cost["strategy"].values and s not in selected:
            selected.append(s)
    for s in at_cost["strategy"].tolist():
        if s not in selected:
            selected.append(s)
        if len(selected) >= n_alpha:
            break
    return selected[:n_alpha]


def _figure_44_equity_curves(
    grid: pd.DataFrame,
    equities: dict[str, pd.DataFrame],
    reports_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    selected = ["buy_and_hold_all"] + _select_key_strategies(grid, 4)
    plt.figure(figsize=(11.5, 6.5))
    for s in selected:
        eq = equities.get(s)
        if eq is None or len(eq) == 0:
            continue
        col = "liquidated_equity_rub" if "liquidated_equity_rub" in eq.columns else "equity_rub"
        y = pd.to_numeric(eq[col], errors="coerce")
        y = y / y.dropna().iloc[0]
        style = "--" if s == "buy_and_hold_all" else "-"
        lw = 2.8 if s == "buy_and_hold_all" else 2.2
        color = "black" if s == "buy_and_hold_all" else None
        plt.plot(eq["date"], y, linestyle=style, linewidth=lw, color=color, label=_strategy_name(s))
    plt.axhline(1.0, color="0.35", linewidth=0.9)
    plt.title("Рисунок 4.4 — Накопленная стоимость стратегий при издержках продажи 10%", fontsize=14, pad=12)
    plt.xlabel("Дата")
    plt.ylabel("Нормированная стоимость портфеля")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(reports_dir / "Рис_4_4_equity_curves_стратегий_при_10pct.png", dpi=190, bbox_inches="tight")
    plt.close()


def _figure_45_heatmap(grid: pd.DataFrame, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle

    g = grid[np.isclose(grid["sell_cost_rate"], TARGET_COST)].copy()
    passive_roi = float(g.loc[g["strategy"] == "buy_and_hold_all", "roi_pct"].iloc[0])
    alpha = g[g["strategy_type"] == "alpha_rotation"].copy()
    alpha["excess_roi_vs_buy_hold_pct"] = alpha["roi_pct"].astype(float) - passive_roi
    pivot = alpha.pivot_table(index="top_pct", columns="rebalance_days", values="excess_roi_vs_buy_hold_pct", aggfunc="first").sort_index()
    values = pivot.values.astype(float)
    finite = values[np.isfinite(values)]
    vmax = max(float(np.nanmax(np.abs(finite))) if len(finite) else 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)} дн." for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Top {int(round(float(i) * 100))}%" for i in pivot.index], fontsize=11)
    ax.set_xlabel("Периодичность ребалансировки")
    ax.set_ylabel("Доля верхних предметов по alpha_pred")
    ax.set_title("Рисунок 4.5 — Избыточный ROI стратегий относительно пассивного удержания при c = 10%", fontsize=13, pad=12)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Избыточный ROI, п.п.")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            color = "black" if abs(val) < 0.55 * vmax else "white"
            ax.text(j, i, f"{val:.1f}" if val >= 0 else f"−{abs(val):.1f}", ha="center", va="center", color=color, fontsize=10)
            if val > 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.8))
    fig.text(0.01, 0.012, f"База сравнения: ROI buy-and-hold при c = 10% составляет {passive_roi:.2f}%.", fontsize=9, color="0.25")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(reports_dir / "Рис_4_5_heatmap_excess_ROI_vs_passive_10pct.png", dpi=190, bbox_inches="tight")
    plt.close()


def _interp_y(x: float, xs: list[float], ys: list[float]) -> float | None:
    pairs = sorted((float(a), float(b)) for a, b in zip(xs, ys) if pd.notna(a) and pd.notna(b))
    if not pairs or x < pairs[0][0] or x > pairs[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pairs[-1][1]


def _figure_46_cost_sensitivity(grid: pd.DataFrame, break_even: pd.DataFrame, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = _select_key_strategies(grid, 4)
    passive = grid[grid["strategy"] == "buy_and_hold_all"].sort_values("sell_cost_rate")
    plt.figure(figsize=(11.5, 6.5))
    plt.plot(passive["sell_cost_rate"] * 100, passive["roi_pct"], color="black", linestyle="--", linewidth=3.0, marker="o", label="Пассивное удержание")
    for s in selected:
        g = grid[grid["strategy"] == s].sort_values("sell_cost_rate")
        plt.plot(g["sell_cost_rate"] * 100, g["roi_pct"], marker="o", linewidth=2.2, label=_strategy_name(s))
        be_row = break_even[break_even["strategy"] == s]
        if len(be_row):
            c = pd.to_numeric(be_row["passive_relative_break_even_cost"], errors="coerce").iloc[0]
            if pd.notna(c) and 0 <= float(c) <= 0.20:
                y = _interp_y(float(c), g["sell_cost_rate"].tolist(), g["roi_pct"].tolist())
                if y is not None:
                    plt.scatter([float(c) * 100], [y], s=70, marker="x", color="black", zorder=5)
                    plt.text(float(c) * 100 + 0.25, y, f"{float(c) * 100:.1f}%", fontsize=8, va="center")
    plt.axhline(0, color="0.35", linewidth=0.9)
    plt.title("Рисунок 4.6 — Чувствительность ROI к уровню издержек продажи", fontsize=14, pad=12)
    plt.xlabel("Издержки продажи, %")
    plt.ylabel("Ликвидационный ROI, %")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(reports_dir / "Рис_4_6_cost_sensitivity_curves.png", dpi=190, bbox_inches="tight")
    plt.close()


def _figure_47_turnover_scatter(grid: pd.DataFrame, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt
    g = grid[np.isclose(grid["sell_cost_rate"], TARGET_COST)].copy()
    g = g[g["strategy"].isin(_all_report_strategies())].copy()
    g["annual_turnover"] = g.apply(_annualized_turnover, axis=1)
    alpha = g[g["strategy_type"] == "alpha_rotation"].copy()
    marker_map = {30: "o", 90: "s", 180: "^", 365: "D"}
    color_map = {0.05: "tab:blue", 0.10: "tab:orange", 0.20: "tab:green", 0.30: "tab:red"}
    plt.figure(figsize=(11.5, 6.5))
    for days, gd in alpha.groupby("rebalance_days"):
        for pct, gp in gd.groupby("top_pct"):
            plt.scatter(
                gp["annual_turnover"], gp["roi_pct"],
                marker=marker_map.get(int(days), "o"),
                color=color_map.get(float(pct), None),
                s=95,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.88,
                label=f"Top {int(round(float(pct) * 100))}%, {int(days)} дн.",
            )
    passive = g[g["strategy"] == "buy_and_hold_all"]
    if len(passive):
        plt.scatter(passive["annual_turnover"], passive["roi_pct"], marker="*", s=180, color="black", label="Пассивное удержание", zorder=5)
    for _, r in alpha.iterrows():
        plt.text(r["annual_turnover"], r["roi_pct"], f" {int(round(float(r['top_pct']) * 100))}%/{int(r['rebalance_days'])}", fontsize=7.5, va="center")
    plt.axhline(float(passive["roi_pct"].iloc[0]) if len(passive) else 0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    plt.title("Рисунок 4.7 — Годовой оборот и ROI стратегий при c = 10%", fontsize=14, pad=12)
    plt.xlabel("Годовой оборот, × начального капитала")
    plt.ylabel("Ликвидационный ROI, %")
    plt.grid(True, alpha=0.25)
    # Compact custom legends.
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[p], markeredgecolor="black", markersize=8, label=f"Top {int(p*100)}%") for p in sorted(color_map)]
    marker_handles = [Line2D([0], [0], marker=marker_map[d], color="black", linestyle="None", markersize=8, label=f"{d} дн.") for d in sorted(marker_map)]
    leg1 = plt.legend(handles=color_handles, title="Ширина отбора", loc="upper right", frameon=True, fontsize=8)
    plt.gca().add_artist(leg1)
    plt.legend(handles=marker_handles, title="Ребалансировка", loc="lower right", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / "Рис_4_7_turnover_ROI_scatter_10pct.png", dpi=190, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--friction-script", default=None, help="Path to 19_alpha_rotation_friction_sweep.py")
    parser.add_argument("--force-resim", action="store_true", help="Recompute c=10% equity curves even if parquet files exist.")
    parser.add_argument("--table46-scope", choices=["test", "full"], default="test", help="Use test interval metrics for T4.6 by default; full reproduces the old table.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg, "reports_dir")
    backtests_dir = resolve_path(cfg, "backtests_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)
    friction_script = Path(args.friction_script) if args.friction_script else SCRIPT_DIR / "19_alpha_rotation_friction_sweep.py"
    if not friction_script.exists():
        raise FileNotFoundError(f"Не найден скрипт friction sweep: {friction_script}")

    grid_raw, break_even_raw = _load_reports(reports_dir)
    grid = _select_grid(grid_raw)
    strategies = _all_report_strategies()

    # For T4.6(test), use a fresh out-of-sample simulation that starts at test_start
    # with cash and no inherited pre-test positions. This makes it comparable with
    # T4.9 decomposition, which is also started fresh on the test interval.
    if args.table46_scope == "test":
        equities_test, trades_test = _simulate_test_fresh_equities(cfg, strategies, TARGET_COST, friction_script)
        t46 = _build_table_46_test(equities_test, trades_test, cfg, reports_dir)
    else:
        equities_test = None
        trades_test = None

    # Load full-period equities for the remaining figures/tables that intentionally
    # use full-sweep outputs.
    equities = _load_or_simulate_equities(cfg, strategies, TARGET_COST, backtests_dir, friction_script, force_resim=args.force_resim)
    trades = _load_trades_for_strategies(strategies, TARGET_COST, backtests_dir)

    if args.table46_scope == "full":
        t46 = _build_table_46(grid, equities, reports_dir)
    t47 = _build_table_47(grid, reports_dir)
    t48 = _build_table_48(break_even_raw, reports_dir)

    _figure_44_equity_curves(grid, equities, reports_dir)
    _figure_45_heatmap(grid, reports_dir)
    _figure_46_cost_sensitivity(grid, break_even_raw, reports_dir)
    _figure_47_turnover_scatter(grid, reports_dir)

    print("Сохранены таблицы:")
    print(" ", reports_dir / ("T4_6_метрики_стратегий_на_test_при_10pct_sell_cost.csv" if args.table46_scope == "test" else "T4_6_метрики_стратегий_при_10pct_sell_cost.csv"))
    print(" ", reports_dir / "T4_7_ROI_стратегий_по_уровням_sell_cost.csv")
    print(" ", reports_dir / "T4_8_break_even_cost_по_стратегиям.csv")
    print("Сохранены рисунки:")
    for name in [
        "Рис_4_4_equity_curves_стратегий_при_10pct.png",
        "Рис_4_5_heatmap_excess_ROI_vs_passive_10pct.png",
        "Рис_4_6_cost_sensitivity_curves.png",
        "Рис_4_7_turnover_ROI_scatter_10pct.png",
    ]:
        print(" ", reports_dir / name)


if __name__ == "__main__":
    main()
