#!/usr/bin/env python
"""
Russian academic tables and figures for Chapter 4 of the CS2 trading project.

This script is intentionally separated from the trading/backtest scripts. It reads
already produced artifacts and creates Russian-language tables/figures for the
thesis chapter: sample split, model/universe summary, universe size, train/val/test
calendar, signal diagnostics, friction diagnostics, and metadata interpretation.

Run from repo root:
    python scripts/20_chapter4_russian_tables_figures.py --config configs/cs2_trading_rebuilt.yaml

Optional:
    python scripts/20_chapter4_russian_tables_figures.py --config configs/cs2_trading_rebuilt.yaml --cost-for-matrix 0.10
"""
from __future__ import annotations

import argparse
import json
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

try:
    from cs2_trading_rebuild.models import REGIME_FEATURES, MODEL_NUMERIC_FEATURES, CATEGORICAL_FEATURES
except Exception:  # script is still useful if imports are unavailable
    REGIME_FEATURES = []
    MODEL_NUMERIC_FEATURES = []
    CATEGORICAL_FEATURES = ["quality", "stattrak", "souvenir", "type", "rarity", "metatype"]

META_COLS = ["quality", "stattrak", "souvenir", "type", "rarity", "metatype"]
OPTIONAL_ALPHA_DIAGNOSTIC_COLS = [
    "future_excess_gross_return_30d",
    "future_gross_return_30d",
    "future_net_return_30d",
    "alpha_rank_score_raw",
    "alpha_prob_top",
    "alpha_prob_bottom",
]

SPLIT_RU = {
    "train": "Обучающая выборка",
    "validation": "Валидационная выборка",
    "val": "Валидационная выборка",
    "test": "Тестовая выборка",
}

DIM_RU = {
    "quality": "Качество",
    "rarity": "Редкость",
    "type": "Тип предмета",
    "metatype": "Мета-тип",
    "stattrak": "StatTrak",
    "souvenir": "Сувенирный предмет",
}

CATEGORY_RU = {
    "Factory New": "Прямо с завода",
    "Minimal Wear": "Немного поношенное",
    "Field-Tested": "После полевых испытаний",
    "Well-Worn": "Поношенное",
    "Battle-Scarred": "Закалённое в боях",
    "Consumer Grade": "Потребительское качество",
    "Industrial Grade": "Промышленное качество",
    "Mil-Spec Grade": "Военное качество",
    "Restricted": "Запрещённое",
    "Classified": "Засекреченное",
    "Covert": "Тайное",
    "Contraband": "Контрабандное",
    "Extraordinary": "Экстраординарное",
    "High Grade": "Высокое качество",
    "Remarkable": "Примечательное",
    "Exotic": "Экзотическое",
    "Master": "Мастерское",
    "Base Grade": "Базовое качество",
    "Container": "Контейнер",
    "Weapon": "Оружие",
    "Knife": "Нож",
    "Gloves": "Перчатки",
    "Sticker": "Наклейка",
    "Graffiti": "Граффити",
    "Agent": "Агент",
    "Music Kit": "Музыкальный набор",
    "Patch": "Нашивка",
    "Case": "Кейс",
    "Collectible": "Коллекционный предмет",
    "Normal": "Обычный",
    "StatTrak": "StatTrak",
    "Souvenir": "Сувенирный",
    "Unknown": "Не определено",
    "__MISSING__": "Не определено",
    "True": "Да",
    "False": "Нет",
    "true": "Да",
    "false": "Нет",
    "1": "Да",
    "0": "Нет",
}


def _ru_cat(x: Any, max_len: int = 44) -> str:
    if pd.isna(x):
        s = "Не определено"
    else:
        s = str(x)
    s = CATEGORY_RU.get(s, s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _available_parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return set(pd.read_parquet(path).head(0).columns)
        except Exception:
            return set()


def _read_parquet_existing(path: Path, wanted: list[str], **kwargs) -> pd.DataFrame:
    available = _available_parquet_columns(path)
    cols = [c for c in wanted if c in available]
    if not cols:
        raise KeyError(f"None of requested columns exist in {path}: {wanted}")
    return pd.read_parquet(path, columns=cols, **kwargs)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fmt_date(x: Any) -> str:
    return pd.Timestamp(x).strftime("%d.%m.%Y")


def _fmt_num(x: Any, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if nd == 0:
        return f"{float(x):,.0f}".replace(",", " ")
    return f"{float(x):,.{nd}f}".replace(",", " ").replace(".", ",")


def _annualized_return_pct(roi_pct: Any, n_days: Any, annualization_days: int = 365) -> float | None:
    try:
        roi = float(roi_pct)
        days = float(n_days)
    except Exception:
        return None
    if not np.isfinite(roi) or not np.isfinite(days) or days <= 0:
        return None
    wealth = 1.0 + roi / 100.0
    if wealth <= 0:
        return None
    return float((wealth ** (annualization_days / days) - 1.0) * 100.0)


def _split_ranges(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    s = cfg["splits"]
    return [
        {"split": "train", "name_ru": "Обучающая выборка", "start": pd.Timestamp(s["train_start"]).normalize(), "end": pd.Timestamp(s["train_end"]).normalize()},
        {"split": "validation", "name_ru": "Валидационная выборка", "start": pd.Timestamp(s["val_start"]).normalize(), "end": pd.Timestamp(s["val_end"]).normalize()},
        {"split": "test", "name_ru": "Тестовая выборка", "start": pd.Timestamp(s["test_start"]).normalize(), "end": pd.Timestamp(s["test_end"]).normalize()},
    ]


def _load_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(cfg, "universe_daily")
    wanted = ["date", "market_hash_name", "tradeable_today", "tradability_score"] + META_COLS
    df = _read_parquet_existing(path, wanted)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["tradeable_today"] = df["tradeable_today"].fillna(False).astype(bool)
    df = df.drop_duplicates(subset=["date", "market_hash_name"], keep="last")
    for c in META_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown")
    return df


def _load_alpha(cfg: dict[str, Any]) -> pd.DataFrame | None:
    path = resolve_path(cfg, "models_dir") / "alpha_predictions.parquet"
    if not path.exists():
        return None
    wanted = ["date", "market_hash_name", "alpha_pred"] + META_COLS + OPTIONAL_ALPHA_DIAGNOSTIC_COLS
    df = _read_parquet_existing(path, wanted)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["alpha_pred"] = pd.to_numeric(df["alpha_pred"], errors="coerce")
    for c in OPTIONAL_ALPHA_DIAGNOSTIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in META_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Unknown")
    return df.dropna(subset=["alpha_pred"])


def _load_cost_grid(cfg: dict[str, Any]) -> pd.DataFrame | None:
    path = resolve_path(cfg, "reports_dir") / "19_strategy_cost_grid.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # The cost-grid CSV may already contain derived best-passive/excess columns
    # from a previous version of script 19. Recomputing them after a merge can
    # create pandas suffixes (_x/_y), which then caused KeyError for
    # best_passive_annualized_roi_pct. We therefore remove all derived columns
    # and rebuild them deterministically from the raw strategy rows.
    derived_cols = [
        "best_passive_strategy",
        "best_passive_annualized_roi_pct",
        "best_passive_roi_pct",
        "annualized_excess_roi_vs_best_passive_pct",
        "excess_roi_vs_best_passive_pct",
    ]
    suffix_derived = [
        c for c in df.columns
        if c.endswith("_x") or c.endswith("_y")
        if any(c.startswith(base) for base in derived_cols)
    ]
    df = df.drop(columns=[c for c in derived_cols + suffix_derived if c in df.columns], errors="ignore")

    if "annualized_roi_pct" not in df.columns:
        df["annualized_roi_pct"] = [
            _annualized_return_pct(r, n)
            for r, n in zip(df.get("roi_pct"), df.get("n_days"))
        ]

    required = {"strategy_type", "sell_cost_rate", "strategy", "annualized_roi_pct", "roi_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"Предупреждение: не хватает колонок в 19_strategy_cost_grid.csv: {missing}")
        return df

    passive = df[df["strategy_type"].isin(["passive", "passive_rebalanced"])].copy()
    if not len(passive):
        print("Предупреждение: в cost-grid не найдены пассивные стратегии; сравнительные excess-поля не рассчитаны.")
        return df

    best_passive = (
        passive.sort_values(["sell_cost_rate", "annualized_roi_pct"], ascending=[True, False])
        .groupby("sell_cost_rate", as_index=False)
        .first()[["sell_cost_rate", "strategy", "annualized_roi_pct", "roi_pct"]]
        .rename(columns={
            "strategy": "best_passive_strategy",
            "annualized_roi_pct": "best_passive_annualized_roi_pct",
            "roi_pct": "best_passive_roi_pct",
        })
    )

    df = df.merge(best_passive, on="sell_cost_rate", how="left", validate="many_to_one")
    df["annualized_excess_roi_vs_best_passive_pct"] = (
        df["annualized_roi_pct"] - df["best_passive_annualized_roi_pct"]
    )
    df["excess_roi_vs_best_passive_pct"] = df["roi_pct"] - df["best_passive_roi_pct"]
    return df


def _setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _savefig(plt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def table_4_1(cfg: dict[str, Any], universe: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    rows = []
    for sp in _split_ranges(cfg):
        mask = (universe["date"] >= sp["start"]) & (universe["date"] <= sp["end"]) & universe["tradeable_today"]
        part = universe.loc[mask]
        rows.append({
            "Часть выборки": sp["name_ru"],
            "Дата начала": _fmt_date(sp["start"]),
            "Дата окончания": _fmt_date(sp["end"]),
            "Календарных дней": int((sp["end"] - sp["start"]).days + 1),
            "Item-days": int(len(part)),
            "Уникальных предметов": int(part["market_hash_name"].nunique()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "T4_1_хронологическое_разбиение_выборки.csv", index=False, encoding="utf-8-sig")
    out.to_markdown(reports_dir / "T4_1_хронологическое_разбиение_выборки.md", index=False)
    return out


def table_4_2(cfg: dict[str, Any], universe: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    modeling = cfg.get("modeling", {})
    research = cfg.get("research", {})
    regime_cfg = modeling.get("regime_model", {})
    alpha_cfg = modeling.get("alpha_model", {})
    regime_report = _read_json(resolve_path(cfg, "models_dir") / "regime_report.json")
    alpha_report = _read_json(resolve_path(cfg, "models_dir") / "alpha_report.json")

    def add(block: str, param: str, val: Any, note: str = "") -> None:
        rows.append({"Смысловой блок": block, "Показатель": param, "Значение": val, "Комментарий": note})

    add("Гиперпараметры режимной модели", "Тип алгоритма", "HistGradientBoostingClassifier", "градиентный бустинг по гистограммным разбиениям")
    for k in ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "l2_regularization", "max_bins"]:
        if k in regime_cfg:
            add("Гиперпараметры режимной модели", k, regime_cfg[k])
    add("Гиперпараметры режимной модели", "Целевая переменная", regime_report.get("target_col", research.get("regime_target_col", "—")))
    add("Гиперпараметры режимной модели", "Число признаков", len(REGIME_FEATURES) if REGIME_FEATURES else "—")

    add("Гиперпараметры альфа-модели", "Тип алгоритма", "двухголовая классификационная модель ранжирования", "отдельные вероятности верхнего и нижнего квантилей")
    for k in ["cls_n_estimators", "n_estimators", "cls_max_depth", "max_depth", "cls_min_samples_leaf", "min_samples_leaf", "cls_learning_rate", "learning_rate", "cls_l2_regularization", "l2_regularization", "max_bins", "rank_quantile_low", "rank_quantile_high"]:
        if k in alpha_cfg:
            add("Гиперпараметры альфа-модели", k, alpha_cfg[k])
    add("Гиперпараметры альфа-модели", "Целевая переменная", alpha_report.get("target_col", research.get("primary_target", "—")))
    add("Гиперпараметры альфа-модели", "Горизонт целевой доходности", f"{research.get('primary_alpha_horizon_days', '—')} дней")
    add("Гиперпараметры альфа-модели", "Число числовых признаков", len(MODEL_NUMERIC_FEATURES) if MODEL_NUMERIC_FEATURES else "—")
    add("Гиперпараметры альфа-модели", "Категориальные признаки", ", ".join(CATEGORICAL_FEATURES) if CATEGORICAL_FEATURES else "—")

    sp_test = _split_ranges(cfg)[2]
    test = universe[(universe["date"] >= sp_test["start"]) & (universe["date"] <= sp_test["end"]) & universe["tradeable_today"]]
    daily = test.groupby("date")["market_hash_name"].nunique()
    add("Динамическая торговая вселенная на test-интервале", "Медианный размер", int(daily.median()) if len(daily) else "—", "предметов в день")
    add("Динамическая торговая вселенная на test-интервале", "Минимальный размер", int(daily.min()) if len(daily) else "—", "предметов в день")
    add("Динамическая торговая вселенная на test-интервале", "Максимальный размер", int(daily.max()) if len(daily) else "—", "предметов в день")
    add("Динамическая торговая вселенная на test-интервале", "Уникальных предметов", int(test["market_hash_name"].nunique()) if len(test) else "—")

    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "T4_2_гиперпараметры_и_торговая_вселенная.csv", index=False, encoding="utf-8-sig")
    out.to_markdown(reports_dir / "T4_2_гиперпараметры_и_торговая_вселенная.md", index=False)
    return out


def fig_4_1(cfg: dict[str, Any], universe: pd.DataFrame, reports_dir: Path) -> Path:
    plt = _setup_matplotlib()
    daily = universe[universe["tradeable_today"]].groupby("date")["market_hash_name"].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.plot(daily.index, daily.values, linewidth=2.2)
    colors = {"train": "#E8F2FF", "validation": "#FFF3D6", "test": "#EAF7EA"}
    for sp in _split_ranges(cfg):
        ax.axvspan(sp["start"], sp["end"], color=colors[sp["split"]], alpha=0.55, linewidth=0)
        ax.axvline(sp["start"], color="0.25", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(sp["start"], ax.get_ylim()[1], "  " + sp["name_ru"].split()[0], va="top", fontsize=9, color="0.25")
    ax.set_title("Рисунок 4.1 — Размер динамической торговой вселенной во времени")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Число предметов, доступных к торговле")
    ax.grid(True, alpha=0.25)
    out = reports_dir / "Рис_4_1_размер_динамической_торговой_вселенной.png"
    _savefig(plt, out)
    return out


def fig_4_2(cfg: dict[str, Any], reports_dir: Path) -> Path:
    plt = _setup_matplotlib()
    import matplotlib.dates as mdates
    ranges = _split_ranges(cfg)
    horizon = int(cfg.get("research", {}).get("primary_alpha_horizon_days", 30))
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    colors = {"train": "#5B8FF9", "validation": "#F6BD16", "test": "#5AD8A6"}
    y = 0
    for sp in ranges:
        start_num = mdates.date2num(sp["start"])
        width = mdates.date2num(sp["end"] + pd.Timedelta(days=1)) - start_num
        ax.broken_barh([(start_num, width)], (y - 0.18, 0.36), facecolors=colors[sp["split"]], edgecolors="0.25", linewidth=0.7)
        ax.text(sp["start"] + (sp["end"] - sp["start"]) / 2, y, sp["name_ru"].replace(" выборка", ""), ha="center", va="center", fontsize=10, color="black")
    # Future-return horizon annotation near the validation/test boundary.
    anchor = ranges[2]["start"]
    ax.annotate(
        f"Горизонт будущей доходности: {horizon} дней",
        xy=(anchor, y + 0.42),
        xytext=(anchor + pd.Timedelta(days=horizon), y + 0.42),
        arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": "0.25"},
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.25",
    )
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_title("Рисунок 4.2 — Календарная схема обучения и оценки")
    ax.set_xlabel("Дата")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.grid(True, axis="x", alpha=0.25)
    out = reports_dir / "Рис_4_2_календарная_схема_обучения_и_оценки.png"
    _savefig(plt, out)
    return out


def _assign_daily_deciles(df: pd.DataFrame) -> pd.Series:
    def one_day(s: pd.Series) -> pd.Series:
        r = pd.to_numeric(s, errors="coerce").rank(method="first")
        try:
            return pd.qcut(r, 10, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)
    return df.groupby("date", group_keys=False)["alpha_pred"].transform(one_day)


def _split_label(dates: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    d = pd.to_datetime(dates).dt.normalize()
    out = pd.Series("other", index=dates.index, dtype="object")
    for sp in _split_ranges(cfg):
        out[(d >= sp["start"]) & (d <= sp["end"])] = sp["split"]
    return out


def _target_col(alpha: pd.DataFrame) -> str | None:
    for c in ["future_excess_gross_return_30d", "future_gross_return_30d", "future_net_return_30d"]:
        if c in alpha.columns and pd.to_numeric(alpha[c], errors="coerce").notna().any():
            return c
    return None


def fig_alpha_deciles_ru(cfg: dict[str, Any], alpha: pd.DataFrame | None, reports_dir: Path) -> Path | None:
    if alpha is None:
        return None
    col = _target_col(alpha)
    if col is None:
        return None
    plt = _setup_matplotlib()
    df = alpha.dropna(subset=["alpha_pred", col]).copy()
    df["alpha_decile"] = _assign_daily_deciles(df)
    df = df.dropna(subset=["alpha_decile"])
    df["alpha_decile"] = df["alpha_decile"].astype(int) + 1
    df[col] = pd.to_numeric(df[col], errors="coerce") * 100.0
    df["split"] = _split_label(df["date"], cfg)
    agg = df[df["split"].isin(["train", "validation", "test"])].groupby(["split", "alpha_decile"], as_index=False)[col].mean()
    order = ["train", "validation", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, sp in zip(axes, order):
        g = agg[agg["split"] == sp]
        ax.bar(g["alpha_decile"], g[col], width=0.78)
        ax.axhline(0, color="0.15", linewidth=0.9)
        ax.set_title(SPLIT_RU[sp].replace(" выборка", ""))
        ax.set_xlabel("Дециль альфа-сигнала (10 — наивысший)")
        ax.grid(True, axis="y", alpha=0.25)
        if len(g):
            top = float(g.loc[g["alpha_decile"] == 10, col].mean())
            bot = float(g.loc[g["alpha_decile"] == 1, col].mean())
            ax.text(0.02, 0.96, f"Спред 10–1: {top - bot:+.1f} п.п.", transform=ax.transAxes, va="top", fontsize=9, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.6", "alpha": 0.9})
    axes[0].set_ylabel("Средняя реализованная будущая доходность, %")
    fig.suptitle("Доходность портфелей, отсортированных по альфа-сигналу", y=1.03)
    out = reports_dir / "Рис_4_3_доходность_децилей_альфа_сигнала.png"
    _savefig(plt, out)
    return out


def fig_rolling_spread_ru(cfg: dict[str, Any], alpha: pd.DataFrame | None, reports_dir: Path) -> Path | None:
    if alpha is None:
        return None
    col = _target_col(alpha)
    if col is None:
        return None
    plt = _setup_matplotlib()
    df = alpha.dropna(subset=["alpha_pred", col]).copy()
    df["alpha_decile"] = _assign_daily_deciles(df)
    df = df.dropna(subset=["alpha_decile"])
    df["alpha_decile"] = df["alpha_decile"].astype(int) + 1
    df[col] = pd.to_numeric(df[col], errors="coerce") * 100.0
    rows = []
    for d, g in df.groupby("date", sort=True):
        top = g.loc[g["alpha_decile"] == 10, col].mean()
        bot = g.loc[g["alpha_decile"] == 1, col].mean()
        if pd.notna(top) and pd.notna(bot):
            rows.append({"date": pd.Timestamp(d), "spread": float(top - bot)})
    sp = pd.DataFrame(rows).sort_values("date")
    if len(sp) < 20:
        return None
    sp["rolling"] = sp["spread"].rolling(90, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(sp["date"], sp["rolling"], linewidth=2.2)
    ax.axhline(0, color="0.15", linewidth=0.9)
    for sr in _split_ranges(cfg)[1:]:
        ax.axvline(sr["start"], color="0.35", linestyle="--", linewidth=1.1)
        ax.text(sr["start"], ax.get_ylim()[1], "  " + sr["name_ru"].replace(" выборка", ""), va="top", fontsize=9)
    ax.set_title("Скользящая сила альфа-сигнала: верхний дециль минус нижний дециль")
    ax.set_xlabel("Дата сигнала")
    ax.set_ylabel("90-дневное среднее спреда, п.п.")
    out = reports_dir / "Рис_4_4_скользящий_спред_альфа_сигнала.png"
    _savefig(plt, out)
    return out


def fig_top10_cost_ru(grid: pd.DataFrame | None, reports_dir: Path) -> Path | None:
    if grid is None or not len(grid):
        return None
    plt = _setup_matplotlib()
    g = grid[(grid["strategy_type"] == "alpha_rotation") & (grid["top_pct"].round(2) == 0.10)].copy()
    if not len(g):
        return None
    passive = grid[grid["strategy_type"].isin(["passive", "passive_rebalanced"])].sort_values(["sell_cost_rate", "annualized_roi_pct"], ascending=[True, False]).groupby("sell_cost_rate", as_index=False).first()
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for days, gg in g.groupby("rebalance_days"):
        gg = gg.sort_values("sell_cost_rate")
        ax.plot(gg["sell_cost_rate"] * 100, gg["annualized_roi_pct"], marker="o", linewidth=2.2, label=f"Альфа top 10%, ребалансировка {int(days)} дн.")
    ax.plot(passive["sell_cost_rate"] * 100, passive["annualized_roi_pct"], marker="o", linestyle="--", linewidth=2.4, color="black", label="Лучший пассивный ориентир")
    ax.axhline(0, color="0.15", linewidth=0.9)
    ax.set_xlabel("Издержки продажи, %")
    ax.set_ylabel("Годовая ликвидационная доходность, %")
    ax.set_title("Чувствительность доходности к издержкам: корзина top 10% по альфа-сигналу")
    ax.legend(frameon=True)
    out = reports_dir / "Рис_4_5_чувствительность_top10_к_издержкам.png"
    _savefig(plt, out)
    return out


def fig_excess_heatmap_ru(grid: pd.DataFrame | None, reports_dir: Path, cost: float = 0.10) -> Path | None:
    if grid is None or not len(grid):
        return None
    plt = _setup_matplotlib()
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle
    costs = sorted(grid["sell_cost_rate"].dropna().unique())
    chosen = min(costs, key=lambda c: abs(float(c) - cost))
    alpha = grid[(grid["strategy_type"] == "alpha_rotation") & (grid["sell_cost_rate"] == chosen)].copy()
    bh = grid[(grid["strategy"] == "buy_and_hold_all") & (grid["sell_cost_rate"] == chosen)].copy()
    if not len(alpha) or not len(bh):
        return None
    bh_ann = float(bh["annualized_roi_pct"].iloc[0])
    alpha["excess"] = alpha["annualized_roi_pct"].astype(float) - bh_ann
    pivot = alpha.pivot_table(index="top_pct", columns="rebalance_days", values="excess", aggfunc="first").sort_index()
    vals = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)] if vals.size else np.array([])
    vmax = max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)} дн." for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Top {int(round(float(i) * 100))}%" for i in pivot.index])
    ax.set_xlabel("Интервал ребалансировки")
    ax.set_ylabel("Ширина альфа-корзины")
    ax.set_title(f"Избыточная годовая доходность относительно buy-and-hold при издержках {chosen * 100:.0f}%")
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Избыточная годовая доходность, п.п.")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = vals[i, j]
            if not np.isfinite(val):
                continue
            txt_color = "black" if abs(val) < 0.55 * vmax else "white"
            label = f"{val:.1f} п.п." if val >= 0 else f"−{abs(val):.1f} п.п."
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color=txt_color, bbox={"boxstyle": "round,pad=0.20", "fc": (1, 1, 1, 0.14), "ec": (1, 1, 1, 0)})
            if val > 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.9))
    fig.text(0.012, 0.012, f"База сравнения: buy-and-hold, годовая доходность {bh_ann:.1f}%", fontsize=9, color="0.25")
    out = reports_dir / "Рис_4_6_матрица_избыточной_доходности_при_10pct_издержках.png"
    _savefig(plt, out)
    return out


def _metadata_panel(universe: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    uni_cols = ["date", "market_hash_name", "tradeable_today"] + [c for c in META_COLS if c in universe.columns]
    a_cols = ["date", "market_hash_name", "alpha_pred"] + [c for c in META_COLS if c in alpha.columns]
    p = universe.loc[universe["tradeable_today"], uni_cols].merge(alpha[a_cols], on=["date", "market_hash_name"], how="inner", suffixes=("_universe", "_alpha"))
    for c in META_COLS:
        ca, cu = f"{c}_alpha", f"{c}_universe"
        if ca in p.columns and cu in p.columns:
            p[c] = p[ca].where(p[ca].notna(), p[cu])
        elif ca in p.columns:
            p[c] = p[ca]
        elif cu in p.columns:
            p[c] = p[cu]
        if c in p.columns:
            p[c] = p[c].astype("string").fillna("Unknown")
    p["rank_pct"] = p.groupby("date")["alpha_pred"].rank(method="average", pct=True)
    p["alpha_decile"] = _assign_daily_deciles(p).astype("float")
    p = p.dropna(subset=["alpha_decile"])
    p["alpha_decile"] = p["alpha_decile"].astype(int) + 1
    p["is_top10"] = p["rank_pct"] >= 0.90
    return p


def fig_metadata_preference_ru(universe: pd.DataFrame, alpha: pd.DataFrame | None, reports_dir: Path) -> Path | None:
    if alpha is None:
        return None
    p = _metadata_panel(universe, alpha)
    if not len(p):
        return None
    plt = _setup_matplotlib()
    from matplotlib import colors as mcolors
    cats = [c for c in ["quality", "rarity", "type", "metatype"] if c in p.columns]
    rows = []
    for col in cats:
        base = p[col].value_counts(normalize=True)
        top10 = p.loc[p["is_top10"], col].value_counts(normalize=True)
        cand = pd.concat([base.rename("base"), top10.rename("top10")], axis=1).fillna(0)
        cand["abs_enrichment"] = (cand["top10"] - cand["base"]).abs()
        keep = cand[(cand["base"] >= 0.01) | (cand["top10"] >= 0.01)].sort_values(["abs_enrichment", "top10"], ascending=False).head(6).index.tolist()
        for cat in keep:
            b = float(base.get(cat, 0.0))
            for dec, gd in p.groupby("alpha_decile"):
                share = float((gd[col] == cat).mean()) if len(gd) else 0
                rows.append({"dimension": col, "category": cat, "label": f"{DIM_RU.get(col, col)}: {_ru_cat(cat, 26)}", "decile": int(dec), "share_pct": share * 100, "enrichment_pp": (share - b) * 100})
    df = pd.DataFrame(rows)
    if not len(df):
        return None
    top_enrich = df[df["decile"] == 10].set_index("label")["enrichment_pp"].to_dict()
    dim_order = {DIM_RU.get(c, c): i for i, c in enumerate(cats)}
    labels = sorted(df["label"].unique(), key=lambda x: (dim_order.get(x.split(":", 1)[0], 99), -top_enrich.get(x, 0)))
    y_map = {lab: i for i, lab in enumerate(labels)}
    df["y"] = df["label"].map(y_map)
    finite = df["enrichment_pp"].replace([np.inf, -np.inf], np.nan).dropna()
    vmax = max(float(finite.abs().quantile(0.98)) if len(finite) else 1.0, 1.0)
    fig, ax = plt.subplots(figsize=(12, max(7, 0.34 * len(labels) + 2.3)))
    sizes = 18 + df["share_pct"].clip(0, 60) * 9
    sc = ax.scatter(df["decile"], df["y"], s=sizes, c=df["enrichment_pp"], cmap="RdYlGn", norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), edgecolor="0.25", linewidth=0.25, alpha=0.9)
    ax.axvline(9.5, color="0.2", linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Дециль альфа-сигнала (10 — наивысший)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Карта метаданных: представленность характеристик по децилям альфа-сигнала")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Отклонение от доли во всей торговой вселенной, п.п.")
    for share in [5, 15, 30]:
        ax.scatter([], [], s=18 + share * 9, c="white", edgecolor="0.35", label=f"{share}%")
    ax.legend(title="Размер маркера — доля", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.text(0.012, 0.012, "Цвет показывает перепредставленность/недопредставленность категории относительно всей торговой вселенной.", fontsize=9, color="0.30")
    out = reports_dir / "Рис_4_7_карта_метаданных_по_децилям_альфа_сигнала.png"
    _savefig(plt, out)
    return out


def fig_archetype_lift_ru(universe: pd.DataFrame, alpha: pd.DataFrame | None, reports_dir: Path) -> Path | None:
    if alpha is None:
        return None
    p = _metadata_panel(universe, alpha)
    cols = [c for c in ["quality", "rarity", "type", "metatype"] if c in p.columns]
    if len(cols) < 2:
        return None
    plt = _setup_matplotlib()
    from matplotlib import colors as mcolors
    p = p.copy()
    p["archetype"] = p[cols].astype("string").agg(" | ".join, axis=1)
    base = p["archetype"].value_counts(normalize=True).rename("universe_share")
    top = p.loc[p["is_top10"], "archetype"].value_counts(normalize=True).rename("top10_share")
    counts = p.loc[p["is_top10"], "archetype"].value_counts().rename("top10_count")
    a = pd.concat([base, top, counts], axis=1).fillna(0)
    a["universe_share_pct"] = a["universe_share"] * 100
    a["top10_share_pct"] = a["top10_share"] * 100
    a["lift"] = np.where(a["universe_share"] > 0, a["top10_share"] / a["universe_share"], np.nan)
    a["enrichment_pp"] = a["top10_share_pct"] - a["universe_share_pct"]
    a = a[(a["top10_count"] >= 20) | (a["top10_share_pct"] >= 0.5)].sort_values(["enrichment_pp", "top10_share_pct"], ascending=False).head(18)
    if not len(a):
        return None
    finite = a["lift"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    lift_min, lift_max = float(finite.min()), float(finite.max())
    vmin = min(0.5, lift_min, 0.99)
    vmax = max(1.01, lift_max)
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = 55 + np.sqrt(a["top10_count"].astype(float).clip(lower=1)) * 22
    sc = ax.scatter(a["universe_share_pct"], a["top10_share_pct"], s=sizes, c=finite, cmap="PuOr_r", norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax), alpha=0.86, edgecolor="0.25", linewidth=0.45)
    max_axis = max(float(a["universe_share_pct"].max()), float(a["top10_share_pct"].max()), 1) * 1.16
    ax.plot([0, max_axis], [0, max_axis], color="0.25", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("Доля во всей торговой вселенной, %")
    ax.set_ylabel("Доля в top-10% альфа-корзине, %")
    ax.set_title("Архетипы предметов: концентрация в верхней альфа-корзине")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.035)
    cbar.set_label("Коэффициент концентрации: доля в top-10% / доля во вселенной")
    for name, row in a.head(8).iterrows():
        label = " | ".join(_ru_cat(part, 18) for part in str(name).split(" | "))
        ax.annotate(label, xy=(row["universe_share_pct"], row["top10_share_pct"]), xytext=(5, 5), textcoords="offset points", fontsize=8, bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "0.75", "alpha": 0.88})
    fig.text(0.012, 0.012, "Точки выше пунктирной линии перепредставлены среди наиболее высоко ранжированных предметов.", fontsize=9, color="0.30")
    out = reports_dir / "Рис_4_8_архетипы_предметов_в_top10_альфа_корзине.png"
    _savefig(plt, out)
    return out


def write_tables_index(reports_dir: Path, t41: pd.DataFrame, t42: pd.DataFrame, figures: list[Path]) -> None:
    md = ["# Таблицы и рисунки главы 4 (русская версия)", ""]
    md += ["## Таблица 4.1. Хронологическое разбиение выборки", t41.to_markdown(index=False), ""]
    md += ["## Таблица 4.2. Гиперпараметры моделей и характеристики торговой вселенной", t42.to_markdown(index=False), ""]
    md += ["## Сформированные рисунки", ""]
    for p in figures:
        md.append(f"- `{p.name}`")
    (reports_dir / "20_таблицы_и_рисунки_главы_4_RU.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cost-for-matrix", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(cfg)
    alpha = _load_alpha(cfg)
    grid = _load_cost_grid(cfg)

    t41 = table_4_1(cfg, universe, reports_dir)
    t42 = table_4_2(cfg, universe, reports_dir)

    figures: list[Path] = []
    for maybe in [
        fig_4_1(cfg, universe, reports_dir),
        fig_4_2(cfg, reports_dir),
        fig_alpha_deciles_ru(cfg, alpha, reports_dir),
        fig_rolling_spread_ru(cfg, alpha, reports_dir),
        fig_top10_cost_ru(grid, reports_dir),
        fig_excess_heatmap_ru(grid, reports_dir, cost=args.cost_for_matrix),
        fig_metadata_preference_ru(universe, alpha, reports_dir),
        fig_archetype_lift_ru(universe, alpha, reports_dir),
    ]:
        if maybe is not None:
            figures.append(maybe)

    write_tables_index(reports_dir, t41, t42, figures)
    _write_json(reports_dir / "20_chapter4_ru_outputs_manifest.json", {"tables": ["T4_1_хронологическое_разбиение_выборки.csv", "T4_2_гиперпараметры_и_торговая_вселенная.csv"], "figures": [p.name for p in figures]})

    print("Сформированы русскоязычные таблицы и рисунки главы 4:")
    for p in figures:
        print(" -", p)
    print(" -", reports_dir / "T4_1_хронологическое_разбиение_выборки.csv")
    print(" -", reports_dir / "T4_2_гиперпараметры_и_торговая_вселенная.csv")


if __name__ == "__main__":
    main()
