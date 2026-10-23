#!/usr/bin/env python
"""
Russian academic tables and figures for model quality and alpha-signal diagnostics.

Creates:
  T4.3. Качество режимной и альфа-классификации
  T4.4. Доходность по децилям alpha_pred + D10-D1 Newey-West inference
  T4.5. Устойчивость сигнала по подпериодам
  Рис. 4.2. ROC-кривые: режимная модель, alpha top-label, alpha bottom-label
  Рис. 4.2б. ROC-кривая итогового alpha_pred: верхний квантиль против нижнего
  Рис. 4.3. Средняя будущая доходность по децилям alpha_pred
  Рис. 4.4. Скользящий top-minus-bottom spread во времени

Run from repo root:
  python scripts/21_chapter4_quality_signal_tables_figures.py --config configs/cs2_trading_rebuilt.yaml
"""
from __future__ import annotations

import argparse
import json
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

try:
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except Exception as exc:  # fail early with a clear message at runtime
    raise RuntimeError("scikit-learn is required for Table 4.3 and ROC figures") from exc

SPLIT_RU = {
    "train": "Обучающая выборка",
    "val": "Валидационная выборка",
    "validation": "Валидационная выборка",
    "test": "Тестовая выборка",
}

TASK_RU = {
    "regime": "Режимная модель",
    "alpha_top": "Альфа-модель: верхний квантиль",
    "alpha_bottom": "Альфа-модель: нижний квантиль",
}

TARGET_CANDIDATES = [
    "future_excess_gross_return_30d",
    "future_gross_return_30d",
    "future_net_return_30d",
]


def _available_parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return set(pd.read_parquet(path).head(0).columns)
        except Exception:
            return set()


def _read_parquet_existing(path: Path, wanted: list[str]) -> pd.DataFrame:
    available = _available_parquet_columns(path)
    cols = [c for c in wanted if c in available]
    if not cols:
        raise KeyError(f"None of requested columns exist in {path}: {wanted}")
    return pd.read_parquet(path, columns=cols)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except Exception:
        return "—"
    if not np.isfinite(xf):
        return "—"
    return f"{xf:.{nd}f}".replace(".", ",")


def _fmt_pct(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except Exception:
        return "—"
    if not np.isfinite(xf):
        return "—"
    return f"{xf:.{nd}f}".replace(".", ",")


def _safe_to_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_markdown(path, index=False)
    except Exception:
        path.write_text(df.to_csv(index=False, sep="\t"), encoding="utf-8")


def _save_table(df: pd.DataFrame, reports_dir: Path, stem: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    _safe_to_markdown(df, reports_dir / f"{stem}.md")


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
        "grid.alpha": 0.24,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _savefig(plt, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=230, bbox_inches="tight")
    plt.close()
    return path


def _split_ranges(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    s = cfg["splits"]
    return {
        "train": (pd.Timestamp(s["train_start"]).normalize(), pd.Timestamp(s["train_end"]).normalize()),
        "val": (pd.Timestamp(s["val_start"]).normalize(), pd.Timestamp(s["val_end"]).normalize()),
        "test": (pd.Timestamp(s["test_start"]).normalize(), pd.Timestamp(s["test_end"]).normalize()),
    }


def _split_mask(df: pd.DataFrame, cfg: dict[str, Any], split: str) -> pd.Series:
    start, end = _split_ranges(cfg)[split]
    return (df["date"] >= start) & (df["date"] <= end)


def _load_regime_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(cfg, "models_dir") / "regime_predictions.parquet"
    wanted = ["date", "regime_target", "regime_prob", "regime_pred"]
    df = _read_parquet_existing(path, wanted)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["regime_target"] = pd.to_numeric(df["regime_target"], errors="coerce")
    df["regime_prob"] = pd.to_numeric(df["regime_prob"], errors="coerce")
    if "regime_pred" not in df.columns:
        df["regime_pred"] = (df["regime_prob"] >= 0.5).astype(int)
    else:
        df["regime_pred"] = pd.to_numeric(df["regime_pred"], errors="coerce")
    return df.dropna(subset=["regime_target", "regime_prob"])


def _load_alpha_predictions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, float, float]:
    models_dir = resolve_path(cfg, "models_dir")
    report = _read_json(models_dir / "alpha_report.json")
    target_col = str(report.get("target_col") or cfg.get("research", {}).get("primary_target") or "")
    alpha_cfg = cfg.get("modeling", {}).get("alpha_model", {})
    q_low = float(report.get("quantile_low", alpha_cfg.get("rank_quantile_low", 0.30)))
    q_high = float(report.get("quantile_high", alpha_cfg.get("rank_quantile_high", 0.70)))

    path = models_dir / "alpha_predictions.parquet"
    wanted = [
        "date",
        "market_hash_name",
        "alpha_pred",
        "alpha_rank_score_raw",
        "alpha_prob_top",
        "alpha_prob_bottom",
        target_col,
    ] + TARGET_CANDIDATES
    wanted = [c for c in dict.fromkeys(wanted) if c]
    df = _read_parquet_existing(path, wanted)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ["alpha_pred", "alpha_rank_score_raw", "alpha_prob_top", "alpha_prob_bottom"] + TARGET_CANDIDATES + ([target_col] if target_col else []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if not target_col or target_col not in df.columns:
        for c in TARGET_CANDIDATES:
            if c in df.columns and df[c].notna().any():
                target_col = c
                break
    if not target_col or target_col not in df.columns:
        raise KeyError("Could not identify alpha target column in alpha_predictions.parquet")

    df = df.dropna(subset=["alpha_pred", target_col]).copy()
    return df, target_col, q_low, q_high


def _add_alpha_labels(df: pd.DataFrame, target_col: str, q_low: float, q_high: float) -> pd.DataFrame:
    out = df.copy()
    low_thr = out.groupby("date")[target_col].transform(lambda s: s.quantile(q_low))
    high_thr = out.groupby("date")[target_col].transform(lambda s: s.quantile(q_high))
    out["bottom_label"] = (out[target_col] <= low_thr).astype(int)
    out["top_label"] = (out[target_col] >= high_thr).astype(int)
    return out


def _assign_daily_deciles(df: pd.DataFrame, score_col: str = "alpha_pred") -> pd.Series:
    def one_day(s: pd.Series) -> pd.Series:
        ranks = pd.to_numeric(s, errors="coerce").rank(method="first")
        try:
            return pd.qcut(ranks, 10, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)
    return df.groupby("date", group_keys=False)[score_col].transform(one_day)


def _classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float | int | None]:
    mask = y_true.notna() & y_prob.notna()
    y = pd.to_numeric(y_true[mask], errors="coerce").astype(float)
    p = pd.to_numeric(y_prob[mask], errors="coerce").clip(1e-9, 1 - 1e-9)
    mask2 = y.notna() & p.notna()
    y = y[mask2].astype(int)
    p = p[mask2].astype(float)
    if len(y) == 0:
        return {"n": 0, "auc": None, "accuracy": None, "precision": None, "recall": None, "log_loss": None}
    pred = (p >= threshold).astype(int)
    out: dict[str, float | int | None] = {
        "n": int(len(y)),
        "auc": None,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "log_loss": None,
    }
    if y.nunique() == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def table_4_3(cfg: dict[str, Any], reports_dir: Path) -> pd.DataFrame:
    regime = _load_regime_predictions(cfg)
    alpha, target_col, q_low, q_high = _load_alpha_predictions(cfg)
    alpha = _add_alpha_labels(alpha, target_col, q_low, q_high)

    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rg = regime.loc[_split_mask(regime, cfg, split)]
        m = _classification_metrics(rg["regime_target"], rg["regime_prob"])
        rows.append({
            "Модель / задача": TASK_RU["regime"],
            "Выборка": SPLIT_RU[split],
            "AUC": _fmt_num(m["auc"], 3),
            "Accuracy": _fmt_num(m["accuracy"], 3),
            "Precision": _fmt_num(m["precision"], 3),
            "Recall": _fmt_num(m["recall"], 3),
            "Log-loss": _fmt_num(m["log_loss"], 3),
            "n": int(m["n"] or 0),
        })

        ag = alpha.loc[_split_mask(alpha, cfg, split)]
        for key, label_col, prob_col in [
            ("alpha_top", "top_label", "alpha_prob_top"),
            ("alpha_bottom", "bottom_label", "alpha_prob_bottom"),
        ]:
            if prob_col not in ag.columns:
                continue
            m = _classification_metrics(ag[label_col], ag[prob_col])
            rows.append({
                "Модель / задача": TASK_RU[key],
                "Выборка": SPLIT_RU[split],
                "AUC": _fmt_num(m["auc"], 3),
                "Accuracy": _fmt_num(m["accuracy"], 3),
                "Precision": _fmt_num(m["precision"], 3),
                "Recall": _fmt_num(m["recall"], 3),
                "Log-loss": _fmt_num(m["log_loss"], 3),
                "n": int(m["n"] or 0),
            })

    out = pd.DataFrame(rows)
    _save_table(out, reports_dir, "T4_3_качество_режимной_и_альфа_классификации")
    return out


def _newey_west_se_mean(x: pd.Series, lag: int) -> tuple[float | None, float | None]:
    vals = pd.to_numeric(x, errors="coerce").dropna().astype(float).to_numpy()
    t = len(vals)
    if t < 2:
        return None, None
    lag = int(max(0, min(lag, t - 1)))
    u = vals - vals.mean()
    gamma0 = float(np.dot(u, u) / t)
    var = gamma0
    for ell in range(1, lag + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = float(np.dot(u[ell:], u[:-ell]) / t)
        var += 2.0 * weight * gamma
    var_mean = max(var / t, 0.0)
    se = math.sqrt(var_mean)
    t_stat = float(vals.mean() / se) if se > 0 else None
    return se, t_stat


def _daily_decile_panel(alpha: pd.DataFrame, cfg: dict[str, Any], target_col: str, split: str = "test") -> pd.DataFrame:
    df = alpha.loc[_split_mask(alpha, cfg, split)].dropna(subset=["alpha_pred", target_col]).copy()
    df["alpha_decile"] = _assign_daily_deciles(df, "alpha_pred")
    df = df.dropna(subset=["alpha_decile"])
    df["alpha_decile"] = df["alpha_decile"].astype(int) + 1
    df["target_pct"] = pd.to_numeric(df[target_col], errors="coerce") * 100.0
    return df.dropna(subset=["target_pct"])


def _daily_spreads_from_deciles(dec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in dec.groupby("date", sort=True):
        top = g.loc[g["alpha_decile"] == 10, "target_pct"].mean()
        bot = g.loc[g["alpha_decile"] == 1, "target_pct"].mean()
        if pd.notna(top) and pd.notna(bot):
            rows.append({"date": pd.Timestamp(d), "spread_pct": float(top - bot)})
    return pd.DataFrame(rows)


def table_4_4(cfg: dict[str, Any], reports_dir: Path) -> pd.DataFrame:
    alpha, target_col, q_low, q_high = _load_alpha_predictions(cfg)
    dec = _daily_decile_panel(alpha, cfg, target_col, split="test")
    rows: list[dict[str, Any]] = []
    for d in range(1, 11):
        r = dec.loc[dec["alpha_decile"] == d, "target_pct"]
        rows.append({
            "Портфель": f"D{d}",
            "Среднее, %": _fmt_pct(r.mean(), 2),
            "Медиана, %": _fmt_pct(r.median(), 2),
            "Ст. откл., %": _fmt_pct(r.std(ddof=1), 2),
            "n": int(r.notna().sum()),
            "Newey-West SE": "—",
            "t-статистика": "—",
        })

    spread_daily = _daily_spreads_from_deciles(dec)
    horizon = int(cfg.get("research", {}).get("primary_alpha_horizon_days", 30))
    se, t_stat = _newey_west_se_mean(spread_daily["spread_pct"] if len(spread_daily) else pd.Series(dtype=float), lag=horizon)
    s = spread_daily["spread_pct"] if len(spread_daily) else pd.Series(dtype=float)
    rows.append({
        "Портфель": "D10 − D1",
        "Среднее, %": _fmt_pct(s.mean(), 2),
        "Медиана, %": _fmt_pct(s.median(), 2),
        "Ст. откл., %": _fmt_pct(s.std(ddof=1), 2),
        "n": int(s.notna().sum()),
        "Newey-West SE": _fmt_pct(se, 2),
        "t-статистика": _fmt_num(t_stat, 2),
    })
    out = pd.DataFrame(rows)
    _save_table(out, reports_dir, "T4_4_доходность_по_децилям_alpha_pred")
    return out


def _subperiods_for_test(cfg: dict[str, Any], alpha: pd.DataFrame, n_parts: int = 4) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    start, end = _split_ranges(cfg)["test"]
    dates = pd.DatetimeIndex(sorted(alpha.loc[(alpha["date"] >= start) & (alpha["date"] <= end), "date"].dropna().unique()))
    if len(dates) == 0:
        return []
    parts = np.array_split(dates, min(n_parts, max(1, len(dates))))
    out = [("Полный test", dates.min(), dates.max())]
    for i, p in enumerate(parts, start=1):
        if len(p):
            out.append((f"Test, подпериод {i}", pd.Timestamp(p.min()), pd.Timestamp(p.max())))
    return out


def _period_metrics(alpha: pd.DataFrame, target_col: str, q_low: float, q_high: float, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    p = alpha.loc[(alpha["date"] >= start) & (alpha["date"] <= end)].copy()
    if len(p) == 0:
        return {"n_days": 0, "n_item_days": 0, "spread": None, "spearman": None, "auc": None}
    p = _add_alpha_labels(p, target_col, q_low, q_high)
    dec = p.dropna(subset=["alpha_pred", target_col]).copy()
    dec["alpha_decile"] = _assign_daily_deciles(dec, "alpha_pred")
    dec = dec.dropna(subset=["alpha_decile"])
    dec["alpha_decile"] = dec["alpha_decile"].astype(int) + 1
    dec["target_pct"] = dec[target_col] * 100.0
    spread = _daily_spreads_from_deciles(dec)["spread_pct"].mean() if len(dec) else None

    sp_rows = []
    for _, g in p.groupby("date"):
        if len(g) >= 5 and g["alpha_pred"].nunique() > 1 and g[target_col].nunique() > 1:
            sp_rows.append(g["alpha_pred"].corr(g[target_col], method="spearman"))
    mean_spearman = float(pd.Series(sp_rows).dropna().mean()) if len(pd.Series(sp_rows).dropna()) else None

    auc = None
    if "alpha_prob_top" in p.columns:
        y = p["top_label"].astype(int)
        score = p["alpha_prob_top"]
        if y.nunique() == 2 and score.notna().any():
            auc = float(roc_auc_score(y, score))

    return {
        "n_days": int(p["date"].nunique()),
        "n_item_days": int(len(p)),
        "spread": spread,
        "spearman": mean_spearman,
        "auc": auc,
    }


def table_4_5(cfg: dict[str, Any], reports_dir: Path, n_parts: int = 4) -> pd.DataFrame:
    alpha, target_col, q_low, q_high = _load_alpha_predictions(cfg)
    rows = []
    for name, start, end in _subperiods_for_test(cfg, alpha, n_parts=n_parts):
        m = _period_metrics(alpha, target_col, q_low, q_high, start, end)
        rows.append({
            "Период": name,
            "Дата начала": start.strftime("%d.%m.%Y"),
            "Дата окончания": end.strftime("%d.%m.%Y"),
            "Дней": m["n_days"],
            "Item-days": m["n_item_days"],
            "D10−D1 spread, п.п.": _fmt_pct(m["spread"], 2),
            "Средний Spearman": _fmt_num(m["spearman"], 3),
            "AUC top-label": _fmt_num(m["auc"], 3),
        })
    out = pd.DataFrame(rows)
    _save_table(out, reports_dir, "T4_5_устойчивость_сигнала_по_подпериодам")
    return out


def _plot_roc_curve(ax, y_true: pd.Series, score: pd.Series, label: str) -> None:
    mask = y_true.notna() & score.notna()
    y = pd.to_numeric(y_true[mask], errors="coerce").astype(float)
    s = pd.to_numeric(score[mask], errors="coerce")
    mask2 = y.notna() & s.notna()
    y = y[mask2].astype(int)
    s = s[mask2].astype(float)
    if len(y) == 0 or y.nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ax.plot(fpr, tpr, linewidth=2.1, label=f"{label} (AUC={auc:.3f})")


def _overall_alpha_extreme_panel(alpha: pd.DataFrame, target_col: str, q_low: float, q_high: float) -> pd.DataFrame:
    """Build a binary panel for the final alpha signal as a whole.

    Positive class: item is in the upper daily quantile of future return.
    Negative class: item is in the lower daily quantile of future return.
    Middle observations are excluded, so the ROC evaluates whether the final
    ranking score alpha_pred orders future winners above future losers.
    """
    df = alpha.dropna(subset=["date", "alpha_pred", target_col]).copy()
    low_thr = df.groupby("date")[target_col].transform(lambda s: s.quantile(q_low))
    high_thr = df.groupby("date")[target_col].transform(lambda s: s.quantile(q_high))
    df["overall_alpha_label"] = np.nan
    df.loc[df[target_col] <= low_thr, "overall_alpha_label"] = 0
    df.loc[df[target_col] >= high_thr, "overall_alpha_label"] = 1
    return df.dropna(subset=["overall_alpha_label", "alpha_pred"]).copy()


def fig_4_2b_overall_alpha_roc(cfg: dict[str, Any], reports_dir: Path) -> Path:
    """ROC for the final composite alpha_pred, not the separate top/bottom heads."""
    plt = _setup_matplotlib()
    alpha, target_col, q_low, q_high = _load_alpha_predictions(cfg)
    overall = _overall_alpha_extreme_panel(alpha, target_col, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        g = overall.loc[_split_mask(overall, cfg, split)]
        _plot_roc_curve(
            ax,
            g["overall_alpha_label"],
            g["alpha_pred"],
            "Итоговый alpha_pred",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.35", label="Случайное ранжирование")
        ax.set_title(SPLIT_RU[split])
        ax.set_xlabel("Доля ложноположительных классификаций")
        ax.set_ylabel("Доля истинноположительных классификаций")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=True, loc="lower right")
        ax.grid(True, alpha=0.24)
        ax.text(
            0.02, 0.98,
            f"Положительный класс: верхние {int((1 - q_high) * 100)}%\n"
            f"Отрицательный класс: нижние {int(q_low * 100)}%",
            transform=ax.transAxes,
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.70", "alpha": 0.90},
        )
    fig.suptitle("Рисунок 4.2б — ROC-кривая итогового альфа-сигнала", y=1.02)
    return _savefig(plt, reports_dir / "Рис_4_2б_ROC_итогового_alpha_pred.png")


def fig_4_2_roc(cfg: dict[str, Any], reports_dir: Path) -> Path:
    plt = _setup_matplotlib()
    regime = _load_regime_predictions(cfg)
    alpha, target_col, q_low, q_high = _load_alpha_predictions(cfg)
    alpha = _add_alpha_labels(alpha, target_col, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        rg = regime.loc[_split_mask(regime, cfg, split)]
        ag = alpha.loc[_split_mask(alpha, cfg, split)]
        _plot_roc_curve(ax, rg["regime_target"], rg["regime_prob"], "Режимная модель")
        if "alpha_prob_top" in ag.columns:
            _plot_roc_curve(ax, ag["top_label"], ag["alpha_prob_top"], "Альфа top-label")
        if "alpha_prob_bottom" in ag.columns:
            _plot_roc_curve(ax, ag["bottom_label"], ag["alpha_prob_bottom"], "Альфа bottom-label")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.35", label="Случайная классификация")
        ax.set_title(SPLIT_RU[split])
        ax.set_xlabel("Доля ложноположительных классификаций")
        ax.set_ylabel("Доля истинноположительных классификаций")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=True, loc="lower right")
    fig.suptitle("Рисунок 4.2 — ROC-кривые классификационных моделей", y=1.02)
    return _savefig(plt, reports_dir / "Рис_4_2_ROC_кривые_классификационных_моделей.png")


def fig_4_3_decile_bars(cfg: dict[str, Any], reports_dir: Path) -> Path:
    plt = _setup_matplotlib()
    alpha, target_col, _, _ = _load_alpha_predictions(cfg)
    dec = _daily_decile_panel(alpha, cfg, target_col, split="test")
    agg = dec.groupby("alpha_decile", as_index=False).agg(
        mean=("target_pct", "mean"),
        sd=("target_pct", "std"),
        n=("target_pct", "size"),
    )
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.bar(agg["alpha_decile"], agg["mean"], yerr=1.96 * agg["se"], capsize=4, width=0.76)
    # Visually emphasize D10 and D1 without overloading the figure.
    if len(bars) >= 10:
        bars[0].set_alpha(0.72)
        bars[-1].set_linewidth(1.3)
        bars[-1].set_edgecolor("0.1")
    ax.axhline(0, color="0.15", linewidth=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Дециль alpha_pred (10 — наивысший прогнозный ранг)")
    ax.set_ylabel("Средняя будущая доходность, %")
    ax.set_title("Рисунок 4.3 — Средняя будущая доходность по децилям alpha_pred")
    if len(agg) >= 10:
        spread = float(agg.loc[agg["alpha_decile"] == 10, "mean"].mean() - agg.loc[agg["alpha_decile"] == 1, "mean"].mean())
        ax.text(0.02, 0.96, f"D10−D1: {spread:+.2f} п.п.", transform=ax.transAxes, va="top", bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.65", "alpha": 0.92})
    ax.grid(True, axis="y", alpha=0.25)
    return _savefig(plt, reports_dir / "Рис_4_3_средняя_будущая_доходность_по_децилям_alpha_pred.png")


def fig_4_4_rolling_spread(cfg: dict[str, Any], reports_dir: Path, window: int = 90) -> Path:
    plt = _setup_matplotlib()
    alpha, target_col, _, _ = _load_alpha_predictions(cfg)
    dec = _daily_decile_panel(alpha, cfg, target_col, split="test")
    spreads = _daily_spreads_from_deciles(dec).sort_values("date")
    if len(spreads) == 0:
        raise ValueError("No daily D10-D1 spreads available for rolling spread figure")
    min_periods = max(5, min(window // 3, len(spreads)))
    spreads["rolling"] = spreads["spread_pct"].rolling(window, min_periods=min_periods).mean()
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.plot(spreads["date"], spreads["rolling"], linewidth=2.4, label=f"Скользящее среднее, {window} дней")
    ax.fill_between(spreads["date"], 0, spreads["rolling"], where=spreads["rolling"] >= 0, alpha=0.16)
    ax.fill_between(spreads["date"], 0, spreads["rolling"], where=spreads["rolling"] < 0, alpha=0.10)
    ax.axhline(0, color="0.15", linewidth=0.9)
    ax.set_xlabel("Дата сигнала")
    ax.set_ylabel("D10−D1 spread, п.п.")
    ax.set_title("Рисунок 4.4 — Скользящий top-minus-bottom spread во времени")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    return _savefig(plt, reports_dir / "Рис_4_4_скользящий_top_minus_bottom_spread.png")


def write_index(reports_dir: Path, tables: list[str], figures: list[Path]) -> None:
    md = ["# Таблицы и рисунки качества моделей и альфа-сигнала", ""]
    md.append("## Таблицы")
    for t in tables:
        md.append(f"- `{t}.csv` / `{t}.md`")
    md += ["", "## Рисунки"]
    for p in figures:
        md.append(f"- `{p.name}`")
    (reports_dir / "21_таблицы_и_рисунки_качества_моделей.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subperiods", type=int, default=4)
    ap.add_argument("--rolling-window", type=int, default=90)
    args = ap.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    t43 = table_4_3(cfg, reports_dir)
    t44 = table_4_4(cfg, reports_dir)
    t45 = table_4_5(cfg, reports_dir, n_parts=args.subperiods)

    figures = [
        fig_4_2_roc(cfg, reports_dir),
        fig_4_2b_overall_alpha_roc(cfg, reports_dir),
        fig_4_3_decile_bars(cfg, reports_dir),
        fig_4_4_rolling_spread(cfg, reports_dir, window=args.rolling_window),
    ]
    tables = [
        "T4_3_качество_режимной_и_альфа_классификации",
        "T4_4_доходность_по_децилям_alpha_pred",
        "T4_5_устойчивость_сигнала_по_подпериодам",
    ]
    write_index(reports_dir, tables, figures)

    print("Сформированы таблицы и рисунки качества моделей:")
    for t in tables:
        print(" -", reports_dir / f"{t}.csv")
    for p in figures:
        print(" -", p)


if __name__ == "__main__":
    main()
