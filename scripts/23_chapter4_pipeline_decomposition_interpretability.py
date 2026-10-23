#!/usr/bin/env python
"""
Таблицы 4.9--4.11 и рисунки 4.7--4.9 для декомпозиции пайплайна,
важности признаков и структуры отбора предметов.

Скрипт сохраняет все результаты в отдельную папку:
  data/processed/reports/chapter4_pipeline_decomposition/

Запуск:
  python scripts/23_chapter4_pipeline_decomposition_interpretability.py \
      --config configs/cs2_trading_rebuilt.yaml

Ключевая стратегия по умолчанию:
  top 5%, ребалансировка 90 дней, sell cost = 10%.

Сценарии декомпозиции:
  1) Full pipeline   — базовая alpha-rotation стратегия на итоговом alpha_pred.
  2) No-regime       — контроль без внешнего режимного gate; при наличии
                       counterfactual-score используется alpha_pred_no_regime.
  3) No-alpha        — равновесное удержание пригодной торговой вселенной
                       без ранжирования по alpha_pred.
  4) Random ranking  — alpha_pred заменён случайным ранжированием с фиксированным seed.

Примечание:
  Все режимы стартуют с одинакового капитала и одинакового test-интервала.
  Внешний жёсткий regime-gate здесь не применяется, чтобы таблица 4.9 была
  сопоставима с базовой alpha-rotation сеткой стратегий из таблицы 4.6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
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
from cs2_trading_rebuild.utils import save_table

try:
    from cs2_trading_rebuild import models as model_mod
except Exception:  # pragma: no cover
    model_mod = None

ANNUALIZATION_DAYS = 365
TARGET_COST = 0.10
TARGET_TOP_PCT = 0.05
TARGET_REBALANCE_DAYS = 90
DEFAULT_RANDOM_SEED = 20260518
META_COLS = ["rarity", "type", "quality", "stattrak", "souvenir", "metatype"]
ATTR_FOR_TABLE = ["rarity", "type", "quality", "stattrak", "souvenir"]


# ----------------------------- small utilities -----------------------------

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
        raise KeyError(f"В файле {path} нет ни одной из нужных колонок: {wanted}")
    return pd.read_parquet(path, columns=cols)


def _fmt_num(x: Any, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"


def _write_md_table(df: pd.DataFrame, path: Path, digits: int = 2) -> None:
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_numeric_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.{digits}f}")
        else:
            tmp[col] = tmp[col].fillna("—")
    path.write_text(tmp.to_markdown(index=False), encoding="utf-8")


def _safe_category(x: Any) -> str:
    if pd.isna(x):
        return "Не указано"
    s = str(x)
    if s.lower() in {"true", "1", "yes"}:
        return "Да"
    if s.lower() in {"false", "0", "no"}:
        return "Нет"
    return s


def _short_label(x: Any, n: int = 34) -> str:
    s = _safe_category(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def _strategy_label(mode: str) -> str:
    return {
        "full_pipeline": "Full pipeline",
        "no_regime": "No-regime",
        "no_alpha": "No-alpha",
        "random_ranking": "Random ranking",
    }.get(mode, mode)


def _sortino(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    r = eq.pct_change().dropna()
    downside = r[r < 0]
    if len(downside) < 2:
        return None
    sd = float(downside.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(math.sqrt(ANNUALIZATION_DAYS) * float(r.mean()) / sd)


def _sharpe(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None
    r = eq.pct_change().dropna()
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(math.sqrt(ANNUALIZATION_DAYS) * float(r.mean()) / sd)


def _max_drawdown_pct(equity: pd.Series) -> float | None:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) == 0:
        return None
    runmax = eq.cummax().replace(0, np.nan)
    dd = (eq - runmax) / runmax
    return float(dd.min() * 100.0) if dd.notna().any() else None


def _roi_pct(initial: float, final: float) -> float | None:
    if initial <= 0:
        return None
    return float((final / initial - 1.0) * 100.0)


def _annualized_turnover(trades: pd.DataFrame, initial_capital: float, n_days: int) -> float | None:
    if initial_capital <= 0 or n_days <= 0:
        return None
    if len(trades) == 0 or "gross_value_rub" not in trades.columns:
        return 0.0
    turnover = float(pd.to_numeric(trades["gross_value_rub"], errors="coerce").fillna(0.0).sum())
    return float(turnover / initial_capital * ANNUALIZATION_DAYS / n_days)


def _feature_group(name: str) -> str:
    n = str(name).lower()
    if name in META_COLS or any(k in n for k in ["quality", "rarity", "stattrak", "souvenir", "metatype"]):
        return "Атрибутивные"
    if any(k in n for k in ["turnover", "volume", "units", "sold", "liquid", "tradability"]):
        return "Ликвидность"
    if any(k in n for k in ["vol", "std", "drawdown", "range", "spread"]):
        return "Волатильность"
    if any(k in n for k in ["regime", "market", "universe", "breadth", "tradable"]):
        return "Рыночный контекст"
    return "Рыночная динамика"


# ----------------------------- data loading -----------------------------

def _load_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    models_dir = resolve_path(cfg, "models_dir")
    alpha_path = models_dir / "alpha_predictions.parquet"
    regime_path = models_dir / "regime_predictions.parquet"

    if not alpha_path.exists():
        raise FileNotFoundError(f"Не найден файл прогнозов alpha: {alpha_path}")
    if not regime_path.exists():
        raise FileNotFoundError(f"Не найден файл прогнозов режима: {regime_path}")

    price = pd.read_parquet(feature_path, columns=["date", "market_hash_name", "price_rub"])
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    price["price_rub"] = pd.to_numeric(price["price_rub"], errors="coerce")
    price = price.dropna(subset=["price_rub"]).drop_duplicates(["date", "market_hash_name"], keep="last")
    price_wide = price.pivot(index="date", columns="market_hash_name", values="price_rub").sort_index()
    price_wide_mtm = price_wide.ffill()

    uni_cols = ["date", "market_hash_name", "tradeable_today", "tradability_score"] + META_COLS
    universe = _read_parquet_existing(universe_path, uni_cols)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["tradeable_today"] = universe.get("tradeable_today", False).fillna(False).astype(bool)
    if "tradability_score" not in universe.columns:
        universe["tradability_score"] = 0.0
    universe["tradability_score"] = pd.to_numeric(universe["tradability_score"], errors="coerce").fillna(0.0)
    for c in META_COLS:
        if c in universe.columns:
            universe[c] = universe[c].map(_safe_category)
    universe = universe.drop_duplicates(["date", "market_hash_name"], keep="last")

    alpha_cols = ["date", "market_hash_name", "alpha_pred", "alpha_rank_score_raw", "future_excess_gross_return_30d"] + META_COLS
    alpha = _read_parquet_existing(alpha_path, alpha_cols)
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha["alpha_pred"] = pd.to_numeric(alpha["alpha_pred"], errors="coerce")
    if "alpha_rank_score_raw" in alpha.columns:
        alpha["alpha_rank_score_raw"] = pd.to_numeric(alpha["alpha_rank_score_raw"], errors="coerce")
    if "future_excess_gross_return_30d" in alpha.columns:
        alpha["future_excess_gross_return_30d"] = pd.to_numeric(alpha["future_excess_gross_return_30d"], errors="coerce")
    for c in META_COLS:
        if c in alpha.columns:
            alpha[c] = alpha[c].map(_safe_category)
    alpha = alpha.dropna(subset=["alpha_pred"]).drop_duplicates(["date", "market_hash_name"], keep="last")

    regime = pd.read_parquet(regime_path, columns=[c for c in ["date", "regime_prob", "regime_pred"] if c in _available_parquet_columns(regime_path)])
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    regime["regime_prob"] = pd.to_numeric(regime["regime_prob"], errors="coerce").fillna(0.5)
    regime = regime.drop_duplicates("date", keep="last")

    # Combined daily panel for selection and metadata.
    panel = universe.loc[universe["tradeable_today"], [c for c in ["date", "market_hash_name", "tradeable_today", "tradability_score"] + META_COLS if c in universe.columns]].merge(
        alpha[[c for c in ["date", "market_hash_name", "alpha_pred"] + META_COLS if c in alpha.columns]],
        on=["date", "market_hash_name"],
        how="inner",
        suffixes=("_universe", "_alpha"),
    )
    for c in META_COLS:
        a, u = f"{c}_alpha", f"{c}_universe"
        if a in panel.columns and u in panel.columns:
            panel[c] = panel[a].where(panel[a].notna(), panel[u])
        elif a in panel.columns:
            panel[c] = panel[a]
        elif u in panel.columns:
            panel[c] = panel[u]
        if c in panel.columns:
            panel[c] = panel[c].map(_safe_category)

    return {
        "universe": universe,
        "alpha": alpha,
        "regime": regime,
        "panel": panel,
        "price_wide": price_wide,
        "price_wide_mtm": price_wide_mtm,
    }


def _make_test_dates(price_wide: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg["splits"]["test_start"]).normalize()
    end = pd.Timestamp(cfg["splits"]["test_end"]).normalize()
    dates = sorted(set(price_wide.index) & set(panel["date"].unique()))
    idx = pd.DatetimeIndex(dates)
    idx = idx[(idx >= start) & (idx <= end)]
    if len(idx) == 0:
        raise ValueError("Нет дат для тестового моделирования после пересечения цен, alpha и торговой вселенной.")
    return idx


def _regime_threshold(cfg: dict[str, Any]) -> float:
    policy_path = resolve_path(cfg, "models_dir") / "policy.json"
    policy = _read_json(policy_path)
    thr = policy.get("regime_prob_threshold", None)
    try:
        thr = float(thr)
    except Exception:
        thr = float(cfg.get("research", {}).get("regime_prob_threshold", 0.5))
    if not np.isfinite(thr) or thr < 0.0 or thr > 1.0:
        thr = 0.5
    return float(thr)


# ----------------------------- decomposition simulator -----------------------------

def _random_score(date: pd.Timestamp, item: str, seed: int) -> float:
    key = f"{seed}|{pd.Timestamp(date).date().isoformat()}|{item}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "little") / float(2**64 - 1)


def simulate_component_strategy(
    panel: pd.DataFrame,
    regime: pd.DataFrame,
    price_wide: pd.DataFrame,
    price_wide_mtm: pd.DataFrame,
    sim_dates: pd.DatetimeIndex,
    initial_capital: float,
    mode: str,
    top_pct: float = TARGET_TOP_PCT,
    rebalance_days: int = TARGET_REBALANCE_DAYS,
    sell_cost_rate: float = TARGET_COST,
    regime_threshold: float = 0.5,
    random_seed: int = DEFAULT_RANDOM_SEED,
    execute_next_day: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel[panel["date"].isin(sim_dates)].copy()
    regime_map = regime.set_index("date")["regime_prob"].to_dict()
    daily = {pd.Timestamp(d): g.copy() for d, g in panel.groupby("date", sort=True)}
    price_cols = set(price_wide.columns)
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    pending: dict[pd.Timestamp, list[str]] = {}
    last_signal_date: pd.Timestamp | None = None
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []

    meta_by_date_item = panel.set_index(["date", "market_hash_name"])[[c for c in META_COLS if c in panel.columns]].to_dict("index")

    def px_sell(d: pd.Timestamp, item: str) -> float | None:
        if item not in price_cols:
            return None
        px = price_wide_mtm.loc[d, item]
        if pd.notna(px) and px > 0:
            return float(px)
        return None

    def px_buy(d: pd.Timestamp, item: str) -> float | None:
        if item not in price_cols:
            return None
        px = price_wide.loc[d, item]
        if pd.notna(px) and px > 0:
            return float(px)
        return None

    def current_equity_gross(d: pd.Timestamp) -> float:
        mv = 0.0
        for it, units in positions.items():
            px = px_sell(d, it)
            if px is not None:
                mv += units * px
        return cash + mv

    def execute_target(d: pd.Timestamp, target_items: list[str]) -> None:
        nonlocal cash, positions
        # Match the main friction-sweep simulator: an empty or non-buyable target
        # means "no rebalance", not forced liquidation. Forced exits would be a
        # separate risk-off experiment and would make T4.9 incomparable with T4.6.
        target_items = sorted(set([it for it in target_items if it in price_cols and px_buy(d, it) is not None]))
        if not target_items:
            return
        equity_gross = current_equity_gross(d)
        if equity_gross <= 0:
            return
        budget = equity_gross / len(target_items)
        desired_units: dict[str, int] = {}
        for it in target_items:
            px = px_buy(d, it)
            if px is not None:
                desired_units[it] = int(budget // px)

        # sell first
        for it in list(positions.keys()):
            old = int(positions[it])
            new = int(desired_units.get(it, 0))
            if new >= old:
                continue
            units = old - new
            px = px_sell(d, it)
            if px is None:
                continue
            gross = units * px
            net = gross * (1.0 - sell_cost_rate)
            cash += net
            if new > 0:
                positions[it] = new
            else:
                positions.pop(it, None)
            trade_rows.append({"date": d, "side": "sell", "market_hash_name": it, "price_rub": px, "units": units, "gross_value_rub": gross, "net_value_rub": net, "cost_paid_rub": gross * sell_cost_rate})

        # buy/top up
        for it, desired in desired_units.items():
            old = int(positions.get(it, 0))
            if desired <= old:
                continue
            px = px_buy(d, it)
            if px is None:
                continue
            units = min(desired - old, int(cash // px))
            if units <= 0:
                continue
            gross = units * px
            cash -= gross
            positions[it] = old + units
            trade_rows.append({"date": d, "side": "buy", "market_hash_name": it, "price_rub": px, "units": units, "gross_value_rub": gross, "net_value_rub": gross, "cost_paid_rub": 0.0})

    def choose_target(d: pd.Timestamp) -> list[str]:
        g = daily.get(d)
        if g is None or len(g) == 0:
            return []

        # No external hard regime gate here. The baseline alpha_pred already
        # contains regime information as a soft feature. Applying a second hard
        # gate would change the strategy relative to the main friction-sweep table.
        if mode == "no_alpha":
            # Equal-weight control: all tradeable items, no ranking signal.
            return g["market_hash_name"].astype(str).tolist()

        score_col = "alpha_pred"
        if mode == "no_regime" and "alpha_pred_no_regime" in g.columns:
            score_col = "alpha_pred_no_regime"

        gg = g.dropna(subset=[score_col]).copy()
        if len(gg) == 0:
            return []
        if mode == "random_ranking":
            gg["score"] = [_random_score(d, str(x), random_seed) for x in gg["market_hash_name"]]
        else:
            gg["score"] = pd.to_numeric(gg[score_col], errors="coerce")
        # Match scripts/19_alpha_rotation_friction_sweep.py exactly.
        gg = gg.sort_values(["score", "market_hash_name"], ascending=[False, True])
        k = max(1, int(math.ceil(len(gg) * top_pct)))
        return gg.head(k)["market_hash_name"].astype(str).tolist()

    for i, d0 in enumerate(sim_dates):
        d = pd.Timestamp(d0)
        if d in pending:
            for tgt in pending.pop(d):
                execute_target(d, tgt)

        gross_mv = 0.0
        liq_mv = 0.0
        for it, units in positions.items():
            px = px_sell(d, it)
            if px is not None:
                gross_mv += units * px
                liq_mv += units * px * (1.0 - sell_cost_rate)
                meta = meta_by_date_item.get((d, it), {})
                row = {"date": d, "market_hash_name": it, "units": units, "mark_price_rub": px, "market_value_rub": units * px}
                row.update({c: meta.get(c, "Не указано") for c in META_COLS})
                holding_rows.append(row)

        equity_rows.append({
            "date": d,
            "cash_rub": cash,
            "market_value_rub": gross_mv,
            "liquidated_market_value_rub": liq_mv,
            "equity_rub": cash + gross_mv,
            "liquidated_equity_rub": cash + liq_mv,
            "n_positions": len(positions),
            "regime_prob": float(regime_map.get(d, np.nan)),
        })

        due = last_signal_date is None or (d - last_signal_date).days >= rebalance_days
        if due:
            target = choose_target(d)
            if execute_next_day and i + 1 < len(sim_dates):
                exec_date = pd.Timestamp(sim_dates[i + 1])
            elif not execute_next_day:
                exec_date = d
            else:
                exec_date = None
            if exec_date is not None:
                pending.setdefault(exec_date, []).append(target)
                last_signal_date = d

    return pd.DataFrame(equity_rows), pd.DataFrame(trade_rows), pd.DataFrame(holding_rows)


def _summary_for_mode(mode: str, eq: pd.DataFrame, tr: pd.DataFrame, initial: float) -> dict[str, Any]:
    n_days = int(len(eq))
    final_liq = float(eq["liquidated_equity_rub"].iloc[-1]) if len(eq) else initial
    return {
        "Режим исполнения": _strategy_label(mode),
        "ROI, %": _roi_pct(initial, final_liq),
        "Sharpe": _sharpe(eq["liquidated_equity_rub"]) if len(eq) else None,
        "Sortino": _sortino(eq["liquidated_equity_rub"]) if len(eq) else None,
        "MDD, %": _max_drawdown_pct(eq["liquidated_equity_rub"]) if len(eq) else None,
        "Годовой оборот, × капитала": _annualized_turnover(tr, initial, n_days),
    }


# ----------------------------- PFI -----------------------------

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_alpha_work_for_pfi(cfg: dict[str, Any], max_rows: int, random_seed: int) -> pd.DataFrame:
    """Reconstruct a test-period alpha feature panel close to training code."""
    if model_mod is None:
        raise RuntimeError("Не удалось импортировать cs2_trading_rebuild.models для PFI.")
    feature_path = resolve_path(cfg, "feature_store")
    universe_path = resolve_path(cfg, "universe_daily")
    regime_path = resolve_path(cfg, "models_dir") / "regime_predictions.parquet"

    target_col, base_future_col, _h = model_mod._resolve_alpha_target_spec(cfg)
    numeric_features = list(getattr(model_mod, "FEATURE_NUMERIC_COLS"))
    cat_features = list(getattr(model_mod, "CATEGORICAL_FEATURES"))

    feat_cols = ["date", "market_hash_name", target_col, base_future_col] + numeric_features + cat_features
    feat = _read_parquet_existing(feature_path, feat_cols)
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

    uni = _read_parquet_existing(universe_path, ["date", "market_hash_name", "tradeable_today", "tradability_score"])
    uni["date"] = pd.to_datetime(uni["date"]).dt.normalize()
    uni["tradeable_today"] = uni["tradeable_today"].fillna(False).astype(bool)

    reg = _read_parquet_existing(regime_path, ["date", "regime_prob"])
    reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()

    splits = cfg["splits"]
    start = pd.Timestamp(splits["test_start"]).normalize()
    end = pd.Timestamp(splits["test_end"]).normalize()

    work = feat.merge(uni, on=["date", "market_hash_name"], how="inner")
    work = work.loc[work["tradeable_today"].fillna(False)].copy()
    work = work.merge(reg, on="date", how="left")
    work = work[(work["date"] >= start) & (work["date"] <= end)].copy()

    base_future = pd.to_numeric(work[base_future_col], errors="coerce")
    daily_base = pd.DataFrame({"date": work["date"], base_future_col: base_future}).groupby("date", as_index=False)[base_future_col].median().rename(columns={base_future_col: "__tradable_baseline__"})
    work = work.merge(daily_base, on="date", how="left")
    work[target_col] = (base_future - pd.to_numeric(work["__tradable_baseline__"], errors="coerce")).astype("float32")
    work["tradability_score"] = pd.to_numeric(work["tradability_score"], errors="coerce").fillna(0.0)
    work["regime_prob"] = pd.to_numeric(work["regime_prob"], errors="coerce").fillna(0.5)

    for c in cat_features:
        if c in work.columns:
            work[c] = work[c].astype("string").fillna("__MISSING__")
    work = work.dropna(subset=[target_col]).sort_values(["date", "market_hash_name"]).reset_index(drop=True)
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=random_seed).sort_values(["date", "market_hash_name"]).reset_index(drop=True)
    return work


def _score_alpha_model(saved_model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    if model_mod is None:
        raise RuntimeError("Не удалось импортировать models.py")
    cat_maps = saved_model["category_maps"]
    X = model_mod._build_dense_matrix(df, cat_maps)
    top = saved_model["top_classifier"].predict_proba(X)[:, 1]
    bottom = saved_model["bottom_classifier"].predict_proba(X)[:, 1]
    return np.asarray(top - bottom, dtype=float)


def _safe_spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    s1 = pd.Series(a).astype(float)
    s2 = pd.Series(b).astype(float)
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method="spearman"))


def compute_permutation_importance(cfg: dict[str, Any], out_dir: Path, max_rows: int, n_repeats: int, random_seed: int) -> pd.DataFrame:
    models_dir = resolve_path(cfg, "models_dir")
    model_path = models_dir / "alpha_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден alpha_model.pkl для permutation importance: {model_path}")
    saved_model = _load_pickle(model_path)
    work = _build_alpha_work_for_pfi(cfg, max_rows=max_rows, random_seed=random_seed)
    target_col = saved_model.get("target_col") or model_mod._resolve_alpha_target_spec(cfg)[0]
    y = pd.to_numeric(work[target_col], errors="coerce")
    baseline_score = _score_alpha_model(saved_model, work)
    baseline_metric = _safe_spearman(baseline_score, y)

    numeric_features = [c for c in saved_model.get("numeric_features", []) if c in work.columns]
    categorical_features = [c for c in saved_model.get("categorical_features", []) if c in work.columns]

    rows = []
    rng = np.random.default_rng(random_seed)
    candidate_features = numeric_features + categorical_features
    for feat in candidate_features:
        drops = []
        for _ in range(int(n_repeats)):
            perm = work.copy()
            vals = perm[feat].to_numpy(copy=True)
            rng.shuffle(vals)
            perm[feat] = vals
            perm_score = _score_alpha_model(saved_model, perm)
            metric = _safe_spearman(perm_score, y)
            drops.append(baseline_metric - metric)
        rows.append({
            "feature": feat,
            "feature_group": _feature_group(feat),
            "permutation_importance": float(np.nanmean(drops)),
            "std": float(np.nanstd(drops, ddof=1)) if len(drops) > 1 else 0.0,
            "baseline_spearman": baseline_metric,
            "n_rows": int(len(work)),
        })
    imp = pd.DataFrame(rows).sort_values("permutation_importance", ascending=False)

    top_num = imp[imp["feature"].isin(numeric_features)].head(10)
    cats = imp[imp["feature"].isin(categorical_features)]
    selected = pd.concat([top_num, cats], axis=0).drop_duplicates("feature")
    selected = selected.sort_values("permutation_importance", ascending=False).reset_index(drop=True)

    table = selected.rename(columns={
        "feature": "Признак",
        "feature_group": "Смысловая группа",
        "permutation_importance": "Permutation importance",
        "std": "Std по перестановкам",
    })[["Признак", "Смысловая группа", "Permutation importance", "Std по перестановкам"]]
    table.to_csv(out_dir / "T4_10_permutation_feature_importance_alpha_model.csv", index=False)
    _write_md_table(table, out_dir / "T4_10_permutation_feature_importance_alpha_model.md", digits=4)
    return table


# ----------------------------- composition tables/plots -----------------------------

def _daily_shares(df: pd.DataFrame, attr: str, date_col: str = "date", weight_col: str | None = None) -> pd.DataFrame:
    if len(df) == 0 or attr not in df.columns:
        return pd.DataFrame(columns=[date_col, attr, "share"])
    d = df[[date_col, attr] + ([weight_col] if weight_col and weight_col in df.columns else [])].copy()
    d[attr] = d[attr].map(_safe_category)
    if weight_col and weight_col in d.columns:
        d["w"] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0)
    else:
        d["w"] = 1.0
    sums = d.groupby([date_col, attr], as_index=False)["w"].sum()
    totals = d.groupby(date_col, as_index=False)["w"].sum().rename(columns={"w": "total"})
    out = sums.merge(totals, on=date_col, how="left")
    out["share"] = np.where(out["total"] > 0, out["w"] / out["total"], np.nan)
    return out[[date_col, attr, "share"]]


def build_selection_structure_table(holdings: pd.DataFrame, panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for attr in ATTR_FOR_TABLE:
        if attr not in panel.columns:
            continue
        basket = _daily_shares(holdings, attr)
        universe = _daily_shares(panel, attr)
        b_mean = basket.groupby(attr)["share"].mean() if len(basket) else pd.Series(dtype=float)
        u_mean = universe.groupby(attr)["share"].mean() if len(universe) else pd.Series(dtype=float)
        cats = sorted(set(b_mean.index.astype(str)).union(set(u_mean.index.astype(str))))
        for cat in cats:
            b = float(b_mean.get(cat, 0.0)) * 100.0
            u = float(u_mean.get(cat, 0.0)) * 100.0
            rows.append({
                "Атрибутивный разрез": attr,
                "Подгруппа": cat,
                "Доля в корзине, %": b,
                "Доля в торговой вселенной, %": u,
                "Избыток, п.п.": b - u,
            })
    table = pd.DataFrame(rows)
    table = table.sort_values(["Атрибутивный разрез", "Избыток, п.п."], ascending=[True, False])
    table.to_csv(out_dir / "T4_11_структура_отбора_по_характеристикам_предметов.csv", index=False)
    _write_md_table(table, out_dir / "T4_11_структура_отбора_по_характеристикам_предметов.md", digits=2)
    return table


# ----------------------------- figures -----------------------------

def fig_equity_modes(equities: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    styles = {
        "full_pipeline": dict(lw=3.2, ls="-", color="black"),
        "no_regime": dict(lw=2.0, ls="-", color="tab:blue"),
        "no_alpha": dict(lw=2.0, ls="-", color="tab:green"),
        "random_ranking": dict(lw=2.1, ls="--", color="tab:red"),
    }
    for mode in ["full_pipeline", "no_regime", "no_alpha", "random_ranking"]:
        eq = equities.get(mode)
        if eq is None or len(eq) == 0:
            continue
        y = pd.to_numeric(eq["liquidated_equity_rub"], errors="coerce")
        y = y / y.dropna().iloc[0]
        ax.plot(eq["date"], y, label=_strategy_label(mode), **styles.get(mode, {}))
    ax.axhline(1.0, color="0.35", lw=0.9)
    ax.set_title("Рисунок 4.7 — Накопленная стоимость режимов исполнения пайплайна", fontsize=14, pad=12)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Нормированная стоимость портфеля")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "Рис_4_7_equity_curves_режимов_пайплайна.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_permutation_importance(table: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if table is None or len(table) == 0:
        return
    df = table.copy().head(15).sort_values("Permutation importance", ascending=True)
    groups = df["Смысловая группа"].astype(str).unique().tolist()
    palette = {
        "Рыночная динамика": "tab:blue",
        "Ликвидность": "tab:orange",
        "Волатильность": "tab:green",
        "Рыночный контекст": "tab:purple",
        "Атрибутивные": "tab:red",
    }
    colors = [palette.get(g, "0.55") for g in df["Смысловая группа"]]
    xerr = pd.to_numeric(df["Std по перестановкам"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.barh(df["Признак"], df["Permutation importance"], xerr=xerr, color=colors, alpha=0.88, ecolor="0.25", capsize=3)
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_title("Рисунок 4.8 — Permutation feature importance альфа-модели", fontsize=14, pad=12)
    ax.set_xlabel("Снижение Spearman при перестановке признака")
    ax.grid(True, axis="x", alpha=0.25)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette.get(g, "0.55"), label=g) for g in groups]
    ax.legend(handles=handles, title="Смысловая группа", loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "Рис_4_8_permutation_feature_importance_alpha.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _top_categories_overall(df: pd.DataFrame, attr: str, top_n: int = 6) -> list[str]:
    if attr not in df.columns or len(df) == 0:
        return []
    return df[attr].map(_safe_category).value_counts().head(top_n).index.astype(str).tolist()


def fig_stacked_area_structure(holdings: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    attrs = [a for a in ["rarity", "type"] if a in holdings.columns]
    if not attrs or len(holdings) == 0:
        return
    fig, axes = plt.subplots(len(attrs), 1, figsize=(12, 4.4 * len(attrs)), sharex=True)
    axes = np.array([axes]).reshape(-1) if len(attrs) == 1 else np.array(axes).reshape(-1)
    for ax, attr in zip(axes, attrs):
        h = holdings.copy()
        keep = _top_categories_overall(h, attr, 6)
        h[attr] = h[attr].map(lambda x: _safe_category(x) if _safe_category(x) in keep else "Прочие")
        shares = _daily_shares(h, attr)
        pivot = shares.pivot(index="date", columns=attr, values="share").fillna(0.0).sort_index()
        cols = [c for c in keep if c in pivot.columns] + (["Прочие"] if "Прочие" in pivot.columns else [])
        pivot = pivot.reindex(columns=cols).fillna(0.0)
        ax.stackplot(pivot.index, [pivot[c].values * 100.0 for c in pivot.columns], labels=pivot.columns, alpha=0.88)
        ax.set_ylabel("Доля корзины, %")
        ax.set_title(f"Состав корзины по {attr}")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="upper left", ncol=min(4, len(pivot.columns)), fontsize=8, frameon=True)
    axes[-1].set_xlabel("Дата")
    fig.suptitle("Рисунок 4.9а — Изменение структуры отобранной корзины во времени", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "Рис_4_9а_stacked_area_структура_корзины.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_attribute_snapshot(table: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if table is None or len(table) == 0:
        return
    attrs = [a for a in ATTR_FOR_TABLE if a in table["Атрибутивный разрез"].unique()]
    n = len(attrs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11.5, max(3.0 * n, 6.0)))
    axes = np.array([axes]).reshape(-1) if n == 1 else np.array(axes).reshape(-1)
    for ax, attr in zip(axes, attrs):
        g = table[table["Атрибутивный разрез"] == attr].copy()
        g = g.sort_values("Доля в корзине, %", ascending=False).head(8).sort_values("Доля в корзине, %")
        y = np.arange(len(g))
        ax.barh(y - 0.18, g["Доля в корзине, %"], height=0.35, label="Корзина", color="tab:blue", alpha=0.85)
        ax.barh(y + 0.18, g["Доля в торговой вселенной, %"], height=0.35, label="Торговая вселенная", color="0.70", alpha=0.90)
        ax.set_yticks(y)
        ax.set_yticklabels([_short_label(x, 30) for x in g["Подгруппа"]], fontsize=8)
        ax.set_title(attr)
        ax.set_xlabel("Средняя доля, %")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(fontsize=8, frameon=True, loc="lower right")
    fig.suptitle("Рисунок 4.9б — Структура отбора предметов относительно торговой вселенной", fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "Рис_4_9б_снимок_структуры_отбора.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="chapter4_pipeline_decomposition")
    parser.add_argument("--top-pct", type=float, default=TARGET_TOP_PCT)
    parser.add_argument("--rebalance-days", type=int, default=TARGET_REBALANCE_DAYS)
    parser.add_argument("--sell-cost", type=float, default=TARGET_COST)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--pfi-max-rows", type=int, default=25000)
    parser.add_argument("--pfi-repeats", type=int, default=5)
    parser.add_argument("--skip-pfi", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg, "reports_dir")
    out_dir = reports_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_inputs(cfg)
    sim_dates = _make_test_dates(data["price_wide"], data["panel"], cfg)
    initial_capital = float(cfg["research"]["initial_capital_rub"])
    regime_thr = _regime_threshold(cfg)

    modes = ["full_pipeline", "no_regime", "no_alpha", "random_ranking"]
    equities: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}
    holdings: dict[str, pd.DataFrame] = {}
    rows = []
    for mode in modes:
        eq, tr, ho = simulate_component_strategy(
            panel=data["panel"],
            regime=data["regime"],
            price_wide=data["price_wide"],
            price_wide_mtm=data["price_wide_mtm"],
            sim_dates=sim_dates,
            initial_capital=initial_capital,
            mode=mode,
            top_pct=float(args.top_pct),
            rebalance_days=int(args.rebalance_days),
            sell_cost_rate=float(args.sell_cost),
            regime_threshold=regime_thr,
            random_seed=int(args.random_seed),
            execute_next_day=True,
        )
        equities[mode], trades[mode], holdings[mode] = eq, tr, ho
        save_table(eq, out_dir / f"decomposition_{mode}_equity.parquet")
        save_table(tr, out_dir / f"decomposition_{mode}_trades.parquet")
        save_table(ho, out_dir / f"decomposition_{mode}_holdings.parquet")
        rows.append(_summary_for_mode(mode, eq, tr, initial_capital))

    t49 = pd.DataFrame(rows)
    t49.to_csv(out_dir / "T4_9_декомпозиция_вклада_компонентов_пайплайна.csv", index=False)
    _write_md_table(t49, out_dir / "T4_9_декомпозиция_вклада_компонентов_пайплайна.md", digits=2)

    pfi_table = pd.DataFrame()
    if not args.skip_pfi:
        try:
            pfi_table = compute_permutation_importance(cfg, out_dir, max_rows=int(args.pfi_max_rows), n_repeats=int(args.pfi_repeats), random_seed=int(args.random_seed))
        except Exception as exc:
            msg = f"Permutation importance не рассчитан: {type(exc).__name__}: {exc}"
            (out_dir / "T4_10_PFI_ERROR.txt").write_text(msg, encoding="utf-8")
            print(msg)

    # T4.11 and structure figures use the full-pipeline holdings.
    panel_test = data["panel"][data["panel"]["date"].isin(sim_dates)].copy()
    t411 = build_selection_structure_table(holdings.get("full_pipeline", pd.DataFrame()), panel_test, out_dir)

    fig_equity_modes(equities, out_dir)
    fig_permutation_importance(pfi_table, out_dir)
    fig_stacked_area_structure(holdings.get("full_pipeline", pd.DataFrame()), out_dir)
    fig_attribute_snapshot(t411, out_dir)

    meta = {
        "output_dir": str(out_dir),
        "strategy": {"top_pct": float(args.top_pct), "rebalance_days": int(args.rebalance_days), "sell_cost_rate": float(args.sell_cost)},
        "period": {"start": sim_dates[0].date().isoformat(), "end": sim_dates[-1].date().isoformat(), "n_days": int(len(sim_dates))},
        "regime_threshold_used": regime_thr,
        "random_seed": int(args.random_seed),
        "pfi_repeats": int(args.pfi_repeats),
        "pfi_max_rows": int(args.pfi_max_rows),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Сохранены результаты в отдельную папку:")
    print(" ", out_dir)
    print("Таблицы:")
    for name in [
        "T4_9_декомпозиция_вклада_компонентов_пайплайна.csv",
        "T4_10_permutation_feature_importance_alpha_model.csv",
        "T4_11_структура_отбора_по_характеристикам_предметов.csv",
    ]:
        print(" ", out_dir / name)
    print("Рисунки:")
    for name in [
        "Рис_4_7_equity_curves_режимов_пайплайна.png",
        "Рис_4_8_permutation_feature_importance_alpha.png",
        "Рис_4_9а_stacked_area_структура_корзины.png",
        "Рис_4_9б_снимок_структуры_отбора.png",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
