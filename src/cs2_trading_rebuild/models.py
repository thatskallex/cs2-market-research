from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils import dump_pickle

REGIME_FEATURES = [
    "market_ret_1d_median",
    "market_ret_7d_median",
    "market_ret_30d_median",
    "market_vol_30d_median",
    "market_breadth_pos_7d",
    "market_breadth_pos_30d",
    "market_turnover_median",
    "market_price_median",
    "universe_item_count",
]

FEATURE_NUMERIC_COLS = [
    "price_rub",
    "units_sold",
    "turnover_rub",
    "ret_1d",
    "ret_7d",
    "ret_14d",
    "ret_30d",
    "ret_45d",
    "ret_60d",
    "vol_7d",
    "vol_14d",
    "vol_30d",
    "vol_45d",
    "price_mean_7d",
    "price_mean_30d",
    "price_median_7d",
    "price_median_30d",
    "units_mean_7d",
    "units_mean_30d",
    "units_median_7d",
    "units_median_30d",
    "turnover_mean_7d",
    "turnover_mean_30d",
    "turnover_median_7d",
    "turnover_median_30d",
    "drawdown_30d",
    "rebound_from_30d_low",
    "age_days",
    "market_ret_1d_median",
    "market_ret_7d_median",
    "market_ret_30d_median",
    "market_vol_30d_median",
    "market_breadth_pos_7d",
    "market_breadth_pos_30d",
    "market_turnover_median",
    "market_price_median",
    "universe_item_count",
    "type_ret_30d_median",
    "type_breadth_pos_30d",
    "type_turnover_median",
    "future_gross_return_30d",
    "future_net_return_30d",
]

MODEL_NUMERIC_FEATURES = [
    c for c in FEATURE_NUMERIC_COLS if not c.startswith("future_")
] + ["tradability_score"]

CATEGORICAL_FEATURES = [
    "quality",
    "stattrak",
    "souvenir",
    "type",
    "rarity",
    "metatype",
]


@dataclass
class TrainResult:
    model: Any
    train_report: dict[str, Any]
    predictions: pd.DataFrame
    feature_columns: dict[str, list[str]]
    target_col: str


def _split_mask(df: pd.DataFrame, start: str, end: str, embargo_days: int = 0) -> pd.Series:
    """
    Build a boolean mask for rows whose date is in [start, end - embargo_days].

    embargo_days > 0 trims the right edge of the window to prevent target leakage
    across the next fold boundary. A row dated `t` whose target uses prices
    `embargo_days` ahead has its target realized at `t + embargo_days`; if that
    falls inside the next fold, the row leaks. Default 0 preserves prior behavior
    so existing callers (e.g. train_regime_model) are unaffected.
    """
    d = pd.to_datetime(df["date"]).dt.normalize()
    end_ts = pd.Timestamp(end) - pd.Timedelta(days=embargo_days)
    return (d >= pd.Timestamp(start)) & (d <= end_ts)


def _downcast_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")


def _regime_target_col(cfg: dict[str, Any]) -> str:
    if cfg["research"].get("regime_target_col"):
        return str(cfg["research"]["regime_target_col"])
    h = int(cfg["research"]["regime_horizon_days"])
    kind = str(cfg["research"].get("regime_target_kind", "gross")).lower()
    return f"market_future_{kind}_return_{h}d"


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    y = pd.Series(y_true).dropna()
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _fit_category_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for c in CATEGORICAL_FEATURES:
        ser = train_df[c].astype("string").fillna("__MISSING__")
        vc = ser.value_counts(dropna=False)
        maps[c] = {str(v): i for i, v in enumerate(vc.index.tolist())}
    return maps


def _build_dense_matrix(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    n = len(df)
    X = np.empty((n, len(MODEL_NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)), dtype=np.float32)

    for j, c in enumerate(MODEL_NUMERIC_FEATURES):
        # Numeric NaNs are passed through unchanged. HistGradientBoostingClassifier
        # natively handles missing values: at each split it learns whether NaNs
        # should go left or right based on training gain, and routes predict-time
        # NaNs the same way. This replaces the prior train-fit median imputation
        # (Finding 12), which was applied to absolute-scale features that drift
        # over time (price, turnover, units) — items with thin history in test
        # rows were being imputed with 2023–2024 train medians, biasing them
        # toward "looks low-priced/low-turnover" relative to their actual peers.
        ser = pd.to_numeric(df[c], errors="coerce").astype("float32")
        X[:, j] = ser.to_numpy(copy=False)

    base = len(MODEL_NUMERIC_FEATURES)
    for k, c in enumerate(CATEGORICAL_FEATURES):
        mapping = cat_maps[c]
        ser = df[c].astype("string").fillna("__MISSING__")
        X[:, base + k] = ser.map(mapping).fillna(-1).astype("float32").to_numpy(copy=False)
    return X


def _resolve_alpha_target_spec(cfg: dict[str, Any]) -> tuple[str, str, int]:
    target_col = str(cfg["research"]["primary_target"])
    h = int(cfg["research"]["primary_alpha_horizon_days"])
    kind = "net" if "net" in target_col else "gross"
    base_future_col = f"future_{kind}_return_{h}d"
    return target_col, base_future_col, h


def _unique_preserve_order(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.has_duplicates:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _get_single_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def _compute_daily_quantile_labels(
    dates: pd.Series,
    target: pd.Series,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    temp = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize(), "target": pd.to_numeric(target, errors="coerce")})
    q = (
        temp.groupby("date")["target"]
        .quantile([q_low, q_high])
        .unstack()
        .rename(columns={q_low: "q_low", q_high: "q_high"})
        .reset_index()
    )
    temp = temp.merge(q, on="date", how="left")
    temp["top_label"] = (temp["target"] >= temp["q_high"]).astype("int8")
    temp["bottom_label"] = (temp["target"] <= temp["q_low"]).astype("int8")
    return temp[["q_low", "q_high", "top_label", "bottom_label"]]


def _daily_percentile_centered(dates: pd.Series, score: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize(), "score": pd.to_numeric(score, errors="coerce")})
    pct = tmp.groupby("date", group_keys=False)["score"].rank(method="average", pct=True)
    centered = (pct - 0.5) * 2.0
    return centered.astype("float32")


def _top_bottom_spread(dates: pd.Series, score: pd.Series, target: pd.Series, q: float = 0.1) -> float:
    df = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.normalize(),
        "score": pd.to_numeric(score, errors="coerce"),
        "target": pd.to_numeric(target, errors="coerce"),
    }).dropna()
    if len(df) == 0:
        return float("nan")

    spreads = []
    for _, g in df.groupby("date", sort=False):
        if len(g) < 10:
            continue
        q_hi = g["score"].quantile(1 - q)
        q_lo = g["score"].quantile(q)
        top_mean = g.loc[g["score"] >= q_hi, "target"].mean()
        bot_mean = g.loc[g["score"] <= q_lo, "target"].mean()
        if pd.notna(top_mean) and pd.notna(bot_mean):
            spreads.append(float(top_mean - bot_mean))
    return float(np.mean(spreads)) if spreads else float("nan")


def train_regime_model(market_df: pd.DataFrame, cfg: dict[str, Any]) -> TrainResult:
    target_col = _regime_target_col(cfg)
    required = ["date", target_col] + REGIME_FEATURES
    missing = [c for c in required if c not in market_df.columns]
    if missing:
        raise KeyError(f"Missing regime columns: {missing}")

    df = market_df.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    _downcast_numeric_inplace(df, REGIME_FEATURES + [target_col])
    df = df.dropna(subset=REGIME_FEATURES + [target_col]).sort_values("date").reset_index(drop=True)

    splits = cfg["splits"]
    tr = _split_mask(df, splits["train_start"], splits["train_end"])
    va = _split_mask(df, splits["val_start"], splits["val_end"])
    te = _split_mask(df, splits["test_start"], splits["test_end"])

    threshold = float(cfg["research"].get("regime_positive_threshold", 0.0))
    df["regime_target"] = (df[target_col] > threshold).astype("int8")

    threshold_source = "config"
    effective_threshold = threshold
    if df.loc[tr, "regime_target"].nunique() < 2:
        fallback_threshold = float(df.loc[tr, target_col].median())
        df["regime_target"] = (df[target_col] > fallback_threshold).astype("int8")
        threshold_source = "train_median_fallback"
        effective_threshold = fallback_threshold

    X_train = df.loc[tr, REGIME_FEATURES]
    y_train = df.loc[tr, "regime_target"]

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)

    if y_train.nunique() < 2:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        mcfg = cfg["modeling"]["regime_model"]
        clf = HistGradientBoostingClassifier(
            max_depth=int(mcfg["max_depth"]),
            min_samples_leaf=int(mcfg["min_samples_leaf"]),
            max_iter=int(mcfg["n_estimators"]),
            learning_rate=float(mcfg.get("learning_rate", 0.05)),
            l2_regularization=float(mcfg.get("l2_regularization", 0.0)),
            max_bins=int(mcfg.get("max_bins", 255)),
            random_state=int(cfg["project"]["random_seed"]),
        )

    clf.fit(X_train_imp, y_train)

    X_all = imp.transform(df[REGIME_FEATURES])
    proba = clf.predict_proba(X_all)
    df["regime_prob"] = proba[:, 1] if proba.shape[1] > 1 else 0.0
    df["regime_pred"] = (df["regime_prob"] >= 0.5).astype("int8")

    report: dict[str, Any] = {
        "target_col": target_col,
        "threshold_source": threshold_source,
        "effective_threshold": float(effective_threshold),
    }
    for label, mask in [("train", tr), ("val", va), ("test", te)]:
        part = df.loc[mask].copy()
        if len(part) == 0:
            report[label] = {"n_rows": 0}
            continue
        report[label] = {
            "n_rows": int(len(part)),
            "accuracy": float(accuracy_score(part["regime_target"], part["regime_pred"])),
            "class_count": int(part["regime_target"].nunique()),
            "positive_rate": float(part["regime_target"].mean()),
            "roc_auc": _safe_auc(part["regime_target"], part["regime_prob"]),
        }

    pred_cols = ["date", target_col, "regime_target", "regime_prob", "regime_pred"] + REGIME_FEATURES
    saved_model = {
        "type": "regime_model",
        "target_col": target_col,
        "threshold_source": threshold_source,
        "effective_threshold": float(effective_threshold),
        "imputer": imp,
        "classifier": clf,
        "features": REGIME_FEATURES,
    }
    return TrainResult(
        model=saved_model,
        train_report=report,
        predictions=df[pred_cols].copy(),
        feature_columns={"numeric": REGIME_FEATURES, "categorical": []},
        target_col="regime_target",
    )


def train_alpha_model(
    feature_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    regime_pred_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> TrainResult:
    target_col, base_future_col, alpha_horizon_days = _resolve_alpha_target_spec(cfg)

    required_feature_cols = _unique_preserve_order(
        ["date", "market_hash_name", target_col, base_future_col] + FEATURE_NUMERIC_COLS + CATEGORICAL_FEATURES
    )
    missing = [c for c in required_feature_cols if c not in feature_df.columns]
    if missing:
        raise KeyError(f"Missing alpha feature columns: {missing}")

    split_min = min(
        pd.Timestamp(cfg["splits"]["train_start"]),
        pd.Timestamp(cfg["splits"]["val_start"]),
        pd.Timestamp(cfg["splits"]["test_start"]),
    )
    split_max = max(
        pd.Timestamp(cfg["splits"]["train_end"]),
        pd.Timestamp(cfg["splits"]["val_end"]),
        pd.Timestamp(cfg["splits"]["test_end"]),
    )

    feat = feature_df.loc[:, required_feature_cols].copy()
    feat = _ensure_unique_columns(feat)
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat = feat.loc[(feat["date"] >= split_min) & (feat["date"] <= split_max)].copy()

    uni = universe_df.loc[:, ["date", "market_hash_name", "tradeable_today", "tradability_score"]].copy()
    uni["date"] = pd.to_datetime(uni["date"]).dt.normalize()
    uni = uni.loc[(uni["date"] >= split_min) & (uni["date"] <= split_max)].copy()
    uni = uni.drop_duplicates(subset=["date", "market_hash_name"], keep="last")

    reg = regime_pred_df.loc[:, ["date", "regime_prob"]].copy()
    reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
    reg = reg.loc[(reg["date"] >= split_min) & (reg["date"] <= split_max)].copy()
    reg = reg.drop_duplicates(subset=["date"], keep="last")

    work = feat.merge(uni, on=["date", "market_hash_name"], how="inner")
    work = work.loc[work["tradeable_today"].fillna(False)].copy()
    work = work.merge(reg, on="date", how="left")
    work = _ensure_unique_columns(work)

    base_future = _get_single_series(work, base_future_col).astype("float32")
    daily_base = (
        pd.DataFrame({"date": work["date"], base_future_col: base_future})
        .groupby("date", as_index=False)[base_future_col]
        .median()
        .rename(columns={base_future_col: "__tradable_baseline__"})
    )
    work = work.merge(daily_base, on="date", how="left")
    baseline = _get_single_series(work, "__tradable_baseline__").astype("float32")
    work[target_col] = (base_future - baseline).astype("float32")

    work["tradability_score"] = pd.to_numeric(work["tradability_score"], errors="coerce").fillna(0.0)
    work["regime_prob"] = pd.to_numeric(work["regime_prob"], errors="coerce").fillna(0.5)

    _downcast_numeric_inplace(
        work,
        _unique_preserve_order(FEATURE_NUMERIC_COLS + ["tradability_score", target_col, "regime_prob"]),
    )
    for c in CATEGORICAL_FEATURES:
        if c in work.columns:
            work[c] = work[c].astype("string").fillna("__MISSING__")

    work = work.dropna(subset=[target_col]).sort_values(["date", "market_hash_name"]).reset_index(drop=True)

    quantile_low = float(cfg["modeling"]["alpha_model"].get("rank_quantile_low", 0.30))
    quantile_high = float(cfg["modeling"]["alpha_model"].get("rank_quantile_high", 0.70))
    q_info = _compute_daily_quantile_labels(work["date"], work[target_col], quantile_low, quantile_high)
    work = pd.concat([work.reset_index(drop=True), q_info.reset_index(drop=True)], axis=1)

    splits = cfg["splits"]
    # Purge the last `alpha_horizon_days` of train and val to prevent target
    # leakage across fold boundaries. A row dated `t` carries a target realized
    # at `t + alpha_horizon_days`; without this trim, the last 30 train rows
    # have targets that resolve inside validation, and the same applies at the
    # val/test seam. Test gets no right-edge embargo: rows whose targets reach
    # past the data end have NaN targets and are already dropped by the
    # `dropna(subset=[target_col])` call above.
    tr = _split_mask(work, splits["train_start"], splits["train_end"], embargo_days=alpha_horizon_days)
    va = _split_mask(work, splits["val_start"], splits["val_end"], embargo_days=alpha_horizon_days)
    te = _split_mask(work, splits["test_start"], splits["test_end"])

    train_df = work.loc[tr].copy()
    if len(train_df) == 0:
        raise ValueError("Alpha training set is empty. Check universe filters and split dates.")

    # Numeric imputation removed (Finding 12). The previous scheme fit medians on
    # train_df only and then applied those train-era medians to val and test rows
    # via .fillna(). For absolute-scale features (price_median_30d,
    # turnover_median_30d, units_median_30d) which drift over time as the CS2
    # economy grows, items with thin history at predict time were imputed with
    # 2023–2024 train medians while their actual peer-group values had moved
    # higher, biasing the model toward "low price / low turnover / likely loser"
    # for those rows. HistGradientBoostingClassifier handles missing values
    # natively: at each split it learns a direction for NaN rows from training
    # gain, and routes predict-time NaNs the same way. This is strictly cleaner
    # than imputation — no drift problem at the root, and the model can use
    # "is missing" as informative when it actually is.
    cat_maps = _fit_category_maps(train_df)
    X_train = _build_dense_matrix(train_df, cat_maps)

    mcfg = cfg["modeling"]["alpha_model"]

    common_params = dict(
        loss="log_loss",
        max_depth=int(mcfg.get("cls_max_depth", mcfg.get("max_depth", 4))),
        min_samples_leaf=int(mcfg.get("cls_min_samples_leaf", mcfg.get("min_samples_leaf", 60))),
        max_iter=int(mcfg.get("cls_n_estimators", mcfg.get("n_estimators", 300))),
        learning_rate=float(mcfg.get("cls_learning_rate", mcfg.get("learning_rate", 0.03))),
        l2_regularization=float(mcfg.get("cls_l2_regularization", mcfg.get("l2_regularization", 1.0))),
        max_bins=int(mcfg.get("max_bins", 255)),
    )

    top_model = HistGradientBoostingClassifier(
        **common_params,
        random_state=int(cfg["project"]["random_seed"]),
    )
    bottom_model = HistGradientBoostingClassifier(
        **common_params,
        random_state=int(cfg["project"]["random_seed"]) + 1,
    )

    # Sample weighting removed (Findings 9 + 11). Previously each row's loss
    # weight was turnover_w * |target|/median(|target|) * class_balance, with
    # each term clipped — combined ratio up to ~24x between heaviest and
    # lightest training row. Two problems with that scheme:
    #   1. |target|/median(|target|) mixed the target's magnitude into the loss
    #      weight of a classifier whose label is itself a daily quantile of
    #      that same target. Rows the model could already separate got up to 3x
    #      weight — soft label leakage that inflated train AUC without teaching
    #      anything that generalized.
    #   2. The reweighted training distribution did not match the uniformly
    #      weighted predict-time panel. Tree boosters fit on heavily reweighted
    #      data produce miscalibrated probabilities out of sample.
    # Step 05's universe filter already enforces liquidity, so the turnover
    # weight was duplicating that selection inside the loss with no benefit.
    # With 30/70 daily quantile labels the positive base rate is ~30% by
    # construction, so log_loss without class balancing is well-behaved and
    # predicted probabilities now reflect the true base rate.
    y_top = train_df["top_label"].astype("int8").to_numpy(copy=False)
    y_bottom = train_df["bottom_label"].astype("int8").to_numpy(copy=False)
    top_model.fit(X_train, y_top)
    bottom_model.fit(X_train, y_bottom)

    block_size = 100_000
    prob_top = np.empty(len(work), dtype=np.float32)
    prob_bottom = np.empty(len(work), dtype=np.float32)
    for start in range(0, len(work), block_size):
        stop = min(start + block_size, len(work))
        X_block = _build_dense_matrix(work.iloc[start:stop], cat_maps)
        prob_top[start:stop] = top_model.predict_proba(X_block)[:, 1].astype("float32")
        prob_bottom[start:stop] = bottom_model.predict_proba(X_block)[:, 1].astype("float32")

    work["alpha_prob_top"] = prob_top
    work["alpha_prob_bottom"] = prob_bottom
    work["alpha_rank_score_raw"] = (work["alpha_prob_top"] - work["alpha_prob_bottom"]).astype("float32")
    work["alpha_pred"] = _daily_percentile_centered(work["date"], work["alpha_rank_score_raw"])

    report: dict[str, Any] = {
        "target_col": target_col,
        "base_future_col": base_future_col,
        "baseline_col": "__tradable_baseline__",
        "alpha_mode": "two_head_rank_classifier",
        "embargo_days": int(alpha_horizon_days),
        "sample_weighting_mode": "uniform",
        "numeric_imputation_mode": "native_nan",
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "top_rate_train": float(y_top.mean()),
        "bottom_rate_train": float(y_bottom.mean()),
        "signal_semantics": "alpha_pred is a centered daily rank score, not a return forecast",
    }

    for label, mask in [("train", tr), ("val", va), ("test", te)]:
        part = work.loc[mask].copy()
        if len(part) == 0:
            report[label] = {"n_rows": 0}
            continue
        y_true = _get_single_series(part, target_col)
        score = _get_single_series(part, "alpha_rank_score_raw")
        score_centered = _get_single_series(part, "alpha_pred")
        top_lbl = pd.to_numeric(part["top_label"], errors="coerce")
        bottom_lbl = pd.to_numeric(part["bottom_label"], errors="coerce")
        report[label] = {
            "n_rows": int(len(part)),
            "mean_target": float(y_true.mean()),
            "mean_score_raw": float(score.mean()),
            "mean_score_centered": float(score_centered.mean()),
            "hit_rate_true_gt_0": float((y_true > 0).mean()),
            "hit_rate_pred_gt_0": float((score_centered > 0).mean()),
            "direction_accuracy": float(((y_true > 0) == (score_centered > 0)).mean()),
            "roc_auc_up": _safe_auc((y_true > 0).astype(int), score),
            "roc_auc_top_bucket": _safe_auc(top_lbl, part["alpha_prob_top"]),
            "roc_auc_bottom_bucket": _safe_auc(bottom_lbl, part["alpha_prob_bottom"]),
            "score_target_spearman": float(pd.Series(score).corr(pd.Series(y_true), method="spearman")) if len(part) > 1 else None,
            "top_bottom_spread_10pct": _top_bottom_spread(part["date"], score, y_true, q=0.10),
        }

    pred_cols = [
        "date",
        "market_hash_name",
        target_col,
        "alpha_pred",
        "alpha_rank_score_raw",
        "alpha_prob_top",
        "alpha_prob_bottom",
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
    pred_cols = [c for c in pred_cols if c in work.columns]

    saved_model = {
        "type": "alpha_model_two_head_rank_classifier",
        "category_maps": cat_maps,
        "top_classifier": top_model,
        "bottom_classifier": bottom_model,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": target_col,
        "base_future_col": base_future_col,
        "baseline_col": "__tradable_baseline__",
        "alpha_mode": "two_head_rank_classifier",
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "signal_semantics": "alpha_pred is a centered daily rank score, not a return forecast",
    }
    return TrainResult(
        model=saved_model,
        train_report=report,
        predictions=work[pred_cols].copy(),
        feature_columns={"numeric": MODEL_NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        target_col=target_col,
    )


def save_train_result(base_dir: Path, model_name: str, result: TrainResult) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    dump_pickle(base_dir / f"{model_name}.pkl", result.model)
