from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_UNIVERSE_INPUT_COLS = [
    "date",
    "market_hash_name",
    "price_rub",
    "units_sold",
    "turnover_rub",
    "price_median_30d",
    "units_median_30d",
    "turnover_median_30d",
    "vol_30d",
    "quality",
    "stattrak",
    "souvenir",
    "type",
    "rarity",
    "metatype",
    "obs_count_30d",
]

OUTPUT_COLS = [
    "date",
    "market_hash_name",
    "tradeable_today",
    "tradability_score",
    "universe_reasons",
    "price_rub",
    "units_sold",
    "turnover_rub",
    "price_median_30d",
    "units_median_30d",
    "turnover_median_30d",
    "vol_30d",
    "quality",
    "stattrak",
    "souvenir",
    "type",
    "rarity",
    "metatype",
]


def _append_reason(reason_arr: np.ndarray, mask: np.ndarray, label: str) -> None:
    if not mask.any():
        return
    current = reason_arr[mask]
    reason_arr[mask] = np.where(current == "", label, current + ";" + label)


def build_dynamic_tradeable_universe(feature_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    uni_cfg = cfg["universe"]

    df = feature_df.loc[:, REQUIRED_UNIVERSE_INPUT_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    for col in ["market_hash_name", "quality", "type", "rarity", "metatype"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "stattrak" in df.columns:
        df["stattrak"] = df["stattrak"].astype("boolean")
    if "souvenir" in df.columns:
        df["souvenir"] = df["souvenir"].astype("boolean")

    for col in [
        "price_rub",
        "units_sold",
        "turnover_rub",
        "price_median_30d",
        "units_median_30d",
        "turnover_median_30d",
        "vol_30d",
        "obs_count_30d",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    exclude_keywords = [kw.lower() for kw in uni_cfg.get("exclude_name_keywords", [])]
    if exclude_keywords:
        name_lower = df["market_hash_name"].fillna("").str.lower()
        keyword_block = pd.Series(False, index=df.index)
        for kw in exclude_keywords:
            keyword_block |= name_lower.str.contains(kw, regex=False)
    else:
        keyword_block = pd.Series(False, index=df.index)

    low_recent_obs = df["obs_count_30d"] < float(uni_cfg["min_recent_obs"])
    low_price = df["price_median_30d"] < float(uni_cfg["min_price_rub"])
    low_units = df["units_median_30d"] < float(uni_cfg["min_units_sold_30d_median"])
    low_turnover = df["turnover_median_30d"] < float(uni_cfg["min_turnover_rub_30d_median"])
    low_vol = df["vol_30d"] < float(uni_cfg["min_vol_30d"])
    high_vol = df["vol_30d"] > float(uni_cfg["max_vol_30d"])
    excluded_metatype = df["metatype"].isin(uni_cfg.get("exclude_metatypes", []))
    excluded_type = df["type"].isin(uni_cfg.get("exclude_types", []))
    excluded_keyword = keyword_block.fillna(False)

    tradeable = (
        (~low_recent_obs) &
        (~low_price) &
        (~low_units) &
        (~low_turnover) &
        (~low_vol) &
        (~high_vol) &
        (~excluded_metatype) &
        (~excluded_type) &
        (~excluded_keyword)
    )

    by_date = df.groupby("date", sort=False, observed=True)
    df["price_rank_date"] = by_date["price_median_30d"].rank(method="average", pct=True)
    df["units_rank_date"] = by_date["units_median_30d"].rank(method="average", pct=True)
    df["turnover_rank_date"] = by_date["turnover_median_30d"].rank(method="average", pct=True)
    df["vol_rank_date"] = by_date["vol_30d"].rank(method="average", pct=True)

    vol_score = (1.0 - (df["vol_rank_date"].fillna(0.5) - 0.5).abs() * 2.0).clip(lower=0.0)

    # Heavier emphasis on price and turnover to push out cheap / non-liquid names.
    df["tradability_score"] = (
        0.30 * df["price_rank_date"].fillna(0.0) +
        0.15 * df["units_rank_date"].fillna(0.0) +
        0.45 * df["turnover_rank_date"].fillna(0.0) +
        0.10 * vol_score
    ).astype("float32")

    reason_arr = np.full(len(df), "", dtype=object)
    _append_reason(reason_arr, low_recent_obs.to_numpy(), "low_recent_obs")
    _append_reason(reason_arr, low_price.to_numpy(), "low_price")
    _append_reason(reason_arr, low_units.to_numpy(), "low_units")
    _append_reason(reason_arr, low_turnover.to_numpy(), "low_turnover")
    _append_reason(reason_arr, low_vol.to_numpy(), "low_vol")
    _append_reason(reason_arr, high_vol.to_numpy(), "high_vol")
    _append_reason(reason_arr, excluded_metatype.to_numpy(), "excluded_metatype")
    _append_reason(reason_arr, excluded_type.to_numpy(), "excluded_type")
    _append_reason(reason_arr, excluded_keyword.to_numpy(), "excluded_keyword")

    df["universe_reasons"] = pd.Series(reason_arr, index=df.index, dtype="string")
    df["tradeable_today"] = tradeable.fillna(False).astype(bool)

    out = df.loc[:, OUTPUT_COLS].copy()
    out = out.sort_values(["date", "tradeable_today", "tradability_score", "market_hash_name"], ascending=[True, False, False, True]).reset_index(drop=True)
    return out
