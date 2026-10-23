from __future__ import annotations

import numpy as np
import pandas as pd

BASE_META_COLS = ["quality", "stattrak", "souvenir", "type", "rarity", "metatype"]


def _safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.astype(float).pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def _rolling_count(s: pd.Series, window: int) -> pd.Series:
    return s.notna().astype("int16").rolling(window, min_periods=1).sum()


def _collapse_daily_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Safety collapse to one row per item-day.

    Even if upstream already did a day-level collapse, this protects the feature
    store from alias fan-out or accidental duplicates created during later joins.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    agg = {
        "price_rub": "last",
        "units_sold": "sum",
    }
    for c in BASE_META_COLS + ["item_id", "metadata_source_file"]:
        if c in df.columns:
            agg[c] = "first"

    out = (
        df.sort_values(["market_hash_name", "date"])
          .groupby(["date", "market_hash_name"], as_index=False, dropna=False)
          .agg(agg)
          .sort_values(["market_hash_name", "date"])
          .reset_index(drop=True)
    )
    return out


def _reindex_to_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex each item to a continuous daily date grid from its first observation
    to its last (Finding 4).

    The previous pipeline ran rolling/shift/pct_change over the raw observation
    sequence, so for thin items `pct_change(30)` covered far more than 30
    calendar days and `shift(-30)` looked further than 30 calendar days ahead.
    After this reindex, every position-based operation downstream is calendar-
    based by construction.

    Fill semantics:
      - price_rub: forward-fill within each item. A missing day inherits the
        last observed market price — the standard convention for prices.
      - units_sold: 0 on missing days. No trade = 0 units sold (and turnover
        derived from price * units therefore = 0).
      - metadata columns (quality, type, rarity, ...): forward + back-fill
        within each item. These are item-constants and shouldn't drift.

    Output is sorted by (market_hash_name, date) with a 0..N-1 index.
    """
    df = df.sort_values(["market_hash_name", "date"]).reset_index(drop=True)

    bounds = df.groupby("market_hash_name", sort=False, as_index=False).agg(
        start=("date", "min"), end=("date", "max"),
    )
    bounds["n_days"] = ((bounds["end"] - bounds["start"]).dt.days + 1).astype(np.int64)

    items = bounds["market_hash_name"].to_numpy()
    n_days = bounds["n_days"].to_numpy()
    starts = bounds["start"].to_numpy().astype("datetime64[D]")

    full_items = np.repeat(items, n_days)
    offsets = np.concatenate([np.arange(n, dtype=np.int64) for n in n_days])
    full_dates = np.repeat(starts, n_days) + offsets.astype("timedelta64[D]")

    full_grid = pd.DataFrame({
        "market_hash_name": full_items,
        "date": pd.to_datetime(full_dates),
    })

    out = full_grid.merge(df, on=["market_hash_name", "date"], how="left")
    out = out.sort_values(["market_hash_name", "date"]).reset_index(drop=True)

    out["price_rub"] = out.groupby("market_hash_name", sort=False)["price_rub"].ffill()
    out["units_sold"] = out["units_sold"].fillna(0.0)

    meta_cols = [c for c in BASE_META_COLS + ["item_id", "metadata_source_file"] if c in out.columns]
    if meta_cols:
        g = out.groupby("market_hash_name", sort=False)
        for c in meta_cols:
            out[c] = g[c].ffill().bfill()

    return out


def build_feature_store(panel: pd.DataFrame, sell_cost_rate: float, alpha_horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build item-level features and a full-market daily series.

    Important:
    - This function still emits a broad-market series for monitoring and fallback.
    - The alpha and regime training code will rebuild *tradable-universe* targets
      later, because using the full market baseline polluted targets with cheap
      and illiquid items.
    """
    df = _collapse_daily_panel(panel)
    # Reindex each item to a continuous daily grid (Finding 4). After this,
    # the rolling/pct_change/shift operations below are calendar-based instead
    # of observation-based.
    df = _reindex_to_daily_grid(df)
    df["price_rub"] = pd.to_numeric(df["price_rub"], errors="coerce").astype("float32")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0.0).astype("float32")
    df = df.sort_values(["market_hash_name", "date"]).reset_index(drop=True)

    for col in BASE_META_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    g = df.groupby("market_hash_name", group_keys=False, sort=False)

    df["log_price"] = np.log(df["price_rub"].clip(lower=1e-8)).astype("float32")
    df["logret_1d"] = g["log_price"].diff().astype("float32")
    df["ret_1d"] = g["price_rub"].pct_change().astype("float32")

    for p in [7, 14, 30, 45, 60]:
        df[f"ret_{p}d"] = g["price_rub"].transform(lambda s, p=p: _safe_pct_change(s, p)).astype("float32")

    for w in [7, 14, 30, 45]:
        minp = max(5, w // 2)
        df[f"vol_{w}d"] = g["logret_1d"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).std()).astype("float32")
        df[f"price_mean_{w}d"] = g["price_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).mean()).astype("float32")
        df[f"price_median_{w}d"] = g["price_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).median()).astype("float32")
        df[f"units_mean_{w}d"] = g["units_sold"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).mean()).astype("float32")
        df[f"units_median_{w}d"] = g["units_sold"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).median()).astype("float32")
        df[f"obs_count_{w}d"] = g["price_rub"].transform(lambda s, w=w: _rolling_count(s, w)).astype("float32")

    df["turnover_rub"] = (df["price_rub"] * df["units_sold"]).astype("float32")
    for w in [7, 14, 30, 45]:
        minp = max(5, w // 2)
        df[f"turnover_mean_{w}d"] = g["turnover_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).mean()).astype("float32")
        df[f"turnover_median_{w}d"] = g["turnover_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).median()).astype("float32")
        df[f"price_max_{w}d"] = g["price_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).max()).astype("float32")
        df[f"price_min_{w}d"] = g["price_rub"].transform(lambda s, w=w, minp=minp: s.rolling(w, min_periods=minp).min()).astype("float32")

    df["drawdown_30d"] = ((df["price_rub"] / df["price_max_30d"]) - 1.0).astype("float32")
    df["rebound_from_30d_low"] = ((df["price_rub"] / df["price_min_30d"]) - 1.0).astype("float32")
    df["age_days"] = (g.cumcount() + 1).astype("float32")

    # Broad-market series for monitoring and as a fallback only.
    daily_market = (
        df.groupby("date", as_index=False)
          .agg(
              market_ret_1d_median=("ret_1d", "median"),
              market_ret_7d_median=("ret_7d", "median"),
              market_ret_30d_median=("ret_30d", "median"),
              market_vol_30d_median=("vol_30d", "median"),
              market_breadth_pos_7d=("ret_7d", lambda s: float((s > 0).mean()) if len(s) else np.nan),
              market_breadth_pos_30d=("ret_30d", lambda s: float((s > 0).mean()) if len(s) else np.nan),
              market_turnover_median=("turnover_rub", "median"),
              market_price_median=("price_rub", "median"),
              universe_item_count=("market_hash_name", "nunique"),
          )
          .sort_values("date")
          .reset_index(drop=True)
    )
    for c in [c for c in daily_market.columns if c != "date"]:
        daily_market[c] = pd.to_numeric(daily_market[c], errors="coerce").astype("float32")

    if "type" in df.columns:
        type_daily = (
            df.groupby(["date", "type"], as_index=False)
              .agg(
                  type_ret_30d_median=("ret_30d", "median"),
                  type_breadth_pos_30d=("ret_30d", lambda s: float((s > 0).mean()) if len(s) else np.nan),
                  type_turnover_median=("turnover_rub", "median"),
              )
        )
        for c in [c for c in type_daily.columns if c not in {"date", "type"}]:
            type_daily[c] = pd.to_numeric(type_daily[c], errors="coerce").astype("float32")
    else:
        type_daily = pd.DataFrame(columns=["date", "type", "type_ret_30d_median", "type_breadth_pos_30d", "type_turnover_median"])

    market_series = daily_market.copy().sort_values("date").reset_index(drop=True)
    for h in alpha_horizons:
        future_market_price = market_series["market_price_median"].shift(-h).astype("float32")
        market_series[f"market_future_gross_return_{h}d"] = ((future_market_price / market_series["market_price_median"]) - 1.0).astype("float32")
        market_series[f"market_future_net_return_{h}d"] = (((1.0 - sell_cost_rate) * future_market_price / market_series["market_price_median"]) - 1.0).astype("float32")

    df = df.merge(daily_market, on="date", how="left")
    if len(type_daily):
        df = df.merge(type_daily, on=["date", "type"], how="left")
    else:
        df["type_ret_30d_median"] = np.nan
        df["type_breadth_pos_30d"] = np.nan
        df["type_turnover_median"] = np.nan

    for h in alpha_horizons:
        future_price = g["price_rub"].shift(-h).astype("float32")
        df[f"future_gross_return_{h}d"] = ((future_price / df["price_rub"]) - 1.0).astype("float32")
        df[f"future_net_return_{h}d"] = (((1.0 - sell_cost_rate) * future_price / df["price_rub"]) - 1.0).astype("float32")

    market_return_cols = ["date"]
    for h in alpha_horizons:
        market_return_cols.extend([f"market_future_gross_return_{h}d", f"market_future_net_return_{h}d"])
    df = df.merge(market_series[market_return_cols], on="date", how="left")

    for h in alpha_horizons:
        df[f"future_excess_gross_return_{h}d"] = (df[f"future_gross_return_{h}d"] - df[f"market_future_gross_return_{h}d"]).astype("float32")
        df[f"future_excess_net_return_{h}d"] = (df[f"future_net_return_{h}d"] - df[f"market_future_net_return_{h}d"]).astype("float32")
        df[f"future_hit_gross_return_{h}d"] = (df[f"future_gross_return_{h}d"] > 0).astype("float32")
        df[f"future_hit_net_return_{h}d"] = (df[f"future_net_return_{h}d"] > 0).astype("float32")

    return df, market_series
