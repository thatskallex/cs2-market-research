from __future__ import annotations

import pandas as pd


LIVE_CANDIDATE_INPUT_COLS = [
    "date",
    "market_hash_name",
    "alpha_pred",
    "regime_prob",
    "tradeable_today",
    "tradability_score",
    "price_rub",
    "units_sold",
    "turnover_rub",
    "price_median_30d",
    "units_median_30d",
    "turnover_median_30d",
    "quality",
    "type",
    "rarity",
    "metatype",
]


def export_live_candidates(policy_panel: pd.DataFrame, policy: dict) -> pd.DataFrame:
    missing = [c for c in LIVE_CANDIDATE_INPUT_COLS if c not in policy_panel.columns]
    if missing:
        raise KeyError(f"Missing required live candidate columns: {missing}")

    df = policy_panel.loc[:, LIVE_CANDIDATE_INPUT_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    latest_date = df["date"].max()
    latest = df.loc[df["date"] == latest_date].copy()
    if len(latest) == 0:
        return latest

    regime_prob = float(latest["regime_prob"].iloc[0]) if "regime_prob" in latest.columns else 0.0
    if regime_prob < float(policy["regime_prob_threshold"]):
        return latest.iloc[0:0].copy()

    out = latest.loc[
        latest["tradeable_today"].fillna(False)
        & (pd.to_numeric(latest["alpha_pred"], errors="coerce") >= float(policy["entry_alpha_threshold"]))
    ].copy()

    out = out.sort_values(["alpha_pred", "tradability_score"], ascending=[False, False]).head(int(policy["top_k"])).reset_index(drop=True)

    if len(out) == 0:
        return out

    out["regime_prob_at_signal"] = regime_prob
    out["recommended_rank"] = range(1, len(out) + 1)
    out["suggested_weight_cap"] = float(policy["max_weight_per_item"])

    keep_cols = [
        "date",
        "market_hash_name",
        "recommended_rank",
        "alpha_pred",
        "regime_prob_at_signal",
        "tradability_score",
        "price_rub",
        "units_sold",
        "turnover_rub",
        "price_median_30d",
        "units_median_30d",
        "turnover_median_30d",
        "quality",
        "type",
        "rarity",
        "metatype",
        "suggested_weight_cap",
    ]
    return out[keep_cols].copy()