from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sqlite3

import pandas as pd
from sqlalchemy import create_engine, text

from .utils import to_bool


REQUIRED_MASTER_COLS = [
    "item_id",
    "market_hash_name",
    "market_hash_name_raw",
    "market_hash_name_new_raw",
    "market_hash_name_join_key",
    "quality",
    "stattrak",
    "souvenir",
    "type",
    "rarity",
    "metatype",
    "metadata_source_file",
]

REQUIRED_DAILY_COLS = [
    "timestamp",
    "price_rub",
    "units_sold",
]


def normalize_market_hash_name(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    s = str(value)
    s = s.replace("™", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else pd.NA


def _rename_using_map(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    reverse = {v: k for k, v in mapping.items() if v}
    missing = [src for src in reverse if src not in df.columns]
    if missing:
        raise KeyError(f"Missing expected source columns: {missing}")
    return df.rename(columns=reverse)


def export_daily_market_from_db(cfg: dict[str, Any]) -> pd.DataFrame:
    db_cfg = cfg["source"]["db"]
    col_map = cfg["source"]["db_column_map"]

    if db_cfg.get("sqlite_path"):
        db_path = Path(db_cfg["sqlite_path"])
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {db_path}")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            raw = pd.read_sql_query(db_cfg["query"], conn)
        finally:
            conn.close()
    else:
        engine = create_engine(db_cfg["connection_uri"])
        with engine.connect() as conn:
            raw = pd.read_sql(text(db_cfg["query"]), conn)

    df = _rename_using_map(raw, col_map)

    if "market_hash_name" not in df.columns:
        df["market_hash_name"] = pd.NA
    if "resolved_market_hash_name" not in df.columns:
        df["resolved_market_hash_name"] = pd.NA
    if "item_id" not in df.columns:
        df["item_id"] = pd.NA

    for col in REQUIRED_DAILY_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing required daily column after renaming: {col}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df["date"] = df["timestamp"].dt.normalize()
    df["price_rub"] = pd.to_numeric(df["price_rub"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0.0)

    df["market_hash_name"] = df["market_hash_name"].astype("string")
    df["resolved_market_hash_name"] = df["resolved_market_hash_name"].astype("string")
    df["market_hash_name_db_original"] = df["market_hash_name"].astype("string")

    df["market_hash_name_join_key"] = df["market_hash_name"].map(normalize_market_hash_name).astype("string")
    df["resolved_market_hash_name_join_key"] = df["resolved_market_hash_name"].map(normalize_market_hash_name).astype("string")

    # Hard guardrail: collapse to exactly one row per raw item x day before any joins.
    sort_cols = ["resolved_market_hash_name", "market_hash_name", "timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    grouped = (
        df.groupby(["date", "resolved_market_hash_name_join_key", "market_hash_name_join_key"], dropna=False, as_index=False)
          .agg(
              market_hash_name=("market_hash_name", "last"),
              resolved_market_hash_name=("resolved_market_hash_name", "last"),
              price_rub=("price_rub", "last"),
              units_sold=("units_sold", "sum"),
          )
    )
    grouped["market_hash_name"] = grouped["market_hash_name"].astype("string")
    grouped["resolved_market_hash_name"] = grouped["resolved_market_hash_name"].astype("string")
    grouped["market_hash_name_db_original"] = grouped["market_hash_name"].astype("string")
    grouped["market_hash_name_join_key"] = grouped["market_hash_name_join_key"].astype("string")
    grouped["resolved_market_hash_name_join_key"] = grouped["resolved_market_hash_name_join_key"].astype("string")
    return grouped


def build_item_master_from_excels(cfg: dict[str, Any]) -> pd.DataFrame:
    meta_cfg = cfg["source"]["metadata"]
    col_map = meta_cfg["column_map"]

    if meta_cfg.get("file_path"):
        files = [Path(meta_cfg["file_path"])]
    elif meta_cfg.get("dir"):
        meta_dir = Path(meta_cfg["dir"])
        files = sorted(meta_dir.glob(meta_cfg.get("glob", "*.xlsx")))
    else:
        raise FileNotFoundError("Metadata path not configured.")

    if not files:
        raise FileNotFoundError("No metadata files found.")

    bool_cfg = cfg["source"].get("booleans", {})
    true_values = set(bool_cfg.get("true_values", [True, "true", "TRUE", "1", 1, "yes", "Yes"]))
    false_values = set(bool_cfg.get("false_values", [False, "false", "FALSE", "0", 0, "no", "No"]))

    frames: list[pd.DataFrame] = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Metadata file not found: {fp}")
        df = pd.read_excel(fp, sheet_name=meta_cfg.get("sheet_name", 0))
        df = _rename_using_map(df, col_map)
        keep_cols = list(col_map.keys())
        df = df[keep_cols].copy()
        df["metadata_source_file"] = fp.name
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    if "item_id" in master.columns:
        master["item_id"] = master["item_id"].astype("string").str.strip()

    for c in ["market_hash_name", "market_hash_name_new", "quality", "type", "rarity", "metatype"]:
        if c in master.columns:
            master[c] = master[c].astype("string").str.strip().replace({"": pd.NA})

    for c in ["stattrak", "souvenir"]:
        if c in master.columns:
            master[c] = master[c].map(lambda x: to_bool(x, true_values, false_values)).astype("boolean")
        else:
            master[c] = pd.Series(pd.NA, index=master.index, dtype="boolean")

    master["market_hash_name_raw"] = master.get("market_hash_name", pd.Series(pd.NA, index=master.index)).astype("string")
    master["market_hash_name_new_raw"] = master.get("market_hash_name_new", pd.Series(pd.NA, index=master.index)).astype("string")

    preferred = master["market_hash_name_new_raw"].where(master["market_hash_name_new_raw"].notna(), master["market_hash_name_raw"])
    master["market_hash_name"] = preferred.astype("string")
    master["market_hash_name_join_key"] = master["market_hash_name"].map(normalize_market_hash_name).astype("string")

    for col in REQUIRED_MASTER_COLS:
        if col not in master.columns:
            master[col] = pd.NA

    return master[REQUIRED_MASTER_COLS]


def _build_metadata_alias_lookup(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [
        "item_id", "market_hash_name", "quality", "stattrak", "souvenir", "type",
        "rarity", "metatype", "metadata_source_file",
    ]
    alias_frames = []
    for alias_source, alias_col in [
        ("metadata_market_hash_name", "market_hash_name_raw"),
        ("metadata_market_hash_name_new", "market_hash_name_new_raw"),
        ("metadata_canonical", "market_hash_name"),
    ]:
        cols = base_cols.copy()
        if alias_col not in cols:
            cols.append(alias_col)
        tmp = master[cols].copy()
        if alias_col == "market_hash_name":
            tmp["alias_raw"] = tmp["market_hash_name"]
        else:
            tmp = tmp.rename(columns={alias_col: "alias_raw"})
        tmp["alias_source"] = alias_source
        alias_frames.append(tmp)

    aliases = pd.concat(alias_frames, ignore_index=True)
    aliases["alias_raw"] = aliases["alias_raw"].astype("string").str.strip().replace({"": pd.NA})
    aliases["alias_join_key"] = aliases["alias_raw"].map(normalize_market_hash_name).astype("string")
    aliases = aliases.loc[aliases["alias_join_key"].notna()].copy()
    aliases = aliases.drop_duplicates(subset=["market_hash_name", "alias_join_key"])

    alias_counts = (
        aliases.groupby("alias_join_key")["market_hash_name"]
        .nunique(dropna=True)
        .reset_index(name="n_items")
    )
    conflicts = alias_counts.loc[alias_counts["n_items"] > 1, ["alias_join_key"]].copy()

    if not conflicts.empty:
        conflict_keys = set(conflicts["alias_join_key"])
        alias_conflicts = aliases.loc[aliases["alias_join_key"].isin(conflict_keys)].copy()
        aliases = aliases.loc[~aliases["alias_join_key"].isin(conflict_keys)].copy()
    else:
        alias_conflicts = aliases.iloc[0:0].copy()

    alias_lookup = (
        aliases.sort_values(["alias_join_key", "alias_source", "market_hash_name"])
        .drop_duplicates(subset=["alias_join_key"], keep="first")
        .reset_index(drop=True)
    )
    return alias_lookup, alias_conflicts


def join_master_panel(daily: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    alias_lookup, alias_conflicts = _build_metadata_alias_lookup(master)

    resolved_map = alias_lookup.add_prefix("resolved_")
    raw_map = alias_lookup.add_prefix("raw_")

    base = daily.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()

    rename_map = {}
    if "market_hash_name" in base.columns:
        rename_map["market_hash_name"] = "market_hash_name_db"
    if "resolved_market_hash_name" in base.columns:
        rename_map["resolved_market_hash_name"] = "resolved_market_hash_name_db"
    if rename_map:
        base = base.rename(columns=rename_map)

    joined = base.merge(
        resolved_map,
        left_on="resolved_market_hash_name_join_key",
        right_on="resolved_alias_join_key",
        how="left",
    )
    joined = joined.merge(
        raw_map,
        left_on="market_hash_name_join_key",
        right_on="raw_alias_join_key",
        how="left",
    )

    joined["market_hash_name"] = joined["resolved_market_hash_name"].where(
        joined["resolved_market_hash_name"].notna(),
        joined["raw_market_hash_name"],
    )
    joined["market_hash_name"] = joined["market_hash_name"].where(
        joined["market_hash_name"].notna(),
        joined["resolved_market_hash_name_db"],
    )
    joined["market_hash_name"] = joined["market_hash_name"].where(
        joined["market_hash_name"].notna(),
        joined["market_hash_name_db"],
    )

    for col in ["quality", "stattrak", "souvenir", "type", "rarity", "metatype", "item_id", "metadata_source_file"]:
        joined[col] = joined[f"resolved_{col}"].where(
            joined[f"resolved_{col}"].notna(),
            joined[f"raw_{col}"],
        )

    keep_cols = [
        "date",
        "market_hash_name",
        "price_rub",
        "units_sold",
        "quality",
        "stattrak",
        "souvenir",
        "type",
        "rarity",
        "metatype",
        "item_id",
        "metadata_source_file",
        "market_hash_name_db",
        "resolved_market_hash_name_db",
    ]
    out = joined[keep_cols].copy()

    # Second hard guardrail: collapse to one canonical item x day after metadata joins.
    # This prevents alias fan-out from creating pseudo-intraday rows.
    out = (
        out.sort_values(["market_hash_name", "date"])
           .groupby(["date", "market_hash_name"], as_index=False, dropna=False)
           .agg(
               price_rub=("price_rub", "last"),
               units_sold=("units_sold", "sum"),
               quality=("quality", "first"),
               stattrak=("stattrak", "first"),
               souvenir=("souvenir", "first"),
               type=("type", "first"),
               rarity=("rarity", "first"),
               metatype=("metatype", "first"),
               item_id=("item_id", "first"),
               metadata_source_file=("metadata_source_file", "first"),
               market_hash_name_db=("market_hash_name_db", "last"),
               resolved_market_hash_name_db=("resolved_market_hash_name_db", "last"),
           )
           .sort_values(["market_hash_name", "date"])
           .reset_index(drop=True)
    )

    report = {
        "n_daily_rows": int(len(daily)),
        "n_master_rows": int(len(master)),
        "n_panel_rows": int(len(out)),
        "n_resolved_alias_conflicts_dropped": int(len(alias_conflicts)),
        "n_unique_items_panel": int(out["market_hash_name"].nunique(dropna=True)),
        "metadata_match_share": float(out["type"].notna().mean()) if len(out) else 0.0,
    }
    return out, report


def build_item_coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("market_hash_name", dropna=False)
    out = g.agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_days=("date", "nunique"),
        median_price_rub=("price_rub", "median"),
        median_units_sold=("units_sold", "median"),
        type=("type", "first"),
        metatype=("metatype", "first"),
    ).reset_index()
    out["coverage_days"] = (pd.to_datetime(out["end_date"]) - pd.to_datetime(out["start_date"])).dt.days + 1
    return out.sort_values(["n_days", "median_units_sold"], ascending=[False, False]).reset_index(drop=True)


def validate_daily_market(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(daily)),
        "n_unique_items_raw": int(daily["resolved_market_hash_name"].nunique(dropna=True)) if "resolved_market_hash_name" in daily.columns else None,
        "date_min": str(pd.to_datetime(daily["date"]).min()) if len(daily) else None,
        "date_max": str(pd.to_datetime(daily["date"]).max()) if len(daily) else None,
        "missing_price_share": float(daily["price_rub"].isna().mean()) if len(daily) else 0.0,
    }


def validate_item_master(master: pd.DataFrame) -> dict[str, Any]:
    alias_lookup, alias_conflicts = _build_metadata_alias_lookup(master)
    return {
        "n_rows": int(len(master)),
        "n_unique_market_hash_name": int(master["market_hash_name"].nunique(dropna=True)),
        "n_alias_lookup_rows": int(len(alias_lookup)),
        "n_alias_conflicts_rows": int(len(alias_conflicts)),
    }
