from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BACKTEST_PANEL_COLS = [
    "date",
    "market_hash_name",
    "alpha_pred",
    "tradeable_today",
    "tradability_score",
    "regime_prob",
    "price_rub",
    "turnover_median_30d",
]


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    daily_holdings: pd.DataFrame
    summary: dict[str, Any]


def _roi_pct(initial_capital: float, final_equity: float) -> float:
    if initial_capital <= 0:
        return np.nan
    return float((final_equity - initial_capital) / initial_capital * 100.0)


def _sharpe_ratio(returns: pd.Series, annualization_days: int = 365) -> float:
    r = returns.dropna()
    if len(r) < 2 or float(r.std(ddof=1)) == 0.0:
        return np.nan
    return float(np.sqrt(annualization_days) * r.mean() / r.std(ddof=1))


def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date","side","market_hash_name","price_rub","units","gross_value_rub","net_value_rub","realized_pnl_rub","reason"])


def _empty_holdings_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date","market_hash_name","units","entry_date","entry_price","mark_price","holding_days","alpha_pred"])


def _empty_equity_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date","cash_rub","market_value_rub","equity_rub","open_positions","regime_prob","daily_return"])


def run_portfolio_backtest(
    panel_df: pd.DataFrame,
    policy: dict[str, Any],
    initial_capital_rub: float,
    sell_cost_rate: float,
    execute_next_day: bool = True,
    min_holding_days: int = 7,
    annualization_days: int = 365,
    store_details: bool = True,
) -> PortfolioBacktestResult:
    missing = [c for c in BACKTEST_PANEL_COLS if c not in panel_df.columns]
    if missing:
        raise KeyError(f"Missing required backtest columns: {missing}")

    data = panel_df.loc[:, BACKTEST_PANEL_COLS].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["tradeable_today"] = data["tradeable_today"].fillna(False).astype(bool)
    for c in ["alpha_pred", "tradability_score", "regime_prob", "price_rub", "turnover_median_30d"]:
        data[c] = pd.to_numeric(data[c], errors="coerce").astype("float32")
    data = data.sort_values(["date", "market_hash_name"]).reset_index(drop=True)

    if len(data) == 0:
        return PortfolioBacktestResult(
            equity_curve=_empty_equity_df(),
            trades=_empty_trades_df(),
            daily_holdings=_empty_holdings_df(),
            summary={
                "initial_capital_rub": float(initial_capital_rub),
                "final_equity_rub": float(initial_capital_rub),
                "roi_pct": 0.0,
                "sharpe": np.nan,
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "avg_open_positions": 0.0,
                "max_open_positions": 0,
            },
        )

    group_indices = data.groupby("date", sort=True).indices
    dates = sorted(group_indices.keys())

    cash = float(initial_capital_rub)
    positions: dict[str, dict[str, Any]] = {}
    pending_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}

    trades_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    equity_track: list[dict[str, Any]] = []
    trade_count = 0
    buy_count = 0
    sell_count = 0

    def queue_order(exec_date: pd.Timestamp, order: dict[str, Any]) -> None:
        pending_orders.setdefault(exec_date, []).append(order)

    for i, today in enumerate(dates):
        rows = data.iloc[group_indices[today]]
        price_map = rows.set_index("market_hash_name")["price_rub"].to_dict()
        alpha_map = rows.set_index("market_hash_name")["alpha_pred"].to_dict()
        regime_prob = float(rows["regime_prob"].iloc[0]) if len(rows) else 0.0

        todays_orders = pending_orders.pop(today, [])
        if todays_orders:
            sells = [o for o in todays_orders if o["side"] == "sell"]
            buys = [o for o in todays_orders if o["side"] == "buy"]

            for order in sells:
                item = order["market_hash_name"]
                pos = positions.get(item)
                if pos is None or item not in price_map or not np.isfinite(price_map[item]):
                    continue
                price = float(price_map[item])
                proceeds = pos["units"] * price * (1.0 - sell_cost_rate)
                realized_pnl = proceeds - pos["cost_basis"]
                cash += proceeds
                positions.pop(item, None)
                trade_count += 1
                sell_count += 1
                if store_details:
                    trades_rows.append(
                        {
                            "date": today,
                            "side": "sell",
                            "market_hash_name": item,
                            "price_rub": price,
                            "units": int(pos["units"]),
                            "gross_value_rub": float(pos["units"] * price),
                            "net_value_rub": float(proceeds),
                            "realized_pnl_rub": float(realized_pnl),
                            "reason": order.get("reason", ""),
                        }
                    )

            buys = sorted(buys, key=lambda o: float(o.get("score", -1e18)), reverse=True)
            for order in buys:
                item = order["market_hash_name"]
                if item in positions or item not in price_map or not np.isfinite(price_map[item]):
                    continue
                price = float(price_map[item])
                if price <= 0:
                    continue
                budget = float(order["budget_rub"])
                units_cap = int(order["max_units"])
                affordable_units = int(cash // price)
                units = min(units_cap, affordable_units, int(budget // price))
                if units < 1:
                    continue
                cost = units * price
                cash -= cost
                positions[item] = {
                    "units": units,
                    "entry_date": today,
                    "entry_price": price,
                    "cost_basis": cost,
                }
                trade_count += 1
                buy_count += 1
                if store_details:
                    trades_rows.append(
                        {
                            "date": today,
                            "side": "buy",
                            "market_hash_name": item,
                            "price_rub": price,
                            "units": int(units),
                            "gross_value_rub": float(cost),
                            "net_value_rub": float(cost),
                            "realized_pnl_rub": np.nan,
                            "reason": order.get("reason", ""),
                        }
                    )

        market_value = 0.0
        for item, pos in positions.items():
            px = price_map.get(item)
            if px is not None and np.isfinite(px):
                market_value += pos["units"] * float(px)

        equity = cash + market_value
        equity_track.append({"date": today, "equity_rub": equity, "open_positions": int(len(positions)), "regime_prob": regime_prob})
        if store_details:
            equity_rows.append({"date": today, "cash_rub": cash, "market_value_rub": market_value, "equity_rub": equity, "open_positions": int(len(positions)), "regime_prob": regime_prob})
            for item, pos in positions.items():
                px = price_map.get(item)
                if px is not None and np.isfinite(px):
                    holdings_rows.append(
                        {
                            "date": today,
                            "market_hash_name": item,
                            "units": int(pos["units"]),
                            "entry_date": pos["entry_date"],
                            "entry_price": float(pos["entry_price"]),
                            "mark_price": float(px),
                            "holding_days": int((pd.Timestamp(today) - pd.Timestamp(pos["entry_date"])).days),
                            "alpha_pred": float(alpha_map.get(item, np.nan)),
                        }
                    )

        if execute_next_day:
            if i + 1 >= len(dates):
                continue
            exec_date = pd.Timestamp(dates[i + 1])
        else:
            exec_date = pd.Timestamp(today)

        regime_on = regime_prob >= float(policy["regime_prob_threshold"])
        candidate_rows = rows.loc[rows["tradeable_today"] & rows["alpha_pred"].notna()].sort_values(
            ["alpha_pred", "tradability_score"],
            ascending=[False, False],
            kind="mergesort",
        )

        candidate_names = set(
            candidate_rows.loc[candidate_rows["alpha_pred"] >= float(policy["entry_alpha_threshold"]), "market_hash_name"]
            .head(int(policy["top_k"]))
            .astype(str)
        )

        for item, pos in list(positions.items()):
            current_alpha = float(alpha_map.get(item, np.nan))
            holding_days = int((pd.Timestamp(today) - pd.Timestamp(pos["entry_date"])).days)
            # Steam marketplace 7-day trade lock: an item bought on day t cannot
            # be sold until day t + min_holding_days regardless of any other
            # exit signal. Skip exit evaluation entirely until the lock has
            # cleared. After the lock clears, the live conditions below are
            # checked each day; signals that fired during the lock are not
            # accumulated.
            if holding_days < int(min_holding_days):
                continue
            # Removed: `or (item not in candidate_names)`. That clause caused a
            # 30-day-horizon strategy to behave as a 1-day rotator: any
            # position that fell out of today's daily top-K (which churns
            # constantly with K=2 over ~58 candidates) was force-sold the next
            # day, eating the 10% sell cost without giving the alpha signal
            # time to express. Exits now come only from regime turning off,
            # hitting max_holding_days, or alpha falling below
            # exit_alpha_threshold — matching the documented design.
            should_sell = (
                (not regime_on)
                or (holding_days >= int(policy["max_holding_days"]))
                or (np.isfinite(current_alpha) and current_alpha < float(policy["exit_alpha_threshold"]))
            )
            if should_sell:
                queue_order(exec_date, {"side": "sell", "market_hash_name": item, "reason": "policy_exit"})

        if not regime_on:
            continue

        open_names = set(positions.keys())
        new_candidates = candidate_rows.loc[
            candidate_rows["market_hash_name"].isin(candidate_names)
            & ~candidate_rows["market_hash_name"].isin(open_names)
        ].copy()
        if len(new_candidates) == 0:
            continue

        capital_for_new = max(cash, 0.0)
        if capital_for_new <= 0:
            continue

        per_name_target = min(
            float(policy["max_weight_per_item"]) * max(equity, 0.0),
            capital_for_new / max(len(new_candidates), 1),
        )

        for _, row in new_candidates.iterrows():
            capacity_rub = float(row.get("turnover_median_30d", 0.0)) * float(policy["max_daily_turnover_fraction"])
            budget = min(per_name_target, capacity_rub)
            price = float(row["price_rub"]) if np.isfinite(row["price_rub"]) else np.nan
            if not np.isfinite(price) or price <= 0 or budget < price:
                continue
            max_units = int(max(0.0, budget // price))
            if max_units < 1:
                continue
            queue_order(
                exec_date,
                {
                    "side": "buy",
                    "market_hash_name": str(row["market_hash_name"]),
                    "budget_rub": float(budget),
                    "max_units": int(max_units),
                    "score": float(row["alpha_pred"]),
                    "reason": "policy_entry",
                },
            )

    eq = pd.DataFrame(equity_track)
    eq["daily_return"] = eq["equity_rub"].pct_change()

    summary = {
        "initial_capital_rub": float(initial_capital_rub),
        "final_equity_rub": float(eq["equity_rub"].iloc[-1]) if len(eq) else float(initial_capital_rub),
        "roi_pct": _roi_pct(float(initial_capital_rub), float(eq["equity_rub"].iloc[-1]) if len(eq) else float(initial_capital_rub)),
        "sharpe": _sharpe_ratio(eq["daily_return"], annualization_days=annualization_days) if len(eq) else np.nan,
        "trade_count": int(trade_count),
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "avg_open_positions": float(eq["open_positions"].mean()) if len(eq) else 0.0,
        "max_open_positions": int(eq["open_positions"].max()) if len(eq) else 0,
        "min_holding_days": int(min_holding_days),
    }

    if store_details:
        equity_df = pd.DataFrame(equity_rows) if len(equity_rows) else _empty_equity_df()
        if len(equity_df):
            equity_df["daily_return"] = equity_df["equity_rub"].pct_change()
        trades_df = pd.DataFrame(trades_rows) if len(trades_rows) else _empty_trades_df()
        holdings_df = pd.DataFrame(holdings_rows) if len(holdings_rows) else _empty_holdings_df()
    else:
        equity_df = _empty_equity_df()
        trades_df = _empty_trades_df()
        holdings_df = _empty_holdings_df()

    return PortfolioBacktestResult(equity_curve=equity_df, trades=trades_df, daily_holdings=holdings_df, summary=summary)
