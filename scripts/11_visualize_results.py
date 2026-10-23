#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SEARCH_ROOTS = [SCRIPT_PATH.parent, SCRIPT_PATH.parent / 'cs2_trading_rebuild', SCRIPT_PATH.parent.parent]
for root in SEARCH_ROOTS:
    src = root / 'src'
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from cs2_trading_rebuild.config import load_config, resolve_path
except Exception as e:
    raise ImportError(
        'Could not import cs2_trading_rebuild. Place this script in the rebuilt repo root or add repo/src to PYTHONPATH.'
    ) from e


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported file type: {path}')


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return equity / roll_max - 1.0


def make_equity_plot(eq: pd.DataFrame, out_dir: Path) -> dict:
    if eq.empty:
        return {}
    eq = eq.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    dd = compute_drawdown(eq['equity_rub'].astype(float))

    plt.figure(figsize=(10, 4.5))
    plt.plot(eq['date'], eq['equity_rub'])
    plt.title('Portfolio equity curve')
    plt.xlabel('Date')
    plt.ylabel('Equity (RUB)')
    save_fig(out_dir / 'equity_curve.png')

    plt.figure(figsize=(10, 3.5))
    plt.plot(eq['date'], dd)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    save_fig(out_dir / 'drawdown.png')

    eq['daily_return'] = eq['equity_rub'].pct_change()
    plt.figure(figsize=(8, 4.5))
    plt.hist(eq['daily_return'].dropna(), bins=40)
    plt.title('Distribution of daily portfolio returns')
    plt.xlabel('Daily return')
    plt.ylabel('Count')
    save_fig(out_dir / 'daily_return_histogram.png')

    return {
        'max_drawdown_pct': float(dd.min() * 100.0),
        'final_equity_rub': float(eq['equity_rub'].iloc[-1]),
        'roi_pct': float((eq['equity_rub'].iloc[-1] / eq['equity_rub'].iloc[0] - 1.0) * 100.0) if len(eq) > 1 and eq['equity_rub'].iloc[0] != 0 else None,
        'n_days': int(len(eq)),
    }


def make_trade_plots(trades: pd.DataFrame, out_dir: Path) -> dict:
    if trades.empty:
        return {}
    trades = trades.copy()
    trades['date'] = pd.to_datetime(trades['date'])
    trades['month'] = trades['date'].dt.to_period('M').astype(str)

    monthly = trades.groupby(['month', 'side']).size().unstack(fill_value=0)
    monthly.plot(kind='bar', figsize=(10, 4.5))
    plt.title('Monthly trade counts')
    plt.xlabel('Month')
    plt.ylabel('Trades')
    save_fig(out_dir / 'monthly_trades.png')

    sells = trades.loc[trades['side'] == 'sell'].copy()
    if 'realized_pnl_rub' in sells.columns and len(sells):
        pnl_monthly = sells.groupby('month')['realized_pnl_rub'].sum()
        plt.figure(figsize=(10, 4.5))
        pnl_monthly.plot(kind='bar')
        plt.title('Monthly realized PnL from sells')
        plt.xlabel('Month')
        plt.ylabel('Realized PnL (RUB)')
        save_fig(out_dir / 'monthly_realized_pnl.png')

        if 'market_hash_name' in sells.columns:
            pnl_by_item = sells.groupby('market_hash_name')['realized_pnl_rub'].sum().sort_values(ascending=False)
            top = pnl_by_item.head(15)
            if len(top):
                plt.figure(figsize=(10, 6))
                top.sort_values().plot(kind='barh')
                plt.title('Top items by realized PnL')
                plt.xlabel('Realized PnL (RUB)')
                plt.ylabel('Item')
                save_fig(out_dir / 'top_items_realized_pnl.png')

                bottom = pnl_by_item.tail(15)
                plt.figure(figsize=(10, 6))
                bottom.sort_values().plot(kind='barh')
                plt.title('Worst items by realized PnL')
                plt.xlabel('Realized PnL (RUB)')
                plt.ylabel('Item')
                save_fig(out_dir / 'worst_items_realized_pnl.png')

    return {
        'trade_count': int(len(trades)),
        'buy_count': int((trades['side'] == 'buy').sum()) if 'side' in trades.columns else None,
        'sell_count': int((trades['side'] == 'sell').sum()) if 'side' in trades.columns else None,
    }


def make_holdings_plot(holdings: pd.DataFrame, out_dir: Path) -> dict:
    if holdings.empty:
        return {}
    holdings = holdings.copy()
    holdings['date'] = pd.to_datetime(holdings['date'])
    daily_count = holdings.groupby('date')['market_hash_name'].nunique()
    plt.figure(figsize=(10, 4.5))
    plt.plot(daily_count.index, daily_count.values)
    plt.title('Open positions over time')
    plt.xlabel('Date')
    plt.ylabel('Open positions')
    save_fig(out_dir / 'open_positions_over_time.png')
    return {
        'avg_open_positions': float(daily_count.mean()),
        'max_open_positions': int(daily_count.max()),
    }


def make_regime_plot(regime_pred: pd.DataFrame, policy: dict, out_dir: Path) -> dict:
    if regime_pred.empty:
        return {}
    regime_pred = regime_pred.copy()
    regime_pred['date'] = pd.to_datetime(regime_pred['date'])
    plt.figure(figsize=(10, 4.5))
    plt.plot(regime_pred['date'], regime_pred['regime_prob'])
    if policy and 'regime_prob_threshold' in policy:
        plt.axhline(float(policy['regime_prob_threshold']), linestyle='--')
    plt.title('Market regime probability over time')
    plt.xlabel('Date')
    plt.ylabel('Regime probability')
    save_fig(out_dir / 'regime_probability.png')
    return {
        'mean_regime_prob': float(regime_pred['regime_prob'].mean()),
        'share_above_threshold': float((regime_pred['regime_prob'] >= float(policy.get('regime_prob_threshold', 0.5))).mean()) if policy else None,
    }


def make_universe_plot(universe: pd.DataFrame, out_dir: Path) -> dict:
    if universe.empty:
        return {}
    universe = universe.copy()
    universe['date'] = pd.to_datetime(universe['date'])
    daily = universe.groupby('date').agg(
        tradeable_items=('tradeable_today', 'sum'),
        avg_tradability=('tradability_score', 'mean'),
    )
    plt.figure(figsize=(10, 4.5))
    plt.plot(daily.index, daily['tradeable_items'])
    plt.title('Tradable universe breadth over time')
    plt.xlabel('Date')
    plt.ylabel('Tradeable items')
    save_fig(out_dir / 'universe_breadth.png')
    return {
        'avg_tradeable_items': float(daily['tradeable_items'].mean()),
        'median_tradeable_items': float(daily['tradeable_items'].median()),
    }


def make_alpha_plots(policy_panel: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    if policy_panel.empty or 'alpha_pred' not in policy_panel.columns:
        return {}
    df = policy_panel.copy()
    df['date'] = pd.to_datetime(df['date'])
    target_col = cfg['research']['primary_target']
    stats = {}

    if target_col in df.columns:
        usable = df.dropna(subset=['alpha_pred', target_col]).copy()
        if len(usable) >= 20:
            # Scatter sample for readability.
            sample = usable.sample(min(3000, len(usable)), random_state=42)
            plt.figure(figsize=(6, 6))
            plt.scatter(sample['alpha_pred'], sample[target_col], s=8, alpha=0.6)
            plt.title('Predicted alpha vs realized target')
            plt.xlabel('Predicted alpha')
            plt.ylabel(f'Realized {target_col}')
            save_fig(out_dir / 'alpha_scatter.png')

            usable['alpha_decile'] = pd.qcut(usable['alpha_pred'], q=min(10, usable['alpha_pred'].nunique()), duplicates='drop')
            dec = usable.groupby('alpha_decile').agg(
                mean_pred=('alpha_pred', 'mean'),
                mean_realized=(target_col, 'mean'),
                obs=('alpha_pred', 'size'),
            ).reset_index()
            plt.figure(figsize=(10, 4.5))
            plt.plot(range(len(dec)), dec['mean_pred'], marker='o', label='Mean predicted alpha')
            plt.plot(range(len(dec)), dec['mean_realized'], marker='o', label='Mean realized target')
            plt.title('Alpha calibration by prediction bucket')
            plt.xlabel('Prediction bucket (low → high)')
            plt.ylabel('Return')
            plt.legend()
            save_fig(out_dir / 'alpha_calibration.png')

            topk_daily = usable.sort_values(['date', 'alpha_pred'], ascending=[True, False]).groupby('date').head(5)
            realized_topk = topk_daily.groupby('date')[target_col].mean().reset_index()
            plt.figure(figsize=(10, 4.5))
            plt.plot(realized_topk['date'], realized_topk[target_col])
            plt.title('Realized target of daily top-5 alpha names')
            plt.xlabel('Date')
            plt.ylabel(f'Mean {target_col}')
            save_fig(out_dir / 'top5_realized_alpha_timeseries.png')

            stats = {
                'corr_pred_to_realized': float(usable[['alpha_pred', target_col]].corr().iloc[0, 1]),
                'mean_realized_top_decile': float(dec['mean_realized'].iloc[-1]),
                'mean_realized_bottom_decile': float(dec['mean_realized'].iloc[0]),
            }
    return stats


def make_policy_search_plots(grid: pd.DataFrame, out_dir: Path) -> dict:
    if grid.empty:
        return {}
    stats = {}
    if {'entry_alpha_threshold', 'top_k', 'final_equity_rub'}.issubset(grid.columns):
        pivot = grid.pivot_table(index='entry_alpha_threshold', columns='top_k', values='final_equity_rub', aggfunc='max')
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.values, aspect='auto')
        plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
        plt.title('Best validation final equity by entry threshold and top-K')
        plt.xlabel('top_k')
        plt.ylabel('entry_alpha_threshold')
        plt.colorbar(label='Final equity (RUB)')
        save_fig(out_dir / 'policy_search_heatmap.png')
        best = grid.sort_values('selection_score', ascending=False).iloc[0].to_dict()
        stats['best_validation_final_equity_rub'] = float(best.get('final_equity_rub', np.nan))
        stats['best_selection_score'] = float(best.get('selection_score', np.nan))
    return stats


def make_live_plot(live: pd.DataFrame, out_dir: Path) -> dict:
    if live.empty:
        return {'n_live_candidates': 0}
    live = live.copy()
    if 'recommended_rank' in live.columns:
        live = live.sort_values('recommended_rank')
    plt.figure(figsize=(10, 5))
    plt.barh(live['market_hash_name'].astype(str), live['alpha_pred'].astype(float))
    plt.title('Current live candidates by alpha score')
    plt.xlabel('Predicted alpha')
    plt.ylabel('Item')
    save_fig(out_dir / 'live_candidates_alpha.png')
    return {'n_live_candidates': int(len(live))}


def render_markdown(summary: dict, out_path: Path) -> None:
    lines = ['# Results interpretation guide', '']
    lines.append('## First files to open')
    lines.append('')
    lines.append('- `backtests/portfolio_backtest_test_summary.json`')
    lines.append('- `backtests/portfolio_backtest_test_equity.parquet`')
    lines.append('- `backtests/portfolio_backtest_test_trades.parquet`')
    lines.append('- `models/policy_validation_summary.json`')
    lines.append('- `models/policy_grid_search.parquet`')
    lines.append('- `models/regime_report.json`')
    lines.append('- `models/alpha_report.json`')
    lines.append('- `live/live_candidates.parquet`')
    lines.append('')
    lines.append('## How to read the outputs')
    lines.append('')
    lines.append('Start with validation policy performance, then test equity curve, then trade behavior, then alpha calibration.')
    lines.append('A good result is not just high final equity. You also want reasonable drawdown, regular trade activity, and a top-alpha bucket that clearly outperforms the bottom bucket.')
    lines.append('')
    lines.append('## Summary metrics')
    lines.append('')
    for section, payload in summary.items():
        lines.append(f'### {section}')
        if not payload:
            lines.append('No data found.')
            lines.append('')
            continue
        for k, v in payload.items():
            lines.append(f'- **{k}**: {v}')
        lines.append('')
    lines.append('## Main interpretation questions')
    lines.append('')
    lines.append('1. Does validation policy selection translate to test equity growth?')
    lines.append('2. Is drawdown acceptable relative to return?')
    lines.append('3. Are profits concentrated in a few items, or broadly distributed?')
    lines.append('4. Does the regime filter actually reduce bad periods?')
    lines.append('5. Does higher predicted alpha correspond to higher realized alpha?')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--outdir', default=None, help='Optional custom output directory for plots and summary files.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(cfg['_root'])
    out_dir = Path(args.outdir) if args.outdir else repo_root / 'visuals'
    ensure_dir(out_dir)

    paths = {
        'universe': resolve_path(cfg, 'universe_daily'),
        'models_dir': resolve_path(cfg, 'models_dir'),
        'signals_dir': resolve_path(cfg, 'signals_dir'),
        'backtests_dir': resolve_path(cfg, 'backtests_dir'),
        'live_dir': resolve_path(cfg, 'live_dir'),
    }

    universe = load_any(paths['universe'])
    regime_pred = load_any(paths['models_dir'] / 'regime_predictions.parquet')
    alpha_pred = load_any(paths['models_dir'] / 'alpha_predictions.parquet')
    policy_panel = load_any(paths['signals_dir'] / 'policy_panel.parquet')
    policy_grid = load_any(paths['models_dir'] / 'policy_grid_search.parquet')
    equity = load_any(paths['backtests_dir'] / 'portfolio_backtest_test_equity.parquet')
    trades = load_any(paths['backtests_dir'] / 'portfolio_backtest_test_trades.parquet')
    holdings = load_any(paths['backtests_dir'] / 'portfolio_backtest_test_holdings.parquet')
    live = load_any(paths['live_dir'] / 'live_candidates.parquet')

    policy = {}
    policy_path = paths['models_dir'] / 'policy.json'
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding='utf-8'))

    summary = {
        'equity': make_equity_plot(equity, out_dir),
        'trades': make_trade_plots(trades, out_dir),
        'holdings': make_holdings_plot(holdings, out_dir),
        'regime': make_regime_plot(regime_pred, policy, out_dir),
        'universe': make_universe_plot(universe, out_dir),
        'alpha': make_alpha_plots(policy_panel, cfg, out_dir),
        'policy_search': make_policy_search_plots(policy_grid, out_dir),
        'live': make_live_plot(live, out_dir),
    }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    render_markdown(summary, out_dir / 'RESULTS_GUIDE.md')
    print(f'Visualization pack saved to: {out_dir}')


if __name__ == '__main__':
    main()
