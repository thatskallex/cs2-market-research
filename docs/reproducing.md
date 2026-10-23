# setup and data

this repository preserves the source scripts and selected results from the original research project. it is intended primarily as a research archive; the market database and local working environment are not distributed.

## requirements

- python and the libraries in [`requirements.txt`](../requirements.txt)
- a compatible sqlite database of steam marketplace price history
- an item metadata workbook with the expected columns
- adequate memory and disk space for the daily item-level feature store

install the listed dependencies in a virtual environment:

```bash
python -m venv .venv
# activate the environment for your operating system
python -m pip install -r requirements.txt
```

the configuration expects input data at:

```text
data/external/steamwebapi_history.db
data/external/metadata.xlsx
```

if needed, change the data paths in `configs/cs2_trading_rebuilt.yaml` to match your own sources.

## original script sequence

scripts `01` through `10` cover extraction, dataset construction, feature engineering, modeling, portfolio evaluation, and export. later scripts contain research diagnostics and thesis analysis.

```text
01_extract_daily_market.py
02_build_item_master.py
03_join_master_panel.py
04_build_feature_store.py
05_build_dynamic_tradeable_universe.py
06_train_market_regime_model.py
07_train_cross_sectional_alpha_model.py
08_calibrate_trade_policy.py
09_walkforward_portfolio_backtest.py
10_export_live_candidates.py
```

the scripts reflect the original project structure and depend on the research inputs and intermediate artifacts used during development. the saved aggregate results and selected equity series can be examined directly without the original database.

## results files

- `results/tables/` — portfolio performance and model diagnostics
- `results/figures/` — selected illustrations from the thesis
- `results/equity/` — selected parquet equity time series
- `results/experimental_policy/` — separate exploratory modeling outputs

use `pandas` with `pyarrow` to read parquet files.
