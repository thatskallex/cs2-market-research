# data

the source database and item metadata are not included in this repository.

the research used daily item-level prices and trading activity from the steam marketplace ecosystem, plus descriptive attributes such as item quality, rarity, type, and special item flags.

to rerun the code, provide your own appropriately licensed dataset matching the sqlite schema and excel column mapping in `configs/cs2_trading_rebuilt.yaml`.

the `.gitignore` intentionally excludes local databases, spreadsheets, and generated features. selected aggregated historical results appear under `results/`.
