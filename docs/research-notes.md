# research notes

## research question

can cs2 item characteristics and market history help identify relatively stronger-performing assets, and does that signal remain useful when translated into portfolio decisions with trading costs?

## data

- daily prices and trading volumes from the steam marketplace ecosystem, joined to item-level metadata
- history discussed in the thesis: october 2023 to march 2026
- modeling split: training through february 2025, validation from march to august 2025, and testing from september 2025 to february 2026
- a dynamic item universe using rolling filters for observation history, price, trading activity, turnover, volatility, and item category

## modeling and evaluation

- a market-regime signal and a cross-sectional ranking model built using gradient boosting
- a 30-day relative-return target for the ranking signal
- long-only portfolios with a seven-day minimum holding period
- evaluation with simulated returns, drawdowns, turnover, rebalancing intervals, and sensitivity to selling costs
- comparisons against passive holding of the eligible asset universe

## reading the results

the headline returns in the main readme come from the thesis's **full historical-window** portfolio simulations at 10% selling costs. they should not be interpreted as results from a separate holdout period.

`results/tables/strategies_test_only_10pct.csv` contains the corresponding test-only performance tables. the repository also preserves selected exploratory outputs under `results/experimental_policy/`; these are separate from the thesis tables and should not be combined into one comparison.

simulated performance is sensitive to platform rules, selling fees, liquidity, execution timing, and unpredictable game updates. observed prices do not guarantee executable trades, and the research does not report live investing performance.

## source

the code and figures come from the author's original bachelor's thesis project. the images were extracted directly from the thesis, and the signed declaration and personal documents have not been included.
