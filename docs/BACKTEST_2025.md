# 2025 causal SPOT_PROXY backtest v2

This is a **research simulation**, not observed Kalshi historical order-book data or realized returns.

## Run

```bash
python -m pip install --only-binary=:all: -r requirements.txt
python scripts/backtest_2025.py --full
```

Downloads checksum-verified Binance BTCUSDT 1-minute bars (December 2024 + 2025), trains HGB only on December 2024 and holds it fixed for 2025. 2026 production artifacts are not used.

## Look-ahead prevention

Decisions are at minute start, using only the prior **completed** minute close and lagged volatility. Active-minute OHLC extremes are never used. No intraminute stop claims are possible with this data resolution. Contract strike and settlement remain **spot proxies**, not official Kalshi BRTI.

## Experiment matrix

48 configurations: Delta-Stop on/off × take-profit 75/80/85/88 cents × hard stop 10/14/20 cents × (spread 2 cents/slippage 0 or spread 4 cents/slippage 1). Assumed fee 2 cents per side. One contract per cycle.

Two parallel ledgers: unconstrained signal-level PnL (may be negative; not a real account) and funded $1,000 execution (skips unaffordable entries). Both are simulations. Compare signal-level edge without conflating insolvency.

Outputs in `reports/backtest-2025/`:
- `scenario_summary.csv` — overall PnL, win rate, Profit Factor, global drawdown, funded cash skips.
- `monthly_metrics.csv` — monthly PnL, trades, Profit Factor, drawdown (global-peak drawdown observed within month).
- `unconstrained_trades.csv` — audit-level exit reasons and per-trade PnL.
- `summary.json` — manifest, assumptions, and scenario summaries.

**Limitations:** 1-minute data cannot validate second-level Delta-Stop; no Kalshi book, queue position, actual fees, executable fills, or official settlement. Parameter-grid results are exploratory: do not pick a winning combination on 2025 and then call that same period an independent out-of-sample test. Confirm on a later untouched holdout and actual Kalshi history before live use.
