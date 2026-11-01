# 2025 BTC15M SPOT_PROXY research backtest

Run from repo root with Python 3.12:

```sh
python -m pip install --only-binary=:all: -r requirements.txt
python scripts/backtest_2025.py
```

The script downloads 2024-12 and 2025 BTCUSDT 1m candles from Binance public monthly archives, verifies published checksums, trains a **new model entirely on 2024-12** and tests the 525600 minutes of 2025. It intentionally DOES NOT load `models/hgb_btc15m_prod.joblib`, which was trained in 2026.

Scenarios: OHLC paths OLHC and OHLC; synthetic spread 2c/slippage 0c or spread 4c/slippage 1c; fee 2c per executed side, single contract, 1 entry per 15-minute cycle.

Reports: `reports/backtest-2025/summary.json` and per-scenario trade CSVs.

**Research-only:** Synthetic normal-CDF YES/NO quotes and BTCUSDT spot-proxy outcomes are NOT official Kalshi KXBTC15M historical quotes, BRTI, fill volume or settlement. Do not interpret the output as live-trading returns. Full-year KXBTC15M history does not cover all of 2025; verify actual availability before claims.
