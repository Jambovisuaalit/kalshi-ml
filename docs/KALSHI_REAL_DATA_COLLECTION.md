# Collect genuine KXBTC15M market data (read-only)

## Historical collection

```bash
python scripts/collect_kalshi_real_data.py --mode archive --start-date 2025-12-01 --end-date 2025-12-31 --max-markets 100 --output reports/kalshi_real_data_2025
```

Public Kalshi API, no API key, no orders. Saves raw archive market pages, settled market records (official result), historical one-minute bid/ask/trade-price candles and public trade tape, plus a cutoff/manifest with timestamps. Verify completeness and API response schema before making claims. Archive pagination and coverage depend on Kalshi's current historical API.

```bash
python scripts/collect_kalshi_real_data.py --mode live --output reports/kalshi_live
```

Live mode captures the current public market object including quoted bid/ask fields; it does **not** capture full order-book depth or pre-settlement model predictions. For high-frequency order-book replay, capture an authorized live WebSocket feed separately.

## Validation is deliberately gated

The companion `scripts/validate_kalshi_market.py` requires **real** contemporaneous `model_p_yes` values, actual YES/NO quotes and verified settlement times/results in one CSV. This collector does not manufacture predictions for old data and does not auto-join incomplete data. A later adapter must join data only with strict ticker/time alignment, enforce model training cutoff, and refuse absent bid/ask observations.

For retrospective 2025 validation, predictions must be generated with a model trained strictly before the tested observations; the current 2026-trained production model is invalid for 2025 backfills. For prospective validation, log frozen model version, inference time, quote receipt time and source market ticker together before settlement.

**Historical 1-minute candles are not historical order books or guaranteed executable quotes.** Fee tier, order-book depth, slippage, latency, and real fills remain separate evidence requirements. The validation report is hypothetical entry-to-settlement performance, not actual executed PnL.
